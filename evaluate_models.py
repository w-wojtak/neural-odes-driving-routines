import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, f1_score
from models import ODERNN_Time, ODERNN_Power, ODERNN_POI, ODERNN_Drivers  # Import trained model classes

# Load the dataset
df_test = pd.read_csv("Datasets/DL_HMI_01_with_POI.csv")

# Normalize within a day (same as training)
df_test["TimeDay"] = df_test["TimeDay"] / (24 * 60)
df_test["TimeDay"] += df_test.groupby("DayNum").cumcount() * 1e-4

# Ensure sorted order within each day
df_test = df_test.sort_values(by=["DayNum", "TimeDay"]).reset_index(drop=True)

num_poi_classes = df_test["POI"].nunique()
driver_labels = df_test.columns[df_test.columns.get_loc("Temperature") + 1 : df_test.columns.get_loc("POI")].tolist()
num_driver_features = len(driver_labels)

df_test = df_test.apply(pd.to_numeric, errors='coerce').fillna(0)

# Initialize models with correct input sizes
hidden_dim = 16
model_time = ODERNN_Time(df_test.drop(columns=["TimeDay"]).shape[1], hidden_dim)
model_power = ODERNN_Power(df_test.drop(columns=["Power"]).shape[1], hidden_dim)
model_poi = ODERNN_POI(df_test.drop(columns=["POI"]).shape[1], hidden_dim, num_poi_classes)
model_drivers = ODERNN_Drivers(len(driver_labels), hidden_dim, num_driver_features)

# Load trained weights
model_time.load_state_dict(torch.load("models/ODERNN_Time.pth"))
model_power.load_state_dict(torch.load("models/ODERNN_Power.pth"))
model_poi.load_state_dict(torch.load("models/ODERNN_POI.pth"))
model_drivers.load_state_dict(torch.load("models/ODERNN_Drivers.pth"))

# Set models to evaluation mode
model_time.eval()
model_power.eval()
model_poi.eval()
model_drivers.eval()

# Storage for evaluation results
all_rmse_time, all_accuracy_power, all_auc_power, all_accuracy_poi, all_f1_drivers = [], [], [], [], []

# Iterate over each day
for day, df_day in df_test.groupby("DayNum"):
    X_test_time = torch.tensor(df_day.drop(columns=["TimeDay"]).values, dtype=torch.float32)
    t_test_time = torch.tensor(df_day["TimeDay"].values, dtype=torch.float32)

    with torch.no_grad():
        y_test_pred_time = model_time(X_test_time.unsqueeze(1), t_test_time).squeeze(1)

    # True values
    target_time_test = torch.tensor(df_day["TimeDay"].values, dtype=torch.float32)

    # Compute RMSE for Time Prediction
    rmse_time = np.sqrt(mean_squared_error(target_time_test[:-1], y_test_pred_time.numpy()))
    all_rmse_time.append(rmse_time)

    # Evaluate other models similarly...
    X_test_power = torch.tensor(df_day.drop(columns=["Power"]).values, dtype=torch.float32)
    X_test_poi = torch.tensor(df_day.drop(columns=["POI"]).values, dtype=torch.float32)
    X_test_drivers = torch.tensor(df_day[driver_labels].values, dtype=torch.float32)

    with torch.no_grad():
        y_test_pred_power = model_power(X_test_power.unsqueeze(1), t_test_time).squeeze(1)
        y_test_pred_poi = model_poi(X_test_poi.unsqueeze(1), t_test_time).squeeze(1)
        y_test_pred_drivers = model_drivers(X_test_drivers.unsqueeze(1), t_test_time).squeeze(1)

    # Convert predictions
    y_test_pred_power_binary = (y_test_pred_power > 0.5).float()
    y_test_pred_poi_labels = torch.argmax(y_test_pred_poi, dim=1)
    y_test_pred_drivers_binary = (torch.sigmoid(y_test_pred_drivers) > 0.5).float()

    # True values
    target_power_test = torch.tensor(df_day["Power"].values, dtype=torch.float32)
    target_poi_test = torch.tensor(df_day["POI"].values, dtype=torch.long)
    target_drivers_test = torch.tensor(df_day[driver_labels].values, dtype=torch.float32)

    # Compute other metrics
    accuracy_power = accuracy_score(target_power_test[:-1], y_test_pred_power_binary.numpy())
    auc_power = roc_auc_score(target_power_test[:-1], y_test_pred_power.numpy())
    accuracy_poi = accuracy_score(target_poi_test[:-1], y_test_pred_poi_labels.numpy())
    f1_drivers = f1_score(target_drivers_test[:-1].numpy(), y_test_pred_drivers_binary.numpy(), average='samples')

    # Store results
    all_accuracy_power.append(accuracy_power)
    all_auc_power.append(auc_power)
    all_accuracy_poi.append(accuracy_poi)
    all_f1_drivers.append(f1_drivers)

# **Print Final Evaluation Metrics**
print(f"Test RMSE for Time Prediction: {np.mean(all_rmse_time):.4f}")
print(f"Test Accuracy for Power Prediction: {np.mean(all_accuracy_power):.4f}")
print(f"Test AUC for Power Prediction: {np.mean(all_auc_power):.4f}")
print(f"Test Accuracy for POI Prediction: {np.mean(all_accuracy_poi):.4f}")
print(f"Test F1-Score for Drivers Prediction: {np.mean(all_f1_drivers):.4f}")
