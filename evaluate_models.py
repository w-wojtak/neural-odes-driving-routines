import pandas as pd
import torch
import torch.nn as nn
import torchdiffeq
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, f1_score
from models import ODERNN_Time, ODERNN_Power, ODERNN_POI, ODERNN_Drivers  # Import model classes

# Load dataset
data_path = "Datasets/DL_HMI_01_with_POI.csv"
df = pd.read_csv(data_path)

# Extract driver labels dynamically
driver_labels = df.columns[df.columns.get_loc("Temperature") + 1 : df.columns.get_loc("POI")].tolist()
num_driver_features = len(driver_labels)
num_poi_classes = df["POI"].nunique()

# Normalize within a day
df["TimeDay"] = df["TimeDay"] / (24 * 60)
df["TimeDay"] += df.groupby("DayNum").cumcount() * 1e-4
df = df.sort_values(by=["DayNum", "TimeDay"]).reset_index(drop=True)

# Get unique days and split
unique_days = df["DayNum"].unique()
num_train_days = int(len(unique_days) * 0.7)  # 70% train, remaining for validation & test
num_val_days = int(len(unique_days) * 0.15)   # 15% validation
num_test_days = len(unique_days) - num_train_days - num_val_days  # 15% test

test_days = unique_days[num_train_days + num_val_days:]
df_test = df[df["DayNum"].isin(test_days)].copy()

# Ensure all values are numeric
df_test = df_test.apply(pd.to_numeric, errors='coerce').fillna(0)

# Convert test set to tensor format
X_test_time = torch.tensor(df_test.drop(columns=["TimeDay"]).values, dtype=torch.float32)
t_test_time = torch.tensor(df_test["TimeDay"].values, dtype=torch.float32)

X_test_power = torch.tensor(df_test.drop(columns=["Power"]).values, dtype=torch.float32)
t_test_power = torch.tensor(df_test["TimeDay"].values, dtype=torch.float32)

X_test_poi = torch.tensor(df_test.drop(columns=["POI"]).values, dtype=torch.float32)
t_test_poi = torch.tensor(df_test["TimeDay"].values, dtype=torch.float32)

X_test_drivers = torch.tensor(df_test[driver_labels].values, dtype=torch.float32)
t_test_drivers = torch.tensor(df_test["TimeDay"].values, dtype=torch.float32)

# Initialize models
hidden_dim = 16
model_time = ODERNN_Time(X_test_time.shape[1], hidden_dim)
model_power = ODERNN_Power(X_test_power.shape[1], hidden_dim)
model_poi = ODERNN_POI(X_test_poi.shape[1], hidden_dim, num_poi_classes)
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

# Generate Predictions
with torch.no_grad():
    y_test_pred_time = model_time(X_test_time.unsqueeze(1), t_test_time).squeeze(1)
    y_test_pred_power = model_power(X_test_power.unsqueeze(1), t_test_power).squeeze(1)
    y_test_pred_poi = model_poi(X_test_poi.unsqueeze(1), t_test_poi).squeeze(1)
    y_test_pred_drivers = model_drivers(X_test_drivers.unsqueeze(1), t_test_drivers).squeeze(1)

# Extract True Labels
target_time_test = torch.tensor(df_test["TimeDay"].values, dtype=torch.float32)
target_power_test = torch.tensor(df_test["Power"].values, dtype=torch.float32)
target_poi_test = torch.tensor(df_test["POI"].values, dtype=torch.long)
target_drivers_test = torch.tensor(df_test[driver_labels].values, dtype=torch.float32)

# Convert predictions
y_test_pred_power_binary = (y_test_pred_power > 0.5).float()
y_test_pred_poi_labels = torch.argmax(y_test_pred_poi, dim=1)
y_test_pred_drivers_binary = (torch.sigmoid(y_test_pred_drivers) > 0.5).float()

# Compute Evaluation Metrics
rmse_time = np.sqrt(mean_squared_error(target_time_test[:-1].numpy(), y_test_pred_time.numpy()))
accuracy_power = accuracy_score(target_power_test[:-1].numpy(), y_test_pred_power_binary.numpy())
auc_power = roc_auc_score(target_power_test[:-1].numpy(), y_test_pred_power.numpy())
accuracy_poi = accuracy_score(target_poi_test[:-1].numpy(), y_test_pred_poi_labels.numpy())
f1_drivers = f1_score(target_drivers_test[:-1].numpy(), y_test_pred_drivers_binary.numpy(), average='samples')

# Print Metrics
print(f"Test RMSE for Time Prediction: {rmse_time:.4f}")
print(f"Test Accuracy for Power Prediction: {accuracy_power:.4f}")
print(f"Test AUC for Power Prediction: {auc_power:.4f}")
print(f"Test Accuracy for POI Prediction: {accuracy_poi:.4f}")
print(f"Test F1-Score for Drivers Prediction: {f1_drivers:.4f}")
