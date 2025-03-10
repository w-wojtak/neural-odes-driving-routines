import torch
import pandas as pd
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, f1_score
from models import ODERNN_Time, ODERNN_Power, ODERNN_POI, ODERNN_Drivers  # Import model classes

# Load the dataset
df_test = pd.read_csv("Datasets/DL_HMI_01_with_POI.csv")

# Preprocess test set (same way as training)
df_test = df_test.sort_values(by="TimeDay").drop_duplicates(subset=["TimeDay"], keep="last")
df_test = df_test.apply(pd.to_numeric, errors='coerce').fillna(0)

num_poi_classes = df_test["POI"].nunique()
driver_labels = [f"Driver_{name}" for name in ["Homer", "Bart", "Lisa", "Marge"]]  # Ensure correct driver names
num_driver_features = len(driver_labels)

# Ensure multi-hot encoded drivers exist in test set
if "Drivers_MultiHot" in df_test.columns:
    drivers_encoded_test = np.vstack(df_test["Drivers_MultiHot"].values)
    df_test_drivers = pd.DataFrame(drivers_encoded_test, columns=driver_labels)
    df_test = pd.concat([df_test, df_test_drivers], axis=1)
    df_test.drop(columns=["Drivers_MultiHot"], inplace=True)  # Drop temp column

# Prepare test tensors
X_test_time = torch.tensor(df_test.drop(columns=["TimeDay"]).values, dtype=torch.float32)
t_test_time = torch.tensor(df_test["TimeDay"].values, dtype=torch.float32)

X_test_power = torch.tensor(df_test.drop(columns=["Power"]).values, dtype=torch.float32)
t_test_power = torch.tensor(df_test["TimeDay"].values, dtype=torch.float32)

X_test_poi = torch.tensor(df_test.drop(columns=["POI"]).values, dtype=torch.float32)
t_test_poi = torch.tensor(df_test["TimeDay"].values, dtype=torch.float32)

# Drop only existing driver label columns
X_test_drivers = torch.tensor(
    df_test.drop(columns=[col for col in driver_labels if col in df_test.columns]).values, dtype=torch.float32
)
t_test_drivers = torch.tensor(df_test["TimeDay"].values, dtype=torch.float32)

print(f"Evaluation input size for Time model: {X_test_time.shape[1]}")


# Initialize models
model_time = ODERNN_Time(X_test_time.shape[1], 16)
model_power = ODERNN_Power(X_test_power.shape[1], 16)
model_poi = ODERNN_POI(X_test_poi.shape[1], 16, num_poi_classes)
model_drivers = ODERNN_Drivers(X_test_drivers.shape[1], 16, num_driver_features)

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
rmse_time = mean_squared_error(target_time_test[:-1], y_test_pred_time.numpy(), squared=False)
accuracy_power = accuracy_score(target_power_test[:-1], y_test_pred_power_binary.numpy())
auc_power = roc_auc_score(target_power_test[:-1], y_test_pred_power.numpy())
accuracy_poi = accuracy_score(target_poi_test[:-1], y_test_pred_poi_labels.numpy())
f1_drivers = f1_score(target_drivers_test[:-1].numpy(), y_test_pred_drivers_binary.numpy(), average='samples')

# Print Metrics
print(f"Test RMSE for Time Prediction: {rmse_time:.4f}")
print(f"Test Accuracy for Power Prediction: {accuracy_power:.4f}")
print(f"Test AUC for Power Prediction: {auc_power:.4f}")
print(f"Test Accuracy for POI Prediction: {accuracy_poi:.4f}")
print(f"Test F1-Score for Drivers Prediction: {f1_drivers:.4f}")
