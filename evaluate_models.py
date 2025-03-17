import torch
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
# from train_ode_rnn import preprocess_data

import pandas as pd
import torch
import torch.nn as nn
import torchdiffeq
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np


def preprocess_data(df, driver_labels):
    df["TimeDay"] = df["TimeDay"] / (24 * 60)  # Normalize within a day
    df["TimeDay"] += df.groupby("DayNum").cumcount() * 1e-4  # Small offset within each day to break ties
    df = df.sort_values(by=["DayNum", "TimeDay"]).reset_index(drop=True)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Extracting the target and features
    X = torch.tensor(df.drop(columns=["TimeDay", "Power", "POI"]).values, dtype=torch.float32)
    t = torch.tensor(df["TimeDay"].values, dtype=torch.float32)
    
    return X, t


# Base ODE-RNN Model Class
class BaseODERNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=None):
        super(BaseODERNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.ode_func = ODEFunc(hidden_dim)
        self.rnn = nn.GRUCell(input_dim, hidden_dim)
        self.ode_solver = torchdiffeq.odeint
        self.fc = nn.Linear(hidden_dim, num_classes) if num_classes else None

    def forward(self, x, t):
        h = torch.zeros(x.size(1), self.hidden_dim).to(x.device)  # Ensure device consistency
        outputs = []
        for i in range(len(t) - 1):
            h = self.ode_solver(self.ode_func, h, torch.tensor([t[i], t[i+1]], device=x.device))[1]
            h = self.rnn(x[i], h)
            if self.fc:
                outputs.append(self.fc(h))
            else:
                outputs.append(h)
        return torch.stack(outputs)

# Define ODE Function
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 50),
            nn.Tanh(),
            nn.Linear(50, hidden_dim)
        )

    def forward(self, t, h):
        return self.net(h)

# Model for Time Prediction
class ODERNN_Time(BaseODERNN):
    def __init__(self, input_dim, hidden_dim):
        super(ODERNN_Time, self).__init__(input_dim, hidden_dim)

    def forward(self, x, t):
        outputs = super().forward(x, t)
        return torch.stack([torch.sigmoid(output) for output in outputs])

# Model for POI Prediction
class ODERNN_POI(BaseODERNN):
    def __init__(self, input_dim, hidden_dim, num_poi_classes):
        super(ODERNN_POI, self).__init__(input_dim, hidden_dim, num_poi_classes)



# Load saved models
def load_model(model_class, input_dim, hidden_dim, num_classes=None, model_path=None):
    model = model_class(input_dim, hidden_dim, num_classes) if num_classes else model_class(input_dim, hidden_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Load test data
data_path = "Datasets/DL_HMI_01_with_POI.csv"
df = pd.read_csv(data_path)

# Extract driver labels dynamically
driver_labels = df.columns[df.columns.get_loc("Temperature") + 1 : df.columns.get_loc("POI")].tolist()

df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)
X_test, t_test = preprocess_data(df_test, driver_labels)

# Initialize models and load weights
model_time = load_model(ODERNN_Time, X_test.shape[1], 16, model_path="models/ODERNN_Time.pth")
# model_power = load_model(ODERNN_Power, X_test.shape[1], 16, model_path="models/ode_rnn_power.pth")
model_poi = load_model(ODERNN_POI, X_test.shape[1], 16, df["POI"].nunique(), model_path="models/ODERNN_POI.pth")
# model_drivers = load_model(ODERNN_Drivers, X_test.shape[1], 16, len(driver_labels), model_path="models/ode_rnn_drivers.pth")

# Predictions
y_pred_time = model_time(X_test.unsqueeze(1), t_test).squeeze(1).mean(dim=-1).detach().numpy()
# y_pred_power = model_power(X_test.unsqueeze(1), t_test).squeeze(1).mean(dim=-1).detach().numpy()
y_pred_poi = torch.argmax(model_poi(X_test.unsqueeze(1), t_test).squeeze(1), dim=1).detach().numpy()
# y_pred_drivers = model_drivers(X_test.unsqueeze(1), t_test).squeeze(1).detach().numpy()

# Targets
target_time = df_test["TimeDay"].values
target_poi = df_test["POI"].values
# target_power = df_test["Power"].values
# target_drivers = df_test[driver_labels].values

# Compute Evaluation Metrics
mse_time = mean_squared_error(target_time[:-1], y_pred_time)
# mse_power = mean_squared_error(target_power[:-1], y_pred_power)
accuracy_poi = accuracy_score(target_poi[:-1], y_pred_poi)
# mse_drivers = mean_squared_error(target_drivers[:-1], y_pred_drivers)

# Print Results
print(f"Time Prediction MSE: {mse_time:.4f}")
# print(f"Power Prediction MSE: {mse_power:.4f}")
print(f"POI Classification Accuracy: {accuracy_poi:.4f}")
# print(f"Driver Features MSE: {mse_drivers:.4f}")










# import pandas as pd
# import torch
# import torch.nn as nn
# import torchdiffeq
# import torch.nn.functional as F
# import numpy as np
# from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, f1_score
# from models import ODERNN_Time, ODERNN_Power, ODERNN_POI, ODERNN_Drivers  # Import model classes

# # Load dataset
# data_path = "Datasets/DL_HMI_01_with_POI.csv"
# df = pd.read_csv(data_path)

# # Extract driver labels dynamically
# driver_labels = df.columns[df.columns.get_loc("Temperature") + 1 : df.columns.get_loc("POI")].tolist()
# num_driver_features = len(driver_labels)
# num_poi_classes = df["POI"].nunique()

# # Normalize within a day
# df["TimeDay"] = df["TimeDay"] / (24 * 60)
# df["TimeDay"] += df.groupby("DayNum").cumcount() * 1e-4
# df = df.sort_values(by=["DayNum", "TimeDay"]).reset_index(drop=True)

# # Get unique days and split
# unique_days = df["DayNum"].unique()
# num_train_days = int(len(unique_days) * 0.7)  # 70% train, remaining for validation & test
# num_val_days = int(len(unique_days) * 0.15)   # 15% validation
# num_test_days = len(unique_days) - num_train_days - num_val_days  # 15% test

# test_days = unique_days[num_train_days + num_val_days:]
# df_test = df[df["DayNum"].isin(test_days)].copy()

# # Ensure all values are numeric
# df_test = df_test.apply(pd.to_numeric, errors='coerce').fillna(0)

# # Convert test set to tensor format
# X_test_time = torch.tensor(df_test.drop(columns=["TimeDay"]).values, dtype=torch.float32)
# t_test_time = torch.tensor(df_test["TimeDay"].values, dtype=torch.float32)

# X_test_power = torch.tensor(df_test.drop(columns=["Power"]).values, dtype=torch.float32)
# t_test_power = torch.tensor(df_test["TimeDay"].values, dtype=torch.float32)

# X_test_poi = torch.tensor(df_test.drop(columns=["POI"]).values, dtype=torch.float32)
# t_test_poi = torch.tensor(df_test["TimeDay"].values, dtype=torch.float32)

# X_test_drivers = torch.tensor(df_test[driver_labels].values, dtype=torch.float32)
# t_test_drivers = torch.tensor(df_test["TimeDay"].values, dtype=torch.float32)

# # Initialize models
# hidden_dim = 16
# model_time = ODERNN_Time(X_test_time.shape[1], hidden_dim)
# model_power = ODERNN_Power(X_test_power.shape[1], hidden_dim)
# model_poi = ODERNN_POI(X_test_poi.shape[1], hidden_dim, num_poi_classes)
# model_drivers = ODERNN_Drivers(len(driver_labels), hidden_dim, num_driver_features)

# # Load trained weights
# model_time.load_state_dict(torch.load("models/ODERNN_Time.pth"))
# model_power.load_state_dict(torch.load("models/ODERNN_Power.pth"))
# model_poi.load_state_dict(torch.load("models/ODERNN_POI.pth"))
# model_drivers.load_state_dict(torch.load("models/ODERNN_Drivers.pth"))

# # Set models to evaluation mode
# model_time.eval()
# model_power.eval()
# model_poi.eval()
# model_drivers.eval()

# # Generate Predictions
# with torch.no_grad():
#     y_test_pred_time = model_time(X_test_time.unsqueeze(1), t_test_time).squeeze(1)
#     y_test_pred_power = model_power(X_test_power.unsqueeze(1), t_test_power).squeeze(1)
#     y_test_pred_poi = model_poi(X_test_poi.unsqueeze(1), t_test_poi).squeeze(1)
#     y_test_pred_drivers = model_drivers(X_test_drivers.unsqueeze(1), t_test_drivers).squeeze(1)

# # Extract True Labels
# target_time_test = torch.tensor(df_test["TimeDay"].values, dtype=torch.float32)
# target_power_test = torch.tensor(df_test["Power"].values, dtype=torch.float32)
# target_poi_test = torch.tensor(df_test["POI"].values, dtype=torch.long)
# target_drivers_test = torch.tensor(df_test[driver_labels].values, dtype=torch.float32)

# # Convert predictions
# y_test_pred_power_binary = (y_test_pred_power > 0.5).float()
# y_test_pred_poi_labels = torch.argmax(y_test_pred_poi, dim=1)
# y_test_pred_drivers_binary = (torch.sigmoid(y_test_pred_drivers) > 0.5).float()

# # Compute Evaluation Metrics
# rmse_time = np.sqrt(mean_squared_error(target_time_test[:-1].numpy(), y_test_pred_time.numpy()))
# accuracy_power = accuracy_score(target_power_test[:-1].numpy(), y_test_pred_power_binary.numpy())
# auc_power = roc_auc_score(target_power_test[:-1].numpy(), y_test_pred_power.numpy())
# accuracy_poi = accuracy_score(target_poi_test[:-1].numpy(), y_test_pred_poi_labels.numpy())
# f1_drivers = f1_score(target_drivers_test[:-1].numpy(), y_test_pred_drivers_binary.numpy(), average='samples')

# # Print Metrics
# print(f"Test RMSE for Time Prediction: {rmse_time:.4f}")
# print(f"Test Accuracy for Power Prediction: {accuracy_power:.4f}")
# print(f"Test AUC for Power Prediction: {auc_power:.4f}")
# print(f"Test Accuracy for POI Prediction: {accuracy_poi:.4f}")
# print(f"Test F1-Score for Drivers Prediction: {f1_drivers:.4f}")
