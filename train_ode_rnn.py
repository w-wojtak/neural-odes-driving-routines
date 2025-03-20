import pandas as pd
import torch
import torch.nn as nn
import torchdiffeq
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np

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

# Model for Power Prediction
class ODERNN_Power(BaseODERNN):
    def __init__(self, input_dim, hidden_dim):
        super(ODERNN_Power, self).__init__(input_dim, hidden_dim)

    def forward(self, x, t):
        outputs = super().forward(x, t)
        return torch.sigmoid(torch.stack(outputs))

# Model for Driver Features
class ODERNN_Drivers(BaseODERNN):
    def __init__(self, input_dim, hidden_dim, num_driver_features):
        super(ODERNN_Drivers, self).__init__(input_dim, hidden_dim, num_driver_features)

# Data Preprocessing
def preprocess_data(df, driver_labels):
    df["TimeDay"] = df["TimeDay"] / (24 * 60)  # Normalize within a day
    df["TimeDay"] += df.groupby("DayNum").cumcount() * 1e-4  # Small offset within each day to break ties
    df = df.sort_values(by=["DayNum", "TimeDay"]).reset_index(drop=True)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Extracting the target and features
    X = torch.tensor(df.drop(columns=["TimeDay", "Power", "POI"]).values, dtype=torch.float32)
    t = torch.tensor(df["TimeDay"].values, dtype=torch.float32)
    
    return X, t

# Load Data
data_path = "Datasets/DL_HMI_01_with_POI.csv"
df = pd.read_csv(data_path)

# Dynamically extracting driver labels (assuming they are between Temperature and POI)
driver_labels = df.columns[df.columns.get_loc("Temperature") + 1 : df.columns.get_loc("POI")].tolist()

# Train-Test Split
df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)

# Preprocess training and testing data
X_train, t_train = preprocess_data(df_train, driver_labels)
X_test, t_test = preprocess_data(df_test, driver_labels)

# Model Initialization
model_time = ODERNN_Time(X_train.shape[1], 16)
model_power = ODERNN_Power(X_train.shape[1], 16)
model_poi = ODERNN_POI(X_train.shape[1], 16, df["POI"].nunique())
model_drivers = ODERNN_Drivers(X_train.shape[1], 16, len(driver_labels))

# Optimizer Setup (Using a dictionary for simplicity)
optimizers = {
    'time': torch.optim.Adam(model_time.parameters(), lr=0.0015),
    'power': torch.optim.Adam(model_power.parameters(), lr=0.01),
    'poi': torch.optim.Adam(model_poi.parameters(), lr=0.01),
    'drivers': torch.optim.Adam(model_drivers.parameters(), lr=0.01)
}

# Training Loop
epochs = 1000
for epoch in range(epochs):
    # Reset gradients
    for optimizer in optimizers.values():
        optimizer.zero_grad()

    # Predictions
    y_pred_time = model_time(X_train.unsqueeze(1), t_train).squeeze(1)
    y_pred_time = y_pred_time.mean(dim=-1)  # Take the mean across the 16 dimension
    y_pred_poi = model_poi(X_train.unsqueeze(1), t_train).squeeze(1)

    # Targets
    target_time = torch.tensor(df_train["TimeDay"].values, dtype=torch.float32)
    target_poi = torch.tensor(df_train["POI"].values, dtype=torch.long)

    # Losses
    loss_t = F.mse_loss(y_pred_time, target_time[:-1])   # Time prediction loss
    loss_poi = F.cross_entropy(y_pred_poi, target_poi[:-1])  # POI classification loss

    # Backpropagation
    total_loss = loss_t + loss_poi  # Total loss (you can add more losses as necessary)
    total_loss.backward()

    # Update weights
    for optimizer in optimizers.values():
        optimizer.step()

    if epoch % 50 == 0:  # Print loss every 50 epochs
        print(f"Epoch [{epoch}/{epochs}], Loss: {total_loss.item():.4f}")





# Save the models
torch.save(model_time.state_dict(), "models/ODERNN_Time.pth")
# torch.save(model_power.state_dict(), "models/ODERNN_Power.pth")
torch.save(model_poi.state_dict(), "models/ODERNN_POI.pth")
# torch.save(model_drivers.state_dict(), "models/ODERNN_Drivers.pth")

print("Models saved successfully!")
