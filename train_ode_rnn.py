import pandas as pd
import torch
import torch.nn as nn
import torchdiffeq
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np

# Define ODE and ODERNN models
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

class ODERNN_Time(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ODERNN_Time, self).__init__()
        self.hidden_dim = hidden_dim
        self.ode_func = ODEFunc(hidden_dim)
        self.rnn = nn.GRUCell(input_dim, hidden_dim)
        self.ode_solver = torchdiffeq.odeint
        self.fc_time = nn.Linear(hidden_dim, 1)

    def forward(self, x, t):
        h = torch.zeros(x.size(1), self.hidden_dim)
        outputs = []
        for i in range(len(t) - 1):
            h = self.ode_solver(self.ode_func, h, torch.tensor([t[i], t[i+1]]))[1]
            h = self.rnn(x[i], h)
            outputs.append(self.fc_time(h))
        return torch.stack(outputs)

class ODERNN_POI(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_poi_classes):
        super(ODERNN_POI, self).__init__()
        self.hidden_dim = hidden_dim
        self.ode_func = ODEFunc(hidden_dim)
        self.rnn = nn.GRUCell(input_dim, hidden_dim)
        self.ode_solver = torchdiffeq.odeint
        self.fc_poi = nn.Linear(hidden_dim, num_poi_classes)

    def forward(self, x, t):
        h = torch.zeros(x.size(1), self.hidden_dim)
        outputs = []
        for i in range(len(t) - 1):
            h = self.ode_solver(self.ode_func, h, torch.tensor([t[i], t[i+1]]))[1]
            h = self.rnn(x[i], h)
            outputs.append(self.fc_poi(h))
        return torch.stack(outputs)

class ODERNN_Power(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ODERNN_Power, self).__init__()
        self.hidden_dim = hidden_dim
        self.ode_func = ODEFunc(hidden_dim)
        self.rnn = nn.GRUCell(input_dim, hidden_dim)
        self.ode_solver = torchdiffeq.odeint
        self.fc_power = nn.Linear(hidden_dim, 1)

    def forward(self, x, t):
        h = torch.zeros(x.size(1), self.hidden_dim)
        outputs = []
        for i in range(len(t) - 1):
            h = self.ode_solver(self.ode_func, h, torch.tensor([t[i], t[i+1]]))[1]
            h = self.rnn(x[i], h)
            outputs.append(torch.sigmoid(self.fc_power(h)))
        return torch.stack(outputs)

class ODERNN_Drivers(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_driver_features):
        super(ODERNN_Drivers, self).__init__()
        self.hidden_dim = hidden_dim
        self.ode_func = ODEFunc(hidden_dim)
        self.rnn = nn.GRUCell(input_dim, hidden_dim)
        self.ode_solver = torchdiffeq.odeint
        self.fc = nn.Linear(hidden_dim, num_driver_features)

    def forward(self, x, t):
        h = torch.zeros(x.size(1), self.hidden_dim)
        outputs = []
        for i in range(len(t) - 1):
            h = self.ode_solver(self.ode_func, h, torch.tensor([t[i], t[i+1]]))[1]
            h = self.rnn(x[i], h)
            outputs.append(self.fc(h))
        return torch.stack(outputs)

# Load dataset
data_path = "Datasets/DL_HMI_01_with_POI.csv"
df = pd.read_csv(data_path)

num_poi_classes = df["POI"].nunique()
driver_labels = df.columns[df.columns.get_loc("Temperature") + 1 : df.columns.get_loc("POI")].tolist()
num_driver_features = len(driver_labels)

# Train-test split
df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)

# Normalize within a day
df_train["TimeDay"] = df_train["TimeDay"] / (24 * 60)  
df_train["TimeDay"] += df_train.groupby("DayNum").cumcount() * 1e-4  

# Sort within each day
df_train = df_train.sort_values(by=["DayNum", "TimeDay"]).reset_index(drop=True)

df_train = df_train.apply(pd.to_numeric, errors='coerce').fillna(0)
df_test = df_test.apply(pd.to_numeric, errors='coerce').fillna(0)

# Initialize models
hidden_dim = 16
model_time = ODERNN_Time(df_train.drop(columns=["TimeDay"]).shape[1], hidden_dim)
model_power = ODERNN_Power(df_train.drop(columns=["Power"]).shape[1], hidden_dim)
model_poi = ODERNN_POI(df_train.drop(columns=["POI"]).shape[1], hidden_dim, num_poi_classes)
model_drivers = ODERNN_Drivers(len(driver_labels), hidden_dim, num_driver_features)

# Optimizers
optimizer_time = torch.optim.Adam(model_time.parameters(), lr=0.005)
optimizer_power = torch.optim.Adam(model_power.parameters(), lr=0.01)
optimizer_poi = torch.optim.Adam(model_poi.parameters(), lr=0.01)
optimizer_drivers = torch.optim.Adam(model_drivers.parameters(), lr=0.01)

# Training loop over days
epochs = 500
for epoch in range(epochs):
    optimizer_time.zero_grad()
    all_loss_t = 0

    for day, df_day in df_train.groupby("DayNum"):
        X_train_time = torch.tensor(df_day.drop(columns=["TimeDay"]).values, dtype=torch.float32)
        t_train_time = torch.tensor(df_day["TimeDay"].values, dtype=torch.float32)

        y_pred_time = model_time(X_train_time.unsqueeze(1), t_train_time).squeeze(1)
        target_time = torch.tensor(df_day["TimeDay"].values, dtype=torch.float32)

        loss_t = F.mse_loss(y_pred_time.squeeze(-1), target_time[:-1])  
        all_loss_t += loss_t

    all_loss_t.backward()  # Accumulate gradients across days
    optimizer_time.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Avg Loss Time: {all_loss_t.item() / len(df_train['DayNum'].unique()):.4f}")


# Save the models
torch.save(model_time.state_dict(), "models/ODERNN_Time.pth")
torch.save(model_power.state_dict(), "models/ODERNN_Power.pth")
torch.save(model_poi.state_dict(), "models/ODERNN_POI.pth")
torch.save(model_drivers.state_dict(), "models/ODERNN_Drivers.pth")

print("Models saved successfully!")
