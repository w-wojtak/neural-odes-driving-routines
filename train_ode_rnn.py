import pandas as pd
import torch
import torch.nn as nn
import torchdiffeq
from sklearn.preprocessing import MinMaxScaler
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

# Load and preprocess dataset
data_path = "Datasets/DL_HMI_01_with_POI.csv"
df = pd.read_csv(data_path)

# print("First 50 values of TimeDay in df:")
# print(df["TimeDay"].values[:50])

# Normalize continuous features
# df["TimeDay"] = MinMaxScaler().fit_transform(df[["TimeDay"]])

# df = df.sort_values(by="TimeDay").drop_duplicates(subset=["TimeDay"], keep="last")
# print(df["TimeDay"])
num_poi_classes = df["POI"].nunique()

# Extract driver labels dynamically (assuming they are between Temperature and POI)
driver_labels = df.columns[df.columns.get_loc("Temperature") + 1 : df.columns.get_loc("POI")].tolist()
num_driver_features = len(driver_labels)

# Train-test split
df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)
df_train["TimeDay"] = df_train["TimeDay"] / (24 * 60)  # Normalize within a day
# Add a small offset within each day to break ties (ensures strictly increasing order)
df_train["TimeDay"] += df_train.groupby("DayNum").cumcount() * 1e-4  

# Sort to ensure increasing order
df_train = df_train.sort_values(by=["DayNum", "TimeDay"]).reset_index(drop=True)


print("First few TimeDay values after processing:")
print(df_train["TimeDay"].head(20).values)

# Check if all TimeDay values are strictly increasing
increasing_check = (df_train["TimeDay"].diff() > 0).all()
print(f"✅ Is TimeDay strictly increasing? {increasing_check}")

# Find duplicate timestamps
duplicates = df_train[df_train["TimeDay"].diff() == 0]["TimeDay"]
print(f"⚠️ Duplicate timestamps (should be empty): {duplicates}")




# print("First 50 values of TimeDay in df_train:")
# pd.set_option("display.max_rows", None)  # Show all rows
# pd.set_option("display.max_colwidth", None)  # Show full content in each row
# print(df["TimeDay"])


df_train = df_train.apply(pd.to_numeric, errors='coerce').fillna(0)
df_test = df_test.apply(pd.to_numeric, errors='coerce').fillna(0)

# Feature selection
X_train_time = torch.tensor(df_train.drop(columns=["TimeDay"]).values, dtype=torch.float32)
t_train_time = torch.tensor(df_train["TimeDay"].values, dtype=torch.float32)

X_train_power = torch.tensor(df_train.drop(columns=["Power"]).values, dtype=torch.float32)
t_train_power = torch.tensor(df_train["TimeDay"].values, dtype=torch.float32)

X_train_poi = torch.tensor(df_train.drop(columns=["POI"]).values, dtype=torch.float32)
t_train_poi = torch.tensor(df_train["TimeDay"].values, dtype=torch.float32)

X_train_drivers = torch.tensor(df_train[driver_labels].values, dtype=torch.float32)
t_train_drivers = torch.tensor(df_train["TimeDay"].values, dtype=torch.float32)

# Test set
X_test_time = torch.tensor(df_test.drop(columns=["TimeDay"]).values, dtype=torch.float32)
t_test_time = torch.tensor(df_test["TimeDay"].values, dtype=torch.float32)

X_test_power = torch.tensor(df_test.drop(columns=["Power"]).values, dtype=torch.float32)
t_test_power = torch.tensor(df_test["TimeDay"].values, dtype=torch.float32)

X_test_poi = torch.tensor(df_test.drop(columns=["POI"]).values, dtype=torch.float32)
t_test_poi = torch.tensor(df_test["TimeDay"].values, dtype=torch.float32)

X_test_drivers = torch.tensor(df_test[driver_labels].values, dtype=torch.float32)
t_test_drivers = torch.tensor(df_test["TimeDay"].values, dtype=torch.float32)

# Initialize models
model_time = ODERNN_Time(X_train_time.shape[1], 16)
model_power = ODERNN_Power(X_train_power.shape[1], 16)
model_poi = ODERNN_POI(X_train_poi.shape[1], 16, num_poi_classes)
model_drivers = ODERNN_Drivers(X_train_drivers.shape[1], 16, num_driver_features)

# Optimizers
optimizer_time = torch.optim.Adam(model_time.parameters(), lr=0.01)
optimizer_power = torch.optim.Adam(model_power.parameters(), lr=0.01)
optimizer_poi = torch.optim.Adam(model_poi.parameters(), lr=0.01)
optimizer_drivers = torch.optim.Adam(model_drivers.parameters(), lr=0.01)

# Training loop
epochs = 500
for epoch in range(epochs):
    optimizer_time.zero_grad()
    optimizer_power.zero_grad()
    optimizer_poi.zero_grad()
    optimizer_drivers.zero_grad()

    y_pred_time = model_time(X_train_time.unsqueeze(1), t_train_time).squeeze(1)
    y_pred_power = model_power(X_train_power.unsqueeze(1), t_train_power).squeeze(1)
    y_pred_poi = model_poi(X_train_poi.unsqueeze(1), t_train_poi).squeeze(1)
    y_pred_drivers = model_drivers(X_train_drivers.unsqueeze(1), t_train_drivers).squeeze(1)

    # Targets
    target_time = torch.tensor(df_train["TimeDay"].values, dtype=torch.float32)
    target_power = torch.tensor(df_train["Power"].values, dtype=torch.float32)
    target_poi = torch.tensor(df_train["POI"].values, dtype=torch.long)
    target_drivers = torch.tensor(df_train[driver_labels].values, dtype=torch.float32)

    # Compute losses
    loss_t = F.mse_loss(y_pred_time.squeeze(-1), target_time[:-1])  
    loss_p = F.binary_cross_entropy(y_pred_power.squeeze(-1), target_power[:-1])  
    loss_poi = F.cross_entropy(y_pred_poi, target_poi[:-1])  
    loss_d = F.binary_cross_entropy_with_logits(y_pred_drivers, target_drivers[:-1])  

    loss_t.backward()
    loss_p.backward()
    loss_poi.backward()
    loss_d.backward()

    optimizer_time.step()
    optimizer_power.step()
    optimizer_poi.step()
    optimizer_drivers.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss Time: {loss_t.item():.4f}, Loss Power: {loss_p.item():.4f}, Loss POI: {loss_poi.item():.4f}, Loss Drivers: {loss_d.item():.4f}")



# Save the models
torch.save(model_time.state_dict(), "models/ODERNN_Time.pth")
torch.save(model_power.state_dict(), "models/ODERNN_Power.pth")
torch.save(model_poi.state_dict(), "models/ODERNN_POI.pth")
torch.save(model_drivers.state_dict(), "models/ODERNN_Drivers.pth")

print("Models saved successfully!")
