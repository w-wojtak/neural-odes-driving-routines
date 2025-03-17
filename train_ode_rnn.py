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
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)  # Use GRU instead of GRUCell
        self.ode_solver = torchdiffeq.odeint
        self.fc_time = nn.Linear(hidden_dim, 1)

    def forward(self, x, t):
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        h_trajectory = self.ode_solver(self.ode_func, h, torch.tensor(t, device=x.device), method='euler')

        # Reshape hidden state
        h0 = h_trajectory[:-1].permute(1, 0, 2).contiguous()  # (batch, time, hidden_dim) -> (num_layers, batch, hidden_dim)
        h0 = h0[0].unsqueeze(0)  # Reshape for single-layer GRU

        # Ensure the batch size matches the expected input shape
        if h0.size(1) != x.size(0):
            raise ValueError(f"Expected batch size {x.size(0)}, but got {h0.size(1)} in the hidden state.")

        h_next, _ = self.rnn(x, h0)
        outputs = self.fc_time(h_next)

        return outputs



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



# Load dataset
data_path = "Datasets/DL_HMI_01_with_POI.csv"
df = pd.read_csv(data_path)

num_poi_classes = df["POI"].nunique()
driver_labels = df.columns[df.columns.get_loc("Temperature") + 1 : df.columns.get_loc("POI")].tolist()
num_driver_features = len(driver_labels)

# # Train-test split
# df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)

# Get unique days in sorted order
unique_days = df["DayNum"].unique()
num_train_days = int(len(unique_days) * 0.8)  # 80% for training

# Select days for training and testing
train_days = unique_days[:num_train_days]
test_days = unique_days[num_train_days:]

# Filter dataset based on the selected days
df_train = df[df["DayNum"].isin(train_days)].copy()
df_test = df[df["DayNum"].isin(test_days)].copy()

print(f"Training days: {train_days[:5]} ... {train_days[-5:]}")
print(f"Testing days: {test_days[:5]} ... {test_days[-5:]}")
print(f"Train size: {len(df_train)}, Test size: {len(df_test)}")


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
model_poi = ODERNN_POI(df_train.drop(columns=["POI"]).shape[1], hidden_dim, num_poi_classes)

# Optimizers
optimizer_time = torch.optim.Adam(model_time.parameters(), lr=0.005)
optimizer_poi = torch.optim.Adam(model_poi.parameters(), lr=0.01)

# Training loop over days
# Training loop over days, ensuring batches contain only one day's sequence
epochs = 500
batch_size = 1  # Ensuring each batch contains only one day

for epoch in range(epochs):
    total_loss_t = 0
    
    # Shuffle days before each epoch (optional, but could help generalization)
    train_days = df_train["DayNum"].unique()
    np.random.shuffle(train_days)

    for i in range(0, len(train_days), batch_size):
        batch_days = train_days[i : i + batch_size]
        df_batch = df_train[df_train["DayNum"].isin(batch_days)]

        # Prepare inputs
        X_train_time = torch.tensor(df_batch.drop(columns=["TimeDay"]).values, dtype=torch.float32)
        t_train_time = torch.tensor(df_batch["TimeDay"].values, dtype=torch.float32)

        optimizer_time = torch.optim.Adam(model_time.parameters(), lr=0.005, weight_decay=1e-5)

        # Forward pass
        y_pred_time = model_time(X_train_time.unsqueeze(1), t_train_time).squeeze(1)

        # Targets
        target_time = torch.tensor(df_batch["TimeDay"].values, dtype=torch.float32)

        # Compute loss
        loss_t = F.mse_loss(y_pred_time.squeeze(-1), target_time[:-1])  
        loss_t.backward()
        optimizer_time.step()

        total_loss_t += loss_t.item()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Avg Loss Time: {total_loss_t/len(train_days):.4f}")



# Save the models
torch.save(model_time.state_dict(), "models/ODERNN_Time.pth")
torch.save(model_poi.state_dict(), "models/ODERNN_POI.pth")

print("Models saved successfully!")
