import pandas as pd
import torch
import torch.nn as nn
import torchdiffeq
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt


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
        
        # Iterate over the time steps
        for i in range(len(t) - 1):
            # Solve the ODE step
            h = self.ode_solver(self.ode_func, h, torch.tensor([t[i], t[i+1]], device=x.device))[1]
            
            # Pass the hidden state through the RNN
            h = self.rnn(x[i], h)
            
            # Apply fully connected layer if exists
            if self.fc:
                output = self.fc(h)
            else:
                output = h
            
            # Check if the output is valid (non-empty tensor)
            if output is not None and output.nelement() > 0:
                outputs.append(output)
            else:
                print(f"Warning: Empty or invalid output at time step {i}. Output shape: {output.shape if output is not None else 'None'}")
        
        # Ensure outputs list is not empty before stacking
        if len(outputs) == 0:
            print("Error: No valid outputs were generated.")
            return None  # Or handle the error accordingly
        
        # Check for None elements in the outputs list
        if any(output is None for output in outputs):
            print("Error: Found None in the outputs list.")
            return None  # Or handle the error accordingly
        
        # Apply sigmoid and return stacked outputs
        return torch.stack([torch.sigmoid(output) for output in outputs])



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



# Data Augmentation Functions
def add_noise(data, noise_level=0.01):
    noise = noise_level * torch.randn_like(data)
    return data + noise

def time_warp(data, alpha=0.1):
    stretched_time = np.arange(data.shape[0]) * (1 + alpha * np.random.randn())
    stretched_time = np.clip(stretched_time, 0, data.shape[0] - 1)
    return torch.tensor(np.interp(stretched_time, np.arange(data.shape[0]), data.numpy()), dtype=torch.float32)

def time_shift(data, shift_range=10):
    shift = np.random.randint(-shift_range, shift_range)
    return torch.roll(data, shifts=shift, dims=0)

def scale_magnitude(data, scale_range=(0.8, 1.2)):
    scale = torch.tensor(np.random.uniform(scale_range[0], scale_range[1]), dtype=torch.float32)
    return data * scale

def augment_data(X):
    augmentations = [add_noise, time_shift, scale_magnitude]
    # aug_func = random.choice(augmentations)
    aug_func = add_noise
    return aug_func(X)

# Data Preprocessing with Augmentation
def preprocess_data(df, driver_labels, augment=True):
    df["TimeDay"] = df["TimeDay"] / (24 * 60)  # Normalize within a day
    df["TimeDay"] += df.groupby("DayNum").cumcount() * 1e-4  # Small offset within each day to break ties
    df = df.sort_values(by=["DayNum", "TimeDay"]).reset_index(drop=True)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Extracting the target and features
    X = torch.tensor(df.drop(columns=["TimeDay", "Power", "POI"]).values, dtype=torch.float32)
    t = torch.tensor(df["TimeDay"].values, dtype=torch.float32)
    
    t_original = torch.tensor(df["TimeDay"].values, dtype=torch.float32)  # Original time labels

    if augment:
        X_aug = augment_data(X)  # Apply augmentation to X
        t_aug = augment_data(t_original.view(-1, 1)).squeeze()
        X = torch.cat([X, X_aug], dim=0)  # Double the data
        t = torch.cat([t_original, t_aug], dim=0)  # Ensure time labels are doubled
    else:
        X = X  # No augmentation for X
        t = t_original

    # print("X shape:", X.shape)
    # print("t shape:", t.shape)

    return X, t  # Only return X and t (t serves as both input and target)





# Load Data
data_path = "Datasets/DL_HMI_01_with_POI.csv"
df = pd.read_csv(data_path)
print("Original Data Shape:", df.shape)

weeks = df["WeekNum"].unique()
num_weeks = len(weeks)
print("Weeks:", num_weeks)  # Last row

# Extract driver labels dynamically
driver_labels = df.columns[df.columns.get_loc("Temperature") + 1 : df.columns.get_loc("POI")].tolist()

# Split weeks into train, validation, and test weeks
train_weeks = weeks[:int(0.8 * num_weeks)]
val_weeks = weeks[int(0.8 * num_weeks): int(0.9 * num_weeks)]
test_weeks = weeks[int(0.9 * num_weeks):]

# Split data by weeks
train_df = df[df["WeekNum"].isin(train_weeks)]
val_df = df[df["WeekNum"].isin(val_weeks)]
test_df = df[df["WeekNum"].isin(test_weeks)]

# Display shapes of the split data
print("Train Data Shape:", train_df.shape)
print("Validation Data Shape:", val_df.shape)
print("Test Data Shape:", test_df.shape)


# Preprocess training, validation, and testing data (augment training data only)
X_train, t_train = preprocess_data(train_df, driver_labels, augment=True)
X_val, t_val = preprocess_data(val_df, driver_labels, augment=False)  # No augmentation for validation
X_test, t_test = preprocess_data(test_df, driver_labels, augment=False)

# Example of augmenting the training data by duplicating it
augmented_df = pd.concat([train_df, train_df])  # Duplicate the training data
augmented_df = augmented_df.reset_index(drop=True)  # Reset the index after concatenation


print("Preprocessed X_train Shape:", X_train.shape)
print("Preprocessed t_train Shape:", t_train.shape)
print("Preprocessed X_val Shape:", X_val.shape)
print("Preprocessed t_val Shape:", t_val.shape)

# # Train-Test Split
# df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)
# print("Train Data Shape:", df_train.shape)
# print("Test Data Shape:", df_test.shape)

# # Preprocess training and testing data (augment training data only)
# X_train, t_train = preprocess_data(df_train, driver_labels, augment=True)
# X_test, t_test = preprocess_data(df_test, driver_labels, augment=False)

# augmented_df = pd.concat([df_train, df_train])  # Duplicate the training data
# augmented_df = augmented_df.reset_index(drop=True)  # Reset the index after concatenation

# print("Preprocessed X_train Shape:", X_train.shape)
# print("Preprocessed t_train Shape:", t_train.shape)

# Set target_time to be the augmented t_train
# target_time = t_train


# print("X_train shape:", X_train.shape)
# print("t_train shape:", t_train.shape)
# print("target_time shape:", target_time.shape)



# Model Initialization
model_time = ODERNN_Time(X_train.shape[1], 16)
model_power = ODERNN_Power(X_train.shape[1], 16)
model_poi = ODERNN_POI(X_train.shape[1], 16, df["POI"].nunique())
model_drivers = ODERNN_Drivers(X_train.shape[1], 16, len(driver_labels))

# Optimizer Setup
optimizers = {
    'time': torch.optim.AdamW(model_time.parameters(), lr=0.005, weight_decay=1e-5),
    'power': torch.optim.Adam(model_power.parameters(), lr=0.01),
    'poi': torch.optim.Adam(model_poi.parameters(), lr=0.01),
    'drivers': torch.optim.Adam(model_drivers.parameters(), lr=0.01)
}

# Store loss history
train_loss_history = []
val_loss_history = []


# Training Loop
epochs = 1000
train_loss_history = []
val_loss_history = []

try:
    for epoch in range(epochs):
        # Reset gradients
        for optimizer in optimizers.values():
            optimizer.zero_grad()

        # Train step
        y_pred_time = model_time(X_train.unsqueeze(1), t_train).squeeze(1)
        y_pred_time = y_pred_time.mean(dim=-1)  # Take the mean across the hidden dim

        target_time = torch.tensor(augmented_df["TimeDay"].values, dtype=torch.float32)
        loss_t = F.mse_loss(y_pred_time, target_time[:-1])  # Time prediction loss

        # Backpropagation
        loss_t.backward()

        # Gradient Clipping: clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model_time.parameters(), max_norm=1.0)

        for optimizer in optimizers.values():
            optimizer.step()

        # Store training loss
        train_loss_history.append(loss_t.item())

        # Validation step
        with torch.no_grad():
            model_time.eval()  # Switch to evaluation mode
            y_pred_time_val = model_time(X_val.unsqueeze(1), t_val).squeeze(1)
            y_pred_time_val = y_pred_time_val.mean(dim=-1)  # Take the mean across the hidden dim

            # Ensure the prediction and target sizes match
            min_len = min(len(y_pred_time_val), len(t_val))  # Take the minimum length of both
            y_pred_time_val = y_pred_time_val[:min_len]
            t_val = t_val[:min_len]

            # Compute validation loss
            loss_t_val = F.mse_loss(y_pred_time_val, t_val)
            val_loss_history.append(loss_t_val.item())

        # Print the loss at every 50th epoch
        if epoch % 50 == 0:
            print(f"Epoch [{epoch}/{epochs}], Train Loss: {loss_t.item():.4f}, Val Loss: {loss_t_val.item():.4f}")

except Exception as e:
    print("Error detected:", e)

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label="Train Loss", marker="o")
    plt.plot(val_loss_history, label="Validation Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss Progression")
    plt.show()

    raise  # Re-raise the exception so you still see the traceback

# Plot Training and Validation Loss
plt.plot(train_loss_history, label="Training Loss")
plt.plot(val_loss_history, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve Over Training and Validation")
plt.legend()
plt.show()



# # Plot Loss Curve
# plt.plot(loss_history, label="Training Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Loss Curve Over Training")
# plt.legend()
# plt.show()

# Save the models
torch.save(model_time.state_dict(), "models/ODERNN_Time.pth")
# torch.save(model_power.state_dict(), "models/ODERNN_Power.pth")
torch.save(model_poi.state_dict(), "models/ODERNN_POI.pth")
# torch.save(model_drivers.state_dict(), "models/ODERNN_Drivers.pth")

print("Models saved successfully!")