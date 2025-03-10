import pandas as pd
import torch
import torch.nn as nn
import torchdiffeq
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np

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