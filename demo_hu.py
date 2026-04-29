import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
from sympy import sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

data = sio.loadmat('Xtrain.mat')
print(data['Xtrain'].shape)

series = np.array(data['Xtrain']).squeeze().astype(np.float32)
print(series.shape)
print(series[:10])

scaler = StandardScaler()

series_scaled = scaler.fit_transform(series.reshape(-1, 1)).squeeze()

print("Scaled shape:", series_scaled.shape)

class TimeSeriesDataset(Dataset):
    def __init__(self, series, window_size):
        self.series = series
        self.window_size = window_size

    def __len__(self):
        return len(self.series) - self.window_size

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.window_size]
        y = self.series[idx + self.window_size]

        # GRU input shape: [seq_len, input_size]
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y

