import scipy.io
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# --- 1. Hyperparameters & Configuration ---
FILE_PATH = "Xtrain.mat"
SEQ_LENGTH = 10       # Number of past time steps to look at to predict the next step
TRAIN_SPLIT = 1     # 80% for training, 20% for testing
BATCH_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.005
EPOCHS = 50

# --- 2. Data Loading and Preprocessing ---
def load_and_preprocess_data(filepath):
    # Load data
    data = scipy.io.loadmat(filepath)
    raw_data = data['Xtrain'].astype(np.float32) # Convert uint8 to float32
    
    # Scale data to [0, 1] (Crucial for Neural Networks)
    min_val = np.min(raw_data)
    max_val = np.max(raw_data)
    scaled_data = (raw_data - min_val) / (max_val - min_val)
    
    return scaled_data, min_val, max_val

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Load data
print("Loading data...")
data, min_val, max_val = load_and_preprocess_data(FILE_PATH)

# Create sequences
X, y = create_sequences(data, SEQ_LENGTH)

# --- 3. Train/Test Split (Chronological) ---
# IMPORTANT: Never shuffle before splitting in time series!
split_idx = int(len(X) * TRAIN_SPLIT)

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Total sequences: {len(X)}")
print(f"Training shapes: X={X_train.shape}, y={y_train.shape}")
print(f"Testing shapes: X={X_test.shape}, y={y_test.shape}")

# Convert to PyTorch tensors
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) 
# Note: It is okay to shuffle the batches *during* training, but not the train/test split.

# --- 4. LSTM Model Definition ---
class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1):
        super(TimeSeriesLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer expects input shape: (batch_size, seq_length, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer to map LSTM output to final prediction
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

model = TimeSeriesLSTM(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=1)
criterion = nn.MSELoss() # Mean Squared Error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. Training Loop ---
print("\nStarting Training...")
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss/len(train_loader):.6f}')

# --- 6. Recursive Prediction (200 steps) ---
print("\nStarting Recursive Prediction...")
model.eval()

N_PREDICT = 200

# Use the last SEQ_LENGTH points from training data as the starting window
# data is the full scaled dataset (shape: [N, 1])
last_window = data[-SEQ_LENGTH:].copy()  # shape: (SEQ_LENGTH, 1)

predictions_scaled = []

with torch.no_grad():
    for _ in range(N_PREDICT):
        # Prepare input: shape (1, SEQ_LENGTH, 1)
        input_tensor = torch.tensor(last_window).unsqueeze(0)  # (1, SEQ_LENGTH, 1)
        
        # Predict next step
        pred_scaled = model(input_tensor).item()  # scalar
        predictions_scaled.append(pred_scaled)
        
        # Slide the window: drop oldest, append new prediction
        last_window = np.roll(last_window, -1, axis=0)
        last_window[-1, 0] = pred_scaled

# --- 7. Inverse Transform (scale back to original range) ---
predictions_scaled = np.array(predictions_scaled)
predictions = predictions_scaled * (max_val - min_val) + min_val

print(f"Predicted {N_PREDICT} future data points.")
print(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")

# --- 8. Plot ---
# Show last 100 points of Xtrain + 200 predicted points
N_HISTORY = 100

history_raw = data[-N_HISTORY:, 0] * (max_val - min_val) + min_val  # inverse transform

x_history = np.arange(-N_HISTORY, 0)          # -100 ~ -1
x_future  = np.arange(0, N_PREDICT)            #  0   ~ 199

plt.figure(figsize=(14, 5))
plt.plot(x_history, history_raw, label='Xtrain (last 100 pts)', color='steelblue')
plt.plot(x_future,  predictions, label='Recursive Prediction (200 pts)', color='tomato', linestyle='--')
plt.axvline(x=0, color='gray', linestyle=':', linewidth=1.5, label='Prediction Start')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('LSTM Recursive Forecast: 200 Steps Ahead')
plt.legend()
plt.tight_layout()
plt.savefig('recursive_prediction.png', dpi=150)
plt.show()
print("Plot saved to recursive_prediction.png")
