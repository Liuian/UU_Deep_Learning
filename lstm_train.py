import scipy.io
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# --- 1. Hyperparameters & Configuration ---
FILE_PATH = "Xtrain.mat"
SEQ_LENGTH = 20       # Number of past time steps to look at to predict the next step
TRAIN_SPLIT = 0.8     # 80% for training, 20% for testing
BATCH_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 1
LEARNING_RATE = 0.001
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

# --- 6. Evaluation and Plotting ---
print("\nEvaluating on Test Set...")
model.eval()
with torch.no_grad():
    # Get predictions for the test set
    test_predictions = model(X_test)
    test_loss_mse = criterion(test_predictions, y_test)
    
    # Calculate MAE on scaled data
    mae_criterion = nn.L1Loss()
    test_loss_mae = mae_criterion(test_predictions, y_test)
    
    print(f'Test MSE Loss (Scaled): {test_loss_mse.item():.6f}')
    print(f'Test MAE Loss (Scaled): {test_loss_mae.item():.6f}')

# Inverse transform to get original scale
test_predictions_original = test_predictions.numpy() * (max_val - min_val) + min_val
y_test_original = y_test.numpy() * (max_val - min_val) + min_val

# Calculate MAE on the original scale for better interpretability
mae_original = np.mean(np.abs(test_predictions_original - y_test_original))
print(f'Test MAE (Original Scale): {mae_original:.4f}')

# Plotting the results
try:
    plt.figure(figsize=(12, 5))
    plt.plot(y_test_original, label='Actual Values', color='blue')
    plt.plot(test_predictions_original, label='LSTM Predictions', color='red', linestyle='--')
    plt.title('LSTM Time Series Forecasting')
    plt.xlabel('Time Step (in Test Set)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('lstm_predictions.png')
    print("Plot saved as 'lstm_predictions.png'.")
except Exception as e:
    print(f"Could not generate plot: {e}")
