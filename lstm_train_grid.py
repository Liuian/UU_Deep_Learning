import scipy.io
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import json
from datetime import datetime

# --- 1. Hyperparameters & Configuration ---
FILE_PATH = "Xtrain.mat"
TRAIN_SPLIT = 0.8     # 80% for training, 20% for testing
BATCH_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 2
EPOCHS = 50

torch.manual_seed(42)
np.random.seed(42)

# Grid search parameters
SEQ_LENGTH_LIST = [10, 15, 20, 25]
LEARNING_RATE_LIST = [0.0001, 0.0005, 0.001, 0.005]

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

# --- 5. Training Function ---
def train_and_evaluate(seq_length, learning_rate):
    """
    Train LSTM model with given hyperparameters and return evaluation metrics.
    
    Args:
        seq_length: Sequence length for creating sequences
        learning_rate: Learning rate for optimizer
    
    Returns:
        dict: Dictionary containing hyperparameters and metrics
    """
    # Create sequences with given seq_length
    X, y = create_sequences(data, seq_length)
    
    # Train/Test Split
    split_idx = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test)
    
    # Create DataLoader
    def seed_worker(worker_id):
        np.random.seed(42)

    g = torch.Generator()
    g.manual_seed(42)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker, generator=g)
    
    # Create model
    model = TimeSeriesLSTM(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)
        test_loss_mse = criterion(test_predictions, y_test)
        mae_criterion = nn.L1Loss()
        test_loss_mae = mae_criterion(test_predictions, y_test)
    
    # Inverse transform to original scale
    test_predictions_original = test_predictions.numpy() * (max_val - min_val) + min_val
    y_test_original = y_test.numpy() * (max_val - min_val) + min_val
    mae_original = np.mean(np.abs(test_predictions_original - y_test_original))
    
    return {
        "seq_length": seq_length,
        "learning_rate": learning_rate,
        "num_sequences": len(X),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "mse_scaled": float(test_loss_mse.item()),
        "mae_scaled": float(test_loss_mae.item()),
        "mae_original": float(mae_original)
    }

# --- 6. Grid Search ---
print("\nStarting Grid Search...")
print(f"Testing {len(SEQ_LENGTH_LIST)} seq_lengths × {len(LEARNING_RATE_LIST)} learning_rates = {len(SEQ_LENGTH_LIST) * len(LEARNING_RATE_LIST)} combinations\n")

results = []
for i, seq_length in enumerate(SEQ_LENGTH_LIST):
    for j, learning_rate in enumerate(LEARNING_RATE_LIST):
        combination_num = i * len(LEARNING_RATE_LIST) + j + 1
        print(f"[{combination_num}/{len(SEQ_LENGTH_LIST) * len(LEARNING_RATE_LIST)}] Training with SEQ_LENGTH={seq_length}, LEARNING_RATE={learning_rate}...")
        result = train_and_evaluate(seq_length, learning_rate)
        results.append(result)
        print(f"  MSE (Scaled): {result['mse_scaled']:.6f}, MAE (Original): {result['mae_original']:.4f}\n")

# --- 7. Save Results to JSON ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"grid_search_results_{timestamp}.json"

output_data = {
    "timestamp": datetime.now().isoformat(),
    "grid_search_config": {
        "seq_length_list": SEQ_LENGTH_LIST,
        "learning_rate_list": LEARNING_RATE_LIST,
        "batch_size": BATCH_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "epochs": EPOCHS,
        "total_combinations": len(SEQ_LENGTH_LIST) * len(LEARNING_RATE_LIST)
    },
    "results": results,
    "best_result": min(results, key=lambda x: x['mse_scaled'])
}

with open(output_filename, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\nGrid search completed!")
print(f"Results saved to: {output_filename}")
print(f"\nBest result:")
print(f"  SEQ_LENGTH: {output_data['best_result']['seq_length']}")
print(f"  LEARNING_RATE: {output_data['best_result']['learning_rate']}")
print(f"  MSE (Scaled): {output_data['best_result']['mse_scaled']:.6f}")
print(f"  MAE (Original): {output_data['best_result']['mae_original']:.4f}")
