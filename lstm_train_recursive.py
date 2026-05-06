import scipy.io
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import copy

# --- 1. Hyperparameters & Configuration ---
FILE_PATH = "Xtrain.mat"
SEQ_LENGTH = 10
TRAIN_SPLIT = 0.8
BATCH_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.001
EPOCHS = 100

torch.manual_seed(42)
np.random.seed(42)

# --- 2. Data Loading and Preprocessing ---
def load_and_preprocess_data(filepath):
    data = scipy.io.loadmat(filepath)
    raw_data = data['Xtrain'].astype(np.float32)
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

print("Loading data...")
data, min_val, max_val = load_and_preprocess_data(FILE_PATH)
X, y = create_sequences(data, SEQ_LENGTH)

# --- 3. Train/Validation Split ---
split_idx = int(len(X) * TRAIN_SPLIT)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

print(f"Total sequences: {len(X)}")
print(f"Training shapes:   X={X_train.shape}, y={y_train.shape}")
print(f"Validation shapes: X={X_val.shape},   y={y_val.shape}")

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)
X_val_t   = torch.tensor(X_val)
y_val_t   = torch.tensor(y_val)

def seed_worker(worker_id):
    np.random.seed(42)

g = torch.Generator()
g.manual_seed(42)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                           worker_init_fn=seed_worker, generator=g)

val_dataset = TensorDataset(X_val_t, y_val_t)
val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 4. LSTM Model ---
class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

model     = TimeSeriesLSTM(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. Training Loop with Validation & Best Model Tracking ---
print("\nStarting Training...")
train_losses = []
val_losses   = []

best_val_loss   = float('inf')
best_epoch      = 0
best_model_wts  = copy.deepcopy(model.state_dict())

for epoch in range(EPOCHS):
    # -- Train --
    model.train()
    epoch_train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        preds = model(batch_X)
        loss  = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    avg_train_loss = epoch_train_loss / len(train_loader)

    # -- Validate --
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            preds = model(batch_X)
            loss  = criterion(preds, batch_y)
            epoch_val_loss += loss.item()
    avg_val_loss = epoch_val_loss / len(val_loader)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    # -- Save best model --
    if avg_val_loss < best_val_loss:
        best_val_loss  = avg_val_loss
        best_epoch     = epoch + 1
        best_model_wts = copy.deepcopy(model.state_dict())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}]  "
              f"Train Loss: {avg_train_loss:.6f}  |  Val Loss: {avg_val_loss:.6f}")

print(f"\nBest model at epoch {best_epoch}  (Val Loss: {best_val_loss:.6f})")

# Load best weights for inference
model.load_state_dict(best_model_wts)
model.eval()

# ============================================================
# PLOT 1 — Loss vs Epoch
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss',      color='steelblue')
ax.plot(range(1, EPOCHS + 1), val_losses,   label='Validation Loss', color='tomato')
ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=1.5,
           label=f'Best Epoch ({best_epoch})')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('Training vs Validation Loss')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot1_loss_curve.png', dpi=150)
plt.show()
print("Saved: plot1_loss_curve.png")

# ============================================================
# PLOT 2 — One-step prediction on Validation set
# ============================================================
with torch.no_grad():
    val_preds_scaled = model(X_val_t).numpy().flatten()

# Inverse transform
val_preds_orig = val_preds_scaled * (max_val - min_val) + min_val
val_gt_orig    = y_val_t.numpy().flatten() * (max_val - min_val) + min_val

mae_val = np.mean(np.abs(val_preds_orig - val_gt_orig))
mse_val = np.mean((val_preds_orig - val_gt_orig) ** 2)
print(f"\nValidation  MAE: {mae_val:.4f}  |  MSE: {mse_val:.4f}")

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(val_gt_orig,    label='Ground Truth', color='steelblue', linewidth=1)
ax.plot(val_preds_orig, label='Prediction',   color='tomato',    linewidth=1, linestyle='--')
ax.set_xlabel('Time Step (Validation)')
ax.set_ylabel('Value')
ax.set_title(f'One-Step Prediction on Validation Set  '
             f'(MAE={mae_val:.4f}, MSE={mse_val:.4f})')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot2_onestep_val.png', dpi=150)
plt.show()
print("Saved: plot2_onestep_val.png")

# ============================================================
# PLOT 3 — Recursive 200-step forecast
# ============================================================
print("\nStarting Recursive Prediction...")
N_PREDICT   = 200
N_HISTORY   = 100

last_window = data[-SEQ_LENGTH:].copy()   # (SEQ_LENGTH, 1), scaled
predictions_scaled = []

with torch.no_grad():
    for _ in range(N_PREDICT):
        input_tensor = torch.tensor(last_window).unsqueeze(0)   # (1, SEQ_LENGTH, 1)
        pred_scaled  = model(input_tensor).item()
        predictions_scaled.append(pred_scaled)
        last_window        = np.roll(last_window, -1, axis=0)
        last_window[-1, 0] = pred_scaled

predictions = np.array(predictions_scaled) * (max_val - min_val) + min_val
history_raw = data[-N_HISTORY:, 0] * (max_val - min_val) + min_val

print(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")

x_history = np.arange(-N_HISTORY, 0)
x_future  = np.arange(0, N_PREDICT)

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(x_history, history_raw, label='Xtrain (last 100 pts)', color='steelblue')
ax.plot(x_future,  predictions, label='Recursive Forecast (200 pts)',
        color='tomato', linestyle='--')
ax.axvline(x=0, color='gray', linestyle=':', linewidth=1.5, label='Forecast Start')
ax.set_xlabel('Time Step')
ax.set_ylabel('Value')
ax.set_title('LSTM Recursive Forecast: 200 Steps Ahead')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot3_recursive_forecast.png', dpi=150)
plt.show()
print("Saved: plot3_recursive_forecast.png")

# ============================================================
# (d) Evaluate with Xtest — uncomment after May 8
# ============================================================
# xtest_data = scipy.io.loadmat("Xtest.mat")
# xtest_raw  = xtest_data['Xtest'].astype(np.float32).flatten()
# mae_test   = np.mean(np.abs(predictions - xtest_raw))
# mse_test   = np.mean((predictions - xtest_raw) ** 2)
# print(f"\nTest  MAE: {mae_test:.4f}  |  MSE: {mse_test:.4f}")
#
# fig, ax = plt.subplots(figsize=(14, 4))
# ax.plot(xtest_raw,  label='Ground Truth (Xtest)', color='steelblue')
# ax.plot(predictions, label='Recursive Forecast',  color='tomato', linestyle='--')
# ax.set_title(f'Forecast vs Xtest  (MAE={mae_test:.4f}, MSE={mse_test:.4f})')
# ax.legend(); ax.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('plot4_test_eval.png', dpi=150)
# plt.show()