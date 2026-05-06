import json
from datetime import datetime

import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, series, window_size):
        self.series = series
        self.window_size = window_size

    def __len__(self):
        return len(self.series) - self.window_size

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.window_size]
        y = self.series[idx + self.window_size]
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y


class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        last_out = out[:, -1, :]
        y_pred = self.fc(last_out)
        return y_pred.squeeze(-1)


def load_series(mat_file="Xtrain.mat"):
    data = sio.loadmat(mat_file)
    series = np.array(data["Xtrain"]).squeeze().astype(np.float32)
    scaler = StandardScaler()
    series_scaled = scaler.fit_transform(series.reshape(-1, 1)).squeeze()
    return series, series_scaled, scaler


def split_series(series_scaled, train_ratio=0.8):
    split_idx = int(len(series_scaled) * train_ratio)
    return series_scaled[:split_idx], series_scaled[split_idx:]


def train_gru(
    train_series,
    val_series,
    window_size=15,
    hidden_size=64,
    num_layers=1,
    lr=0.001,
    batch_size=64,
    epochs=50,
    patience=10,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TimeSeriesDataset(train_series, window_size)
    val_dataset = TimeSeriesDataset(val_series, window_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GRUModel(hidden_size=hidden_size, num_layers=num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)

        avg_train_loss = train_loss / len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * x_batch.size(0)

        avg_val_loss = val_loss / len(val_dataset)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_model_state)
    return model, best_val_loss


def one_step_predict(model, series_scaled, window_size):
    device = next(model.parameters()).device
    model.eval()

    preds = []
    targets = []
    with torch.no_grad():
        for i in range(len(series_scaled) - window_size):
            x = series_scaled[i : i + window_size]
            y_true = series_scaled[i + window_size]
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            y_pred = model(x_tensor).item()
            preds.append(y_pred)
            targets.append(y_true)

    return np.array(preds), np.array(targets)


def save_results_to_json(results, config, filename):
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "results": results,
        "best_result": min(results, key=lambda x: x["val_mse"]),
    }
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)


def run_grid_search():
    _, series_scaled, scaler = load_series()
    train_series, val_series = split_series(series_scaled, train_ratio=0.8)

    grid = {
        "window_size": [10, 15, 20],
        "hidden_size": [32, 64],
        "num_layers": [1, 2],
        "lr": [0.001, 0.0005],
        "batch_size": [32, 64],
    }
    fixed_params = {
        "epochs": 50,
        "patience": 10,
        "train_ratio": 0.8,
    }

    combinations = []
    for window_size in grid["window_size"]:
        for hidden_size in grid["hidden_size"]:
            for num_layers in grid["num_layers"]:
                for lr in grid["lr"]:
                    for batch_size in grid["batch_size"]:
                        combinations.append(
                            {
                                "window_size": window_size,
                                "hidden_size": hidden_size,
                                "num_layers": num_layers,
                                "lr": lr,
                                "batch_size": batch_size,
                            }
                        )

    results = []
    for idx, params in enumerate(combinations, start=1):
        print(f"[{idx}/{len(combinations)}] Testing {params}")
        model, best_val_loss = train_gru(
            train_series=train_series,
            val_series=val_series,
            window_size=params["window_size"],
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            lr=params["lr"],
            batch_size=params["batch_size"],
            epochs=fixed_params["epochs"],
            patience=fixed_params["patience"],
        )

        preds_scaled, targets_scaled = one_step_predict(model, val_series, params["window_size"])
        preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).squeeze()
        targets = scaler.inverse_transform(targets_scaled.reshape(-1, 1)).squeeze()
        val_mae = mean_absolute_error(targets, preds)
        val_mse = mean_squared_error(targets, preds)

        result = {
            "window_size": params["window_size"],
            "hidden_size": params["hidden_size"],
            "num_layers": params["num_layers"],
            "lr": params["lr"],
            "batch_size": params["batch_size"],
            "best_val_loss": float(best_val_loss),
            "val_mae": float(val_mae),
            "val_mse": float(val_mse),
        }
        results.append(result)
        print(f"    val_mse={val_mse:.6f}, val_mae={val_mae:.6f}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"gru_grid_search_results_{timestamp}.json"
    config = {**grid, **fixed_params}
    save_results_to_json(results, config, output_filename)
    print(f"Grid search finished. Results saved to {output_filename}")


if __name__ == "__main__":
    run_grid_search()
