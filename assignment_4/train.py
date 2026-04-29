import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from minicamels import MiniCamels

from data import create_sequences, normalize_data
from model import LSTMModel
from visualization import plot_observed_vs_predicted, plot_parity


def split_train_val_test(X, y, train_ratio=0.7, val_ratio=0.15):
    n = len(X)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test


def nse(y_true, y_pred):
    numerator = torch.sum((y_true - y_pred) ** 2)
    denominator = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - numerator / denominator


def plot_history(train_losses, val_losses, val_nses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))

    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.plot(epochs, val_nses, label="Validation NSE")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)

    plt.savefig("training_history.png", dpi=300, bbox_inches="tight")
    plt.show()


def train_model():
    # -----------------------------
    # 1. Load data
    # -----------------------------
    ds = MiniCamels()
    basins = ds.basins()

    basin_id = basins.iloc[0]["basin_id"]
    data = ds.open_basin(basin_id)

    X, y = create_sequences(data, seq_len=30)
    X, y, scaler_X, scaler_y = normalize_data(X, y)

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)

    # -----------------------------
    # 2. Convert to torch tensors
    # -----------------------------
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    # -----------------------------
    # 3. Model, loss, optimizer
    # -----------------------------
    model = LSTMModel(
        input_size=5,
        hidden_size=64,
        num_layers=2
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # -----------------------------
    # 4. Training loop
    # -----------------------------
    epochs = 10

    train_losses = []
    val_losses = []
    val_nses = []

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, y_val).item()
            val_nse = nse(y_val, val_pred).item()

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_nses.append(val_nse)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val NSE: {val_nse:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")

    print("Best model saved as best_model.pt")

    # -----------------------------
    # 5. Test evaluation
    # -----------------------------
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()

    with torch.no_grad():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, y_test).item()
        test_nse = nse(y_test, test_pred).item()

    print("\nFinal Test Results")
    print("Test Loss:", round(test_loss, 4))
    print("Test NSE :", round(test_nse, 4))

    # -----------------------------
    # 6. Plot history
    # -----------------------------
    plot_history(train_losses, val_losses, val_nses)


if __name__ == "__main__":
    train_model()