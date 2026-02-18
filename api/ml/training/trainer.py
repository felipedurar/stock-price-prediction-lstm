import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

from api.ml.architectures.lstm_regressor import LSTMRegressor
from api.ml.preprocessing.sequences import SequenceDataset

def train_model(
    X_train, y_train, X_val, y_val,
    hidden_size=64, num_layers=2, dropout=0.2,
    lr=1e-3, epochs=20, batch_size=32,
    device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    model = LSTMRegressor(input_size=X_train.shape[-1], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        # Early best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_loss

