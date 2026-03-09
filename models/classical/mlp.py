"""
models/classical/mlp.py

3-layer MLP using PyTorch, wrapped to expose sklearn-style fit/predict interface.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin


class _MLPNet(nn.Module):
    def __init__(self, input_dim: int, hidden: tuple = (128, 64, 32), dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]  # logit output
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible MLP classifier backed by PyTorch.

    Exposes predict() and predict_proba() for uniform downstream usage.
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden: tuple = (128, 64, 32),
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 256,
        patience: int = 10,
        device: str = "cpu",
    ):
        self.input_dim = input_dim
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device
        self.model_ = None
        self.train_losses_ = []
        self.val_losses_ = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        device = torch.device(self.device)
        self.model_ = _MLPNet(self.input_dim, self.hidden, self.dropout).to(device)

        # Class imbalance weighting
        pos_weight = torch.tensor(
            [(1 - y_train.mean()) / y_train.mean()], dtype=torch.float32
        ).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5,
        )

        X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_t = torch.tensor(y_train, dtype=torch.float32).to(device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model_.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model_(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            self.train_losses_.append(epoch_loss / len(X_train))

            if X_val is not None:
                val_loss = self._compute_loss(X_val, y_val, criterion, device)
                self.val_losses_.append(val_loss)
                scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_best()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"[MLP] Early stopping at epoch {epoch+1}")
                        self._load_best()
                        break

        return self

    def _compute_loss(self, X, y, criterion, device):
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(device)
            y_t = torch.tensor(y, dtype=torch.float32).to(device)
            return criterion(self.model_(X_t), y_t).item()

    def _save_best(self):
        self._best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}

    def _load_best(self):
        if hasattr(self, "_best_state"):
            self.model_.load_state_dict(self._best_state)

    def predict_proba(self, X) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            logits = self.model_(X_t).numpy()
        probs = 1 / (1 + np.exp(-logits))
        return np.column_stack([1 - probs, probs])

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def classes_(self):
        return np.array([0, 1])


def build_mlp(input_dim: int, **kwargs) -> MLPClassifier:
    return MLPClassifier(input_dim=input_dim, **kwargs)
