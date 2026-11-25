"""
DeepSurv Neural Network Estimator (PyTorch)
===========================================
Defines the PyTorchDeepSurvClassifier estimator for multi-epoch gradient optimization
in the StacK-Risk ensemble. Placed in a dedicated module so model serialization
(joblib/pickle) resolves reliably across all CLI runners, APIs, and validation modules.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

log = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class PyTorchDeepSurvClassifier(BaseEstimator, ClassifierMixin):
        """
        Multi-Epoch Deep Survival & Recurrence Risk Neural Network (PyTorch).
        Implements multi-epoch gradient optimization with AdamW, Cosine Annealing LR scheduling,
        Mish non-linear activations, batch normalization, and dropout regularization.
        """
        def __init__(
            self,
            hidden_dim: int = 128,
            dropout: float = 0.3,
            epochs: int = 60,
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
            random_state: int = 42
        ):
            self.hidden_dim = hidden_dim
            self.dropout = dropout
            self.epochs = epochs
            self.lr = lr
            self.weight_decay = weight_decay
            self.random_state = random_state
            self.classes_ = np.array([0, 1])
            self.net_ = None

        def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
            torch.manual_seed(self.random_state)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            in_dim = X.shape[1]

            self.net_ = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.Mish(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.BatchNorm1d(self.hidden_dim // 2),
                nn.Mish(),
                nn.Dropout(self.dropout / 2),
                nn.Linear(self.hidden_dim // 2, 1)
            ).to(device)

            optimizer = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(self.epochs, 1), eta_min=1e-5)

            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
            if sample_weight is not None:
                w_tensor = torch.tensor(sample_weight, dtype=torch.float32).to(device)
            else:
                w_tensor = torch.ones_like(y_tensor)

            self.net_.train()
            log_interval = max(1, self.epochs // 5)
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                logits = self.net_(X_tensor).squeeze(-1)
                loss = (F.binary_cross_entropy_with_logits(logits, y_tensor, reduction="none") * w_tensor).mean()
                loss.backward()
                optimizer.step()
                scheduler.step()

                if (epoch + 1) % log_interval == 0 or epoch == self.epochs - 1:
                    cur_lr = scheduler.get_last_lr()[0]
                    log.info("    [PyTorch DeepSurv Epoch %2d/%2d] Loss: %.4f | LR: %.6f", epoch + 1, self.epochs, float(loss.item()), cur_lr)

            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            if self.net_ is None:
                n = len(X)
                return np.ones((n, 2)) * 0.5
            self.net_.eval()
            device = next(self.net_.parameters()).device
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                logits = self.net_(X_tensor).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                return np.column_stack([1.0 - probs, probs])

        def predict(self, X: np.ndarray) -> np.ndarray:
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
else:
    class PyTorchDeepSurvClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, *args, **kwargs):
            self.classes_ = np.array([0, 1])
            self.net_ = None

        def fit(self, X, y, sample_weight=None):
            return self

        def predict_proba(self, X):
            return np.ones((len(X), 2)) * 0.5

        def predict(self, X):
            return np.zeros(len(X), dtype=int)
