"""LSTM/GRU sepsis models with self-attention, following outline 3.4."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from cdss_project.baselines import BaselineResult, choose_validation_fold
from cdss_project.evaluation import (
    early_detection_summary,
    first_alert_horizons,
    metrics_by_horizon,
    select_threshold,
)
from cdss_project.features import CATEGORICAL_COLUMNS
from cdss_project.windows import WindowDataset

CELL_TYPES: tuple[str, ...] = ("lstm", "gru")


@dataclass(frozen=True)
class SequenceEncoder:
    """Standardize numeric features and one-hot categorical codes per time step."""

    numeric_index: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    categorical_index: np.ndarray
    categories: tuple[np.ndarray, ...]

    @property
    def n_features(self) -> int:
        return len(self.numeric_index) + sum(len(values) for values in self.categories)

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        feature_columns: Sequence[str],
        *,
        categorical_columns: Sequence[str] = CATEGORICAL_COLUMNS,
    ) -> SequenceEncoder:
        names = tuple(feature_columns)
        numeric = np.array([i for i, n in enumerate(names) if n not in categorical_columns], int)
        categorical = np.array([i for i, n in enumerate(names) if n in categorical_columns], int)
        values = X[:, :, numeric].reshape(-1, len(numeric))
        mean = values.mean(axis=0, dtype=np.float64)
        std = values.std(axis=0, dtype=np.float64)
        std[std == 0] = 1.0
        return cls(
            numeric_index=numeric,
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            categorical_index=categorical,
            categories=tuple(np.unique(X[:, :, i]) for i in categorical),
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        parts = [(X[:, :, self.numeric_index] - self.mean) / self.std]
        for index, values in zip(self.categorical_index, self.categories, strict=True):
            parts.append((X[:, :, index][..., None] == values).astype(np.float32))
        return np.concatenate(parts, axis=2).astype(np.float32)


class SepsisRNN(nn.Module):
    """Self-attention -> 3 x (LSTM|GRU, BatchNorm, Dropout) -> Linear(8) -> logit."""

    def __init__(
        self,
        n_features: int,
        *,
        cell: str = "lstm",
        hidden_size: int = 16,
        n_layers: int = 3,
        dense_size: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if cell not in CELL_TYPES:
            raise ValueError(f"Unknown cell {cell!r}; expected one of {CELL_TYPES}")
        recurrent_class = nn.LSTM if cell == "lstm" else nn.GRU
        self.attention = nn.MultiheadAttention(n_features, num_heads=1, batch_first=True)
        self.recurrent = nn.ModuleList()
        self.norms = nn.ModuleList()
        input_size = n_features
        for _ in range(n_layers):
            self.recurrent.append(recurrent_class(input_size, hidden_size, batch_first=True))
            self.norms.append(nn.BatchNorm1d(hidden_size))
            input_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, dense_size),
            nn.BatchNorm1d(dense_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(x, x, x, need_weights=False)
        hidden = x + attended
        for recurrent, norm in zip(self.recurrent, self.norms, strict=True):
            hidden, _ = recurrent(hidden)
            hidden = self.dropout(norm(hidden.transpose(1, 2)).transpose(1, 2))
        return self.head(hidden[:, -1]).squeeze(-1)


@dataclass(frozen=True)
class TrainConfig:
    max_epochs: int = 100
    patience: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    min_delta: float = 0.0
    seed: int = 42
    device: str = "auto"
    eval_batch_size: int = 4096


@dataclass(frozen=True)
class RNNResult(BaselineResult):
    history: tuple[dict[str, float], ...] = ()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig,
) -> list[dict[str, float]]:
    """Weighted BCE + Adam with early stopping on validation loss; restores the best weights."""
    device = resolve_device(config.device)
    model.to(device)
    positives = float(y_train.sum())
    if positives == 0 or positives == len(y_train):
        raise ValueError("Training data needs both classes")
    pos_weight = torch.tensor((len(y_train) - positives) / positives, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(config.seed)
    features = torch.from_numpy(X_train)
    labels = torch.from_numpy(y_train.astype(np.float32))

    history: list[dict[str, float]] = []
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = torch.from_numpy(rng.permutation(len(y_train)))
        total, seen = 0.0, 0
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            if len(batch) < 2:  # BatchNorm needs more than one sample in training mode
                continue
            optimizer.zero_grad()
            loss = loss_fn(model(features[batch].to(device)), labels[batch].to(device))
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            seen += len(batch)

        val_loss = _loss(model, X_val, y_val, loss_fn, device, config.eval_batch_size)
        history.append({"epoch": epoch, "train_loss": total / max(seen, 1), "val_loss": val_loss})
        if val_loss < best_loss - config.min_delta:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def predict_scores(
    model: nn.Module, X: np.ndarray, *, device: torch.device, batch_size: int = 4096
) -> np.ndarray:
    return torch.sigmoid(_logits(model, X, device, batch_size)).numpy()


def run_rnn(
    dataset: WindowDataset,
    cell: str,
    *,
    config: TrainConfig | None = None,
    target_sensitivity: float | None = None,
) -> RNNResult:
    """Train on training folds minus the validation fold; tune threshold on it; score test fold."""
    config = config or TrainConfig()
    if dataset.fold is None or dataset.split is None:
        raise ValueError("Windows need fold and split arrays; rebuild them from the processed CSV")
    split = dataset.split.astype(str)
    test_fold, validation_fold = choose_validation_fold(dataset.fold, split)
    train = (split == "train") & (dataset.fold != validation_fold)
    validation = (split == "train") & (dataset.fold == validation_fold)
    test = split == "test"

    encoder = SequenceEncoder.fit(dataset.X[train], dataset.feature_columns)
    X_val = encoder.transform(dataset.X[validation])
    torch.manual_seed(config.seed)
    model = SepsisRNN(encoder.n_features, cell=cell)
    history = train_model(
        model,
        encoder.transform(dataset.X[train]),
        dataset.y[train],
        X_val,
        dataset.y[validation],
        config,
    )
    device = resolve_device(config.device)
    threshold = select_threshold(
        dataset.y[validation],
        predict_scores(model, X_val, device=device, batch_size=config.eval_batch_size),
        target_sensitivity=target_sensitivity,
    )

    predictions = pd.DataFrame(
        {
            "stay_id": dataset.stay_id[test],
            "horizon": dataset.horizon[test],
            "y": dataset.y[test],
            "score": predict_scores(
                model,
                encoder.transform(dataset.X[test]),
                device=device,
                batch_size=config.eval_batch_size,
            ),
        }
    )
    if dataset.observed_hours is not None:
        predictions["observed_hours"] = dataset.observed_hours[test]

    metrics = metrics_by_horizon(predictions, threshold=threshold)
    metrics.insert(0, "model", cell)
    alerts = first_alert_horizons(predictions, threshold=threshold)
    best_epoch = min(history, key=lambda row: row["val_loss"])["epoch"]
    params: dict[str, Any] = {
        **asdict(config),
        "target_sensitivity": target_sensitivity,
        "input_features": encoder.n_features,
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "device": str(device),
    }
    return RNNResult(
        model_name=cell,
        threshold=threshold,
        validation_fold=validation_fold,
        test_fold=test_fold,
        predictions=predictions,
        metrics=metrics,
        alerts=alerts,
        early_detection=early_detection_summary(alerts),
        params=params,
        history=tuple(history),
    )


@torch.no_grad()
def _logits(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int) -> torch.Tensor:
    model.eval()
    chunks = [
        model(torch.from_numpy(X[start : start + batch_size]).to(device)).cpu()
        for start in range(0, len(X), batch_size)
    ]
    return torch.cat(chunks) if chunks else torch.empty(0)


def _loss(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    loss_fn: nn.Module,
    device: torch.device,
    batch_size: int,
) -> float:
    logits = _logits(model, X, device, batch_size)
    return float(loss_fn(logits.to(device), torch.from_numpy(y.astype(np.float32)).to(device)))
