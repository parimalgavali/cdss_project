from __future__ import annotations

import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cdss_project.windowing import WindowDataset


@dataclass(frozen=True)
class StaySplitIndices:
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray


def load_window_dataset(path: Path) -> WindowDataset:
    loaded = np.load(path, allow_pickle=True)
    metadata = pd.read_json(StringIO(str(loaded["metadata_json"].item())), orient="records")
    mask = loaded["mask"] if "mask" in loaded.files else None
    return WindowDataset(
        X=loaded["X"],
        y=loaded["y"].astype(np.int64),
        metadata=metadata,
        feature_columns=tuple(str(value) for value in loaded["feature_columns"].tolist()),
        window_type=str(loaded["window_type"].item()),  # type: ignore[arg-type]
        mask=mask,
    )


def subset_window_dataset_by_stays(
    dataset: WindowDataset,
    *,
    max_stays: int,
) -> WindowDataset:
    stay_ids = sorted(dataset.metadata["stay_id"].dropna().unique())[:max_stays]
    keep = np.flatnonzero(dataset.metadata["stay_id"].isin(stay_ids).to_numpy())
    mask = dataset.mask[keep] if dataset.mask is not None else None
    return WindowDataset(
        X=dataset.X[keep],
        y=dataset.y[keep],
        metadata=dataset.metadata.iloc[keep].reset_index(drop=True),
        feature_columns=dataset.feature_columns,
        window_type=dataset.window_type,
        mask=mask,
    )


def split_indices_by_stay(
    metadata: pd.DataFrame,
    *,
    test_size: float = 0.2,
    val_size: float = 0.16,
    random_state: int = 42,
) -> StaySplitIndices:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not 0 <= val_size < 1:
        raise ValueError("val_size must be between 0 and 1")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be less than 1")
    if "stay_id" not in metadata.columns:
        raise KeyError("Missing metadata column: stay_id")

    stay_ids = np.array(sorted(metadata["stay_id"].dropna().unique()))
    rng = np.random.default_rng(random_state)
    shuffled = stay_ids.copy()
    rng.shuffle(shuffled)
    test_count = max(1, int(round(len(shuffled) * test_size)))
    val_count = max(1, int(round(len(shuffled) * val_size))) if val_size else 0

    test_stays = set(shuffled[:test_count])
    val_stays = set(shuffled[test_count : test_count + val_count])
    train_stays = set(shuffled[test_count + val_count :])

    stay_values = metadata["stay_id"].to_numpy()
    return StaySplitIndices(
        train_indices=np.flatnonzero(np.isin(stay_values, list(train_stays))),
        val_indices=np.flatnonzero(np.isin(stay_values, list(val_stays))),
        test_indices=np.flatnonzero(np.isin(stay_values, list(test_stays))),
    )


def flatten_for_classical_models(X: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    flattened = X.reshape(X.shape[0], -1)
    if mask is None:
        return flattened
    return np.concatenate([flattened, mask.astype(np.float32)], axis=1)


def choose_threshold_by_f1(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    candidates: Iterable[float] | None = None,
) -> float:
    if candidates is None:
        candidates = np.linspace(0.05, 0.95, 19)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        score = f1_score(y_true, probabilities >= threshold, zero_division=0)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def compute_binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities).astype(float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = _confusion_values(y_true, predictions)
    return {
        "auroc": _safe_score(roc_auc_score, y_true, probabilities),
        "auprc": _safe_score(average_precision_score, y_true, probabilities),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "sensitivity": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "threshold": float(threshold),
        "true_negative": float(tn),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "true_positive": float(tp),
        "positive_count": float(np.sum(y_true == 1)),
        "negative_count": float(np.sum(y_true == 0)),
    }


def compute_metrics_by_hour(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    hours = sorted(metadata["prediction_hour_before"].dropna().unique(), reverse=True)
    for hour in hours:
        mask = metadata["prediction_hour_before"].to_numpy() == hour
        metrics = compute_binary_metrics(y_true[mask], probabilities[mask], threshold=threshold)
        metrics["prediction_hour_before"] = int(hour)
        rows.append(metrics)
    return pd.DataFrame(rows)


def write_predictions_by_hour(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    output_dir: Path,
    threshold: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = (probabilities >= threshold).astype(int)
    frame = metadata.copy()
    frame["true_label"] = y_true
    frame["predicted_probability"] = probabilities
    frame["predicted_label"] = predictions
    for hour, group in frame.groupby("prediction_hour_before", sort=False):
        group.to_csv(output_dir / f"hour_{int(hour):02d}.csv", index=False)
    (output_dir / "prediction_schema.json").write_text(
        json.dumps(
            {
                "threshold": float(threshold),
                "columns": list(frame.columns),
            },
            indent=2,
        )
    )


def run_classical_experiment(
    dataset: WindowDataset,
    *,
    model_name: str,
    output_dir: Path,
    random_state: int = 42,
    max_train_windows: int | None = None,
) -> dict[str, object]:
    split = split_indices_by_stay(dataset.metadata, random_state=random_state)
    X = flatten_for_classical_models(dataset.X, dataset.mask)
    train_indices = _limit_indices(split.train_indices, max_train_windows, random_state=random_state)

    if model_name == "logistic_regression":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                random_state=random_state,
            ),
        )
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported classical model: {model_name}")

    model.fit(X[train_indices], dataset.y[train_indices])
    val_probabilities = _predict_probabilities(model, X[split.val_indices])
    threshold = choose_threshold_by_f1(dataset.y[split.val_indices], val_probabilities)
    test_probabilities = _predict_probabilities(model, X[split.test_indices])
    return _write_experiment_outputs(
        dataset,
        model_name=model_name,
        model_family="classical",
        output_dir=output_dir,
        test_indices=split.test_indices,
        probabilities=test_probabilities,
        threshold=threshold,
        config={
            "random_state": random_state,
            "max_train_windows": max_train_windows,
            "class_weight": "balanced",
        },
    )


def run_neural_experiment(
    dataset: WindowDataset,
    *,
    model_name: str,
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_size: int = 32,
    random_state: int = 42,
    max_train_windows: int | None = None,
) -> dict[str, object]:
    if model_name not in {"lstm", "gru"}:
        raise ValueError(f"Unsupported neural model: {model_name}")
    torch.manual_seed(random_state)
    split = split_indices_by_stay(dataset.metadata, random_state=random_state)
    train_indices = _limit_indices(split.train_indices, max_train_windows, random_state=random_state)
    device = _choose_torch_device()

    model = _SequenceClassifier(
        input_size=dataset.X.shape[2],
        hidden_size=hidden_size,
        model_name=model_name,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    positive_count = float(np.sum(dataset.y[train_indices] == 1))
    negative_count = float(np.sum(dataset.y[train_indices] == 0))
    pos_weight = torch.tensor([negative_count / max(positive_count, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    loader = _sequence_loader(dataset, train_indices, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for X_batch, mask_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            mask_batch = mask_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch, mask_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()

    val_probabilities = _sequence_probabilities(model, dataset, split.val_indices, batch_size, device)
    threshold = choose_threshold_by_f1(dataset.y[split.val_indices], val_probabilities)
    test_probabilities = _sequence_probabilities(model, dataset, split.test_indices, batch_size, device)
    return _write_experiment_outputs(
        dataset,
        model_name=model_name,
        model_family="neural",
        output_dir=output_dir,
        test_indices=split.test_indices,
        probabilities=test_probabilities,
        threshold=threshold,
        config={
            "random_state": random_state,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "hidden_size": hidden_size,
            "max_train_windows": max_train_windows,
            "device": str(device),
        },
    )


def _write_experiment_outputs(
    dataset: WindowDataset,
    *,
    model_name: str,
    model_family: str,
    output_dir: Path,
    test_indices: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    config: dict[str, object],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dataset.metadata.iloc[test_indices].reset_index(drop=True)
    y_true = dataset.y[test_indices]
    overall = compute_binary_metrics(y_true, probabilities, threshold=threshold)
    by_hour = compute_metrics_by_hour(metadata, y_true, probabilities, threshold=threshold)
    overall_row = {
        "model": model_name,
        "model_family": model_family,
        "window_type": dataset.window_type,
        **overall,
    }
    pd.DataFrame([overall_row]).to_csv(output_dir / "metrics_overall.csv", index=False)
    by_hour.insert(0, "model", model_name)
    by_hour.insert(1, "model_family", model_family)
    by_hour.insert(2, "window_type", dataset.window_type)
    by_hour.to_csv(output_dir / "metrics_by_hour.csv", index=False)
    write_predictions_by_hour(
        metadata,
        y_true,
        probabilities,
        output_dir=output_dir / "predictions_by_hour",
        threshold=threshold,
    )
    model_config = {
        "model": model_name,
        "model_family": model_family,
        "feature_count": len(dataset.feature_columns),
        "time_steps": int(dataset.X.shape[1]),
        "threshold": float(threshold),
        **config,
    }
    (output_dir / "model_config.json").write_text(json.dumps(model_config, indent=2))
    return overall_row


def _predict_probabilities(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def _choose_torch_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _SequenceClassifier(nn.Module):
    def __init__(self, *, input_size: int, hidden_size: int, model_name: str) -> None:
        super().__init__()
        recurrent_cls = nn.LSTM if model_name == "lstm" else nn.GRU
        self.recurrent = recurrent_cls(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            X,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.recurrent(packed)
        if isinstance(hidden, tuple):
            hidden_state = hidden[0][-1]
        else:
            hidden_state = hidden[-1]
        return self.output(hidden_state).squeeze(1)


def _sequence_loader(
    dataset: WindowDataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    mask = _mask_or_full(dataset, indices)
    tensor_dataset = TensorDataset(
        torch.from_numpy(dataset.X[indices]).float(),
        torch.from_numpy(mask).bool(),
        torch.from_numpy(dataset.y[indices].astype(np.float32)),
    )
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)


def _sequence_probabilities(
    model: nn.Module,
    dataset: WindowDataset,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = _sequence_loader(dataset, indices, batch_size=batch_size, shuffle=False)
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for X_batch, mask_batch, _ in loader:
            logits = model(X_batch.to(device), mask_batch.to(device))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities) if probabilities else np.array([], dtype=float)


def _mask_or_full(dataset: WindowDataset, indices: np.ndarray) -> np.ndarray:
    if dataset.mask is not None:
        return dataset.mask[indices]
    return np.ones((len(indices), dataset.X.shape[1]), dtype=bool)


def _limit_indices(
    indices: np.ndarray,
    max_count: int | None,
    *,
    random_state: int,
) -> np.ndarray:
    if max_count is None or len(indices) <= max_count:
        return indices
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(indices, size=max_count, replace=False))


def _safe_score(metric, y_true: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(metric(y_true, probabilities))


def _confusion_values(y_true: np.ndarray, predictions: np.ndarray) -> tuple[int, int, int, int]:
    labels = [0, 1]
    matrix = confusion_matrix(y_true, predictions, labels=labels)
    tn, fp, fn, tp = matrix.ravel()
    return int(tn), int(fp), int(fn), int(tp)
