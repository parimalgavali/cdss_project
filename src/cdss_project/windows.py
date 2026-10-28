"""Sliding-window construction for horizon-wise sepsis prediction.

`hour_before` counts down to the decision point: 30 is the first ICU hour and 1
the last. Window k covers hours k..k+5 after admission, and its prediction
horizon is the `hour_before` of its last row (25 for the first 6-hour window,
1 for the last).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cdss_project.features import FEATURE_COLUMNS

WINDOW_SIZE = 6
HOURS_PER_STAY = 30
STAY_LABEL_COLUMN = "stay_has_sepsis"


@dataclass(frozen=True)
class WindowDataset:
    X: np.ndarray
    y: np.ndarray
    horizon: np.ndarray
    stay_id: np.ndarray
    feature_columns: tuple[str, ...]
    observed_hours: np.ndarray | None = None
    observed_mask: np.ndarray | None = None
    fold: np.ndarray | None = None
    split: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.y.shape[0])


@dataclass(frozen=True)
class WindowSummary:
    window_count: int
    stay_count: int
    window_size: int
    feature_count: int
    positive_window_count: int
    horizon_table: pd.DataFrame


def build_windows(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    window_size: int = WINDOW_SIZE,
    hours_per_stay: int = HOURS_PER_STAY,
    label_column: str = STAY_LABEL_COLUMN,
    min_observed_hours: int = 0,
) -> WindowDataset:
    """Turn a preprocessed hourly table into overlapping fixed-size windows."""
    if not 1 <= window_size <= hours_per_stay:
        raise ValueError(f"window_size must be between 1 and {hours_per_stay}")
    feature_columns = tuple(feature_columns)
    required = ["stay_id", "hour_before", label_column, *feature_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing columns: {', '.join(missing)}")

    non_numeric = [
        column for column in feature_columns if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise TypeError(f"Feature columns must be numeric-encoded: {', '.join(non_numeric)}")
    if frame[list(feature_columns)].isna().any().any():
        raise ValueError("Feature columns contain missing values; run preprocessing first")

    ordered = frame.sort_values(
        ["stay_id", "hour_before"], ascending=[True, False], kind="mergesort"
    ).reset_index(drop=True)
    stay_ids = _validate_hourly_grid(ordered, hours_per_stay=hours_per_stay)
    n_stays = len(stay_ids)

    labels = ordered[label_column].to_numpy().reshape(n_stays, hours_per_stay)
    if pd.isna(labels).any() or (labels != labels[:, :1]).any():
        raise ValueError(f"{label_column} must be present and constant within each stay")

    values = ordered[list(feature_columns)].to_numpy(dtype=np.float32)
    values = values.reshape(n_stays, hours_per_stay, len(feature_columns))
    windows_per_stay = hours_per_stay - window_size + 1
    # sliding_window_view on the hour axis yields (stays, windows, features, hours).
    X = np.lib.stride_tricks.sliding_window_view(values, window_size, axis=1)
    X = np.ascontiguousarray(X.transpose(0, 1, 3, 2)).reshape(
        n_stays * windows_per_stay, window_size, len(feature_columns)
    )

    horizon_per_stay = hours_per_stay - window_size + 1 - np.arange(windows_per_stay)
    dataset_fields: dict[str, np.ndarray | None] = {
        "y": np.repeat(labels[:, 0].astype(np.int8), windows_per_stay),
        "horizon": np.tile(horizon_per_stay.astype(np.int16), n_stays),
        "stay_id": np.repeat(stay_ids.astype(np.int64), windows_per_stay),
        "observed_hours": None,
        "observed_mask": None,
        "fold": None,
        "split": None,
    }

    if "is_padding" in ordered.columns:
        observed = 1 - ordered["is_padding"].to_numpy(dtype=np.int16).reshape(
            n_stays, hours_per_stay
        )
        observed_windows = np.lib.stride_tricks.sliding_window_view(observed, window_size, axis=1)
        dataset_fields["observed_hours"] = observed_windows.sum(axis=2).reshape(-1).astype(np.int8)
        dataset_fields["observed_mask"] = observed_windows.reshape(-1, window_size).astype(np.int8)
    elif min_observed_hours > 0:
        raise KeyError("min_observed_hours requires an is_padding column")

    for column in ("fold", "split"):
        if column in ordered.columns:
            per_stay = ordered[column].to_numpy().reshape(n_stays, hours_per_stay)[:, 0]
            dataset_fields[column] = np.repeat(per_stay, windows_per_stay)

    dataset = WindowDataset(X=X, feature_columns=feature_columns, **dataset_fields)
    if min_observed_hours > 0:
        dataset = filter_windows(dataset, dataset.observed_hours >= min_observed_hours)
    return dataset


def filter_windows(dataset: WindowDataset, mask: np.ndarray) -> WindowDataset:
    def pick(array: np.ndarray | None) -> np.ndarray | None:
        return None if array is None else array[mask]

    return WindowDataset(
        X=dataset.X[mask],
        y=dataset.y[mask],
        horizon=dataset.horizon[mask],
        stay_id=dataset.stay_id[mask],
        feature_columns=dataset.feature_columns,
        observed_hours=pick(dataset.observed_hours),
        observed_mask=pick(dataset.observed_mask),
        fold=pick(dataset.fold),
        split=pick(dataset.split),
    )


def summarize_windows(dataset: WindowDataset) -> WindowSummary:
    table = pd.DataFrame({"horizon": dataset.horizon, "y": dataset.y})
    if dataset.observed_hours is not None:
        table["observed_hours"] = dataset.observed_hours
        table["fully_padded"] = dataset.observed_hours == 0
    aggregations = {"windows": ("y", "size"), "positives": ("y", "sum")}
    if dataset.observed_hours is not None:
        aggregations["mean_observed_hours"] = ("observed_hours", "mean")
        aggregations["fully_padded_windows"] = ("fully_padded", "sum")
    horizon_table = (
        table.groupby("horizon").agg(**aggregations).sort_index(ascending=False).reset_index()
    )
    return WindowSummary(
        window_count=len(dataset),
        stay_count=int(np.unique(dataset.stay_id).size),
        window_size=int(dataset.X.shape[1]) if dataset.X.ndim == 3 else 0,
        feature_count=len(dataset.feature_columns),
        positive_window_count=int(dataset.y.sum()),
        horizon_table=horizon_table,
    )


def format_window_summary(summary: WindowSummary, *, title: str = "Sliding windows") -> str:
    lines = [
        title,
        "=" * len(title),
        f"Windows: {summary.window_count} from {summary.stay_count} stays",
        f"Window size: {summary.window_size} hours, features: {summary.feature_count}",
        f"Positive windows: {summary.positive_window_count}",
        "",
        "Per horizon (hours before decision point):",
        summary.horizon_table.to_string(index=False, float_format=lambda value: f"{value:.2f}"),
    ]
    return "\n".join(lines)


def save_windows(dataset: WindowDataset, path: Path, *, overwrite: bool = False) -> Path:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {path}. Use overwrite=True to replace it.")
    arrays: dict[str, np.ndarray] = {
        "X": dataset.X,
        "y": dataset.y,
        "horizon": dataset.horizon,
        "stay_id": dataset.stay_id,
        "feature_columns": np.array(dataset.feature_columns, dtype=str),
    }
    for name in ("observed_hours", "observed_mask", "fold", "split"):
        value = getattr(dataset, name)
        if value is not None:
            arrays[name] = value.astype(str) if name == "split" else value
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    return path


def load_windows(path: Path) -> WindowDataset:
    with np.load(path, allow_pickle=False) as data:
        optional = {
            name: data[name] if name in data.files else None
            for name in ("observed_hours", "observed_mask", "fold", "split")
        }
        return WindowDataset(
            X=data["X"],
            y=data["y"],
            horizon=data["horizon"],
            stay_id=data["stay_id"],
            feature_columns=tuple(str(value) for value in data["feature_columns"]),
            **optional,
        )


def _validate_hourly_grid(ordered: pd.DataFrame, *, hours_per_stay: int) -> np.ndarray:
    counts = ordered.groupby("stay_id", sort=True).size()
    bad_counts = counts[counts != hours_per_stay]
    if not bad_counts.empty:
        examples = ", ".join(f"{stay}={count}" for stay, count in bad_counts.head(5).items())
        raise ValueError(f"Stays must have exactly {hours_per_stay} hourly rows: {examples}")

    hours = ordered["hour_before"].to_numpy().reshape(len(counts), hours_per_stay)
    expected = np.arange(hours_per_stay, 0, -1)
    bad_rows = ~(hours == expected).all(axis=1)
    if bad_rows.any():
        examples = ", ".join(str(stay) for stay in counts.index[bad_rows][:5])
        raise ValueError(
            f"Stays must cover hour_before {hours_per_stay}..1 exactly once: {examples}"
        )
    return counts.index.to_numpy()
