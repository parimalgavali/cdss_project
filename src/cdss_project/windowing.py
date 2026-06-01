from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from cdss_project.features import FEATURE_COLUMNS

WindowType = Literal["fixed", "expanding"]


@dataclass(frozen=True)
class WindowDataset:
    X: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame
    feature_columns: tuple[str, ...]
    window_type: WindowType
    mask: np.ndarray | None = None


@dataclass(frozen=True)
class WindowBuildSummary:
    window_type: WindowType
    window_length: int | None
    min_window_length: int | None
    row_count: int
    stay_count: int
    window_count: int
    positive_window_count: int
    negative_window_count: int
    feature_count: int
    output_path: Path | None = None


def build_fixed_windows(
    frame: pd.DataFrame,
    *,
    window_length: int,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    label_column: str = "stay_has_sepsis",
    include_partial: bool = False,
    pad_value: float = 0.0,
) -> WindowDataset:
    if window_length <= 0:
        raise ValueError("window_length must be positive")
    _validate_window_columns(frame, feature_columns=feature_columns, label_column=label_column)

    feature_columns = tuple(feature_columns)
    windows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict[str, object]] = []

    for stay_id, stay in _iter_stays_chronologically(frame):
        if len(stay) < window_length and not include_partial:
            continue
        stay_label = _stay_label(stay, label_column=label_column)
        stay_features = _feature_matrix(stay, feature_columns=feature_columns)
        if include_partial:
            ranges = [
                (max(0, end_index - window_length + 1), end_index + 1)
                for end_index in range(len(stay))
            ]
        else:
            ranges = [
                (start_index, start_index + window_length)
                for start_index in range(0, len(stay) - window_length + 1)
            ]
        for start_index, end_index in ranges:
            window = stay.iloc[start_index:end_index]
            observed_length = end_index - start_index
            if include_partial:
                padded = np.full(
                    (window_length, len(feature_columns)),
                    fill_value=pad_value,
                    dtype=np.float32,
                )
                padded[:observed_length, :] = stay_features[start_index:end_index]
                mask = np.zeros(window_length, dtype=bool)
                mask[:observed_length] = True
                windows.append(padded)
                masks.append(mask)
            else:
                windows.append(stay_features[start_index:end_index])
            labels.append(stay_label)
            metadata.append(
                _metadata_row(
                    stay_id=stay_id,
                    window_type="fixed",
                    window_length=window_length,
                    window=window,
                    label=stay_label,
                    observed_window_length=observed_length,
                )
            )

    return WindowDataset(
        X=_stack_or_empty(windows, width=window_length, feature_count=len(feature_columns)),
        y=np.array(labels, dtype=np.int64),
        metadata=pd.DataFrame(metadata),
        feature_columns=feature_columns,
        window_type="fixed",
        mask=_stack_mask_or_empty(masks, width=window_length) if include_partial else None,
    )


def build_expanding_windows(
    frame: pd.DataFrame,
    *,
    min_window_length: int,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    label_column: str = "stay_has_sepsis",
    pad_value: float = 0.0,
    max_window_length: int | None = None,
) -> WindowDataset:
    if min_window_length <= 0:
        raise ValueError("min_window_length must be positive")
    _validate_window_columns(frame, feature_columns=feature_columns, label_column=label_column)

    feature_columns = tuple(feature_columns)
    stays = list(_iter_stays_chronologically(frame))
    if max_window_length is None:
        max_window_length = max((len(stay) for _, stay in stays), default=min_window_length)
    if max_window_length < min_window_length:
        raise ValueError("max_window_length must be at least min_window_length")

    windows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict[str, object]] = []

    for stay_id, stay in stays:
        if len(stay) < min_window_length:
            continue
        stay_label = _stay_label(stay, label_column=label_column)
        stay_features = _feature_matrix(stay, feature_columns=feature_columns)
        largest_length = min(len(stay), max_window_length)
        for window_length in range(min_window_length, largest_length + 1):
            window = stay.iloc[:window_length]
            padded = np.full(
                (max_window_length, len(feature_columns)),
                fill_value=pad_value,
                dtype=np.float32,
            )
            values = stay_features[:window_length]
            padded[:window_length, :] = values
            mask = np.zeros(max_window_length, dtype=bool)
            mask[:window_length] = True

            windows.append(padded)
            masks.append(mask)
            labels.append(stay_label)
            metadata.append(
                _metadata_row(
                    stay_id=stay_id,
                    window_type="expanding",
                    window_length=window_length,
                    window=window,
                    label=stay_label,
                    observed_window_length=window_length,
                )
            )

    return WindowDataset(
        X=_stack_or_empty(windows, width=max_window_length, feature_count=len(feature_columns)),
        y=np.array(labels, dtype=np.int64),
        metadata=pd.DataFrame(metadata),
        feature_columns=feature_columns,
        window_type="expanding",
        mask=_stack_mask_or_empty(masks, width=max_window_length),
    )


def build_window_summary(
    frame: pd.DataFrame,
    dataset: WindowDataset,
    *,
    window_length: int | None = None,
    min_window_length: int | None = None,
    output_path: Path | None = None,
) -> WindowBuildSummary:
    positive_count = int(dataset.y.sum()) if len(dataset.y) else 0
    return WindowBuildSummary(
        window_type=dataset.window_type,
        window_length=window_length,
        min_window_length=min_window_length,
        row_count=len(frame),
        stay_count=int(frame["stay_id"].nunique()) if "stay_id" in frame.columns else 0,
        window_count=len(dataset.y),
        positive_window_count=positive_count,
        negative_window_count=int(len(dataset.y) - positive_count),
        feature_count=len(dataset.feature_columns),
        output_path=output_path,
    )


def summarize_window_plan(
    frame: pd.DataFrame,
    *,
    window_type: WindowType,
    window_length: int | None = None,
    min_window_length: int | None = None,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    label_column: str = "stay_has_sepsis",
    include_partial: bool = False,
) -> WindowBuildSummary:
    _validate_window_columns(frame, feature_columns=feature_columns, label_column=label_column)
    window_count = 0
    positive_count = 0

    for _, stay in _iter_stays_chronologically(frame):
        stay_label = _stay_label(stay, label_column=label_column)
        stay_length = len(stay)
        if window_type == "fixed":
            if window_length is None:
                raise ValueError("window_length is required for fixed windows")
            if include_partial:
                stay_window_count = stay_length
            else:
                stay_window_count = max(0, stay_length - window_length + 1)
        elif window_type == "expanding":
            if min_window_length is None:
                raise ValueError("min_window_length is required for expanding windows")
            stay_window_count = max(0, stay_length - min_window_length + 1)
        else:
            raise ValueError(f"Unknown window type: {window_type}")
        window_count += stay_window_count
        positive_count += stay_window_count * stay_label

    return WindowBuildSummary(
        window_type=window_type,
        window_length=window_length,
        min_window_length=min_window_length,
        row_count=len(frame),
        stay_count=int(frame["stay_id"].nunique()),
        window_count=int(window_count),
        positive_window_count=int(positive_count),
        negative_window_count=int(window_count - positive_count),
        feature_count=len(feature_columns),
        output_path=None,
    )


def write_window_dataset(
    dataset: WindowDataset,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use overwrite=True to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "X": dataset.X,
        "y": dataset.y,
        "feature_columns": np.array(dataset.feature_columns),
        "metadata_json": np.array(dataset.metadata.to_json(orient="records")),
        "window_type": np.array(dataset.window_type),
    }
    if dataset.mask is not None:
        arrays["mask"] = dataset.mask
    np.savez_compressed(output_path, **arrays)


def format_window_summary(summary: WindowBuildSummary) -> str:
    title = "Window build summary"
    lines = [
        title,
        "=" * len(title),
        f"Window type: {summary.window_type}",
        f"Rows read: {summary.row_count}",
        f"Unique stays: {summary.stay_count}",
        f"Feature columns: {summary.feature_count}",
        f"Total windows: {summary.window_count}",
        (
            "Window labels: "
            f"positive={summary.positive_window_count}, "
            f"negative={summary.negative_window_count}"
        ),
    ]
    if summary.window_length is not None:
        lines.append(f"Window length: {summary.window_length}")
    if summary.min_window_length is not None:
        lines.append(f"Minimum window length: {summary.min_window_length}")
    if summary.output_path is None:
        lines.append("No output files were written.")
    else:
        lines.append(f"Output file: {summary.output_path}")
    return "\n".join(lines)


def _validate_window_columns(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    label_column: str,
) -> None:
    required = ("stay_id", "hour_before", label_column, *feature_columns)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing columns: {', '.join(missing)}")


def _iter_stays_chronologically(frame: pd.DataFrame):
    sorted_frame = frame.sort_values(["stay_id", "hour_before"], ascending=[True, False])
    yield from sorted_frame.groupby("stay_id", sort=True)


def _stay_label(stay: pd.DataFrame, *, label_column: str) -> int:
    observed = stay[label_column].dropna().unique()
    if len(observed) == 0:
        raise ValueError(f"Stay {stay['stay_id'].iloc[0]} has no label in {label_column}")
    if len(set(int(value) for value in observed)) > 1:
        raise ValueError(f"Stay {stay['stay_id'].iloc[0]} has conflicting window labels")
    return int(observed[0])


def _feature_matrix(window: pd.DataFrame, *, feature_columns: Sequence[str]) -> np.ndarray:
    return (
        window.loc[:, list(feature_columns)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )


def _metadata_row(
    *,
    stay_id: object,
    window_type: WindowType,
    window_length: int,
    window: pd.DataFrame,
    label: int,
    observed_window_length: int,
) -> dict[str, object]:
    start_hour = int(window["hour_before"].iloc[0])
    end_hour = int(window["hour_before"].iloc[-1])
    return {
        "stay_id": int(stay_id),
        "window_type": window_type,
        "window_length": window_length,
        "observed_window_length": observed_window_length,
        "window_start_hour_before": start_hour,
        "window_end_hour_before": end_hour,
        "prediction_hour_before": end_hour,
        "label": label,
    }


def _stack_or_empty(
    windows: list[np.ndarray],
    *,
    width: int,
    feature_count: int,
) -> np.ndarray:
    if not windows:
        return np.empty((0, width, feature_count), dtype=np.float32)
    return np.stack(windows).astype(np.float32)


def _stack_mask_or_empty(masks: list[np.ndarray], *, width: int) -> np.ndarray:
    if not masks:
        return np.empty((0, width), dtype=bool)
    return np.stack(masks).astype(bool)
