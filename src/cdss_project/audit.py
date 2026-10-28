from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from cdss_project.features import FEATURE_COLUMNS, LABEL_COLUMN


class ConflictingDuplicateRowsError(ValueError):
    """Raised when duplicate stay-hour rows contain different values."""


@dataclass(frozen=True)
class DuplicateSummary:
    duplicate_row_count: int
    duplicate_group_count: int


def structural_empty_rows(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    label_column: str = LABEL_COLUMN,
) -> pd.Series:
    available_features = [column for column in feature_columns if column in frame.columns]
    if label_column not in frame.columns:
        raise KeyError(f"Missing label column: {label_column}")
    if not available_features:
        return frame[label_column].isna()
    return frame[label_column].isna() & frame[available_features].isna().all(axis=1)


def derive_stay_outcomes(
    frame: pd.DataFrame,
    *,
    stay_column: str = "stay_id",
    label_column: str = LABEL_COLUMN,
) -> pd.DataFrame:
    missing = [column for column in (stay_column, label_column) if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing columns: {', '.join(missing)}")

    observed = frame.dropna(subset=[label_column])
    observed_counts = observed.groupby(stay_column)[label_column].size()
    all_stays = pd.Index(frame[stay_column].dropna().unique(), name=stay_column)
    unlabeled_stays = all_stays.difference(observed_counts.index)
    if len(unlabeled_stays) > 0:
        examples = ", ".join(str(value) for value in unlabeled_stays[:5])
        raise ValueError(f"Stays have no observed sepsis labels: {examples}")

    outcomes = (
        observed.groupby(stay_column, as_index=False)[label_column]
        .max()
        .rename(columns={label_column: "stay_has_sepsis"})
    )
    outcomes["stay_has_sepsis"] = outcomes["stay_has_sepsis"].astype(int)
    outcomes["observed_label_count"] = outcomes[stay_column].map(observed_counts).astype(int)
    return outcomes.sort_values(stay_column).reset_index(drop=True)


def duplicate_summary(
    frame: pd.DataFrame,
    *,
    key_columns: Sequence[str] = ("stay_id", "hour_before"),
) -> DuplicateSummary:
    duplicate_mask = frame.duplicated(list(key_columns), keep=False)
    if not duplicate_mask.any():
        return DuplicateSummary(duplicate_row_count=0, duplicate_group_count=0)

    duplicate_rows = frame.loc[duplicate_mask, list(key_columns)]
    return DuplicateSummary(
        duplicate_row_count=int(duplicate_mask.sum()),
        duplicate_group_count=int(duplicate_rows.drop_duplicates().shape[0]),
    )


def deduplicate_exact_stay_hours(
    frame: pd.DataFrame,
    *,
    key_columns: Sequence[str] = ("stay_id", "hour_before"),
) -> pd.DataFrame:
    key_columns = tuple(key_columns)
    duplicate_mask = frame.duplicated(list(key_columns), keep=False)
    if not duplicate_mask.any():
        return frame.copy().reset_index(drop=True)

    conflicts: list[str] = []
    for key, group in frame.loc[duplicate_mask].groupby(list(key_columns), dropna=False):
        if group.drop_duplicates().shape[0] > 1:
            if not isinstance(key, tuple):
                key = (key,)
            key_text = ", ".join(f"{column}={value}" for column, value in zip(key_columns, key))
            conflicts.append(key_text)

    if conflicts:
        examples = "; ".join(conflicts[:5])
        raise ConflictingDuplicateRowsError(f"Conflicting duplicate stay-hour rows: {examples}")

    return frame.drop_duplicates(subset=list(key_columns), keep="first").reset_index(drop=True)
