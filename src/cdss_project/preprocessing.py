from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from cdss_project.audit import deduplicate_exact_stay_hours, derive_stay_outcomes
from cdss_project.features import FEATURE_COLUMNS


@dataclass(frozen=True)
class StaySplit:
    train_stay_ids: tuple[int, ...]
    test_stay_ids: tuple[int, ...]


@dataclass(frozen=True)
class PreprocessingDryRunSummary:
    row_count: int
    stay_count: int
    train_stay_count: int
    test_stay_count: int
    feature_count: int
    missing_before: int
    missing_after: int
    stay_positive_count: int
    stay_negative_count: int


def split_stay_ids(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> StaySplit:
    if "stay_id" not in frame.columns:
        raise KeyError("Missing column: stay_id")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    stay_ids = np.array(sorted(frame["stay_id"].dropna().unique()))
    rng = np.random.default_rng(random_state)
    shuffled = stay_ids.copy()
    rng.shuffle(shuffled)
    test_count = max(1, int(round(len(shuffled) * test_size)))
    test_ids = tuple(sorted(int(value) for value in shuffled[:test_count]))
    train_ids = tuple(sorted(int(value) for value in shuffled[test_count:]))
    return StaySplit(train_stay_ids=train_ids, test_stay_ids=test_ids)


def fit_feature_means(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> dict[str, float]:
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {', '.join(missing)}")

    means: dict[str, float] = {}
    for column in feature_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        mean = numeric.mean(skipna=True)
        if pd.isna(mean):
            raise ValueError(f"Cannot compute global mean for entirely missing feature: {column}")
        means[column] = float(mean)
    return means


def transform_with_professor_imputation(
    frame: pd.DataFrame,
    *,
    feature_means: Mapping[str, float],
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {', '.join(missing)}")

    required_means = [column for column in feature_columns if column not in feature_means]
    if required_means:
        raise KeyError(f"Missing feature means: {', '.join(required_means)}")

    transformed = frame.copy()
    original_order = transformed.index
    transformed = transformed.sort_values(["stay_id", "hour_before"], kind="mergesort")

    for column in feature_columns:
        transformed[column] = pd.to_numeric(transformed[column], errors="coerce")
        transformed[column] = (
            transformed.groupby("stay_id", group_keys=False)[column]
            .apply(lambda series: series.interpolate(method="linear", limit_area="inside"))
            .astype(float)
        )
        transformed[column] = transformed[column].fillna(float(feature_means[column]))

    return transformed.loc[original_order].reset_index(drop=True)


def run_preprocessing_dry_run(
    csv_path: Path,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> PreprocessingDryRunSummary:
    frame = pd.read_csv(csv_path)
    deduplicated = deduplicate_exact_stay_hours(frame)
    split = split_stay_ids(deduplicated, test_size=test_size, random_state=random_state)
    train = deduplicated[deduplicated["stay_id"].isin(split.train_stay_ids)]

    means = fit_feature_means(train, feature_columns=feature_columns)
    missing_before = int(deduplicated[list(feature_columns)].isna().sum().sum())
    transformed = transform_with_professor_imputation(
        deduplicated,
        feature_means=means,
        feature_columns=feature_columns,
    )
    missing_after = int(transformed[list(feature_columns)].isna().sum().sum())
    outcomes = derive_stay_outcomes(transformed)
    stay_positive_count = int(outcomes["stay_has_sepsis"].sum())
    stay_negative_count = int(len(outcomes) - stay_positive_count)

    return PreprocessingDryRunSummary(
        row_count=len(transformed),
        stay_count=int(transformed["stay_id"].nunique()),
        train_stay_count=len(split.train_stay_ids),
        test_stay_count=len(split.test_stay_ids),
        feature_count=len(feature_columns),
        missing_before=missing_before,
        missing_after=missing_after,
        stay_positive_count=stay_positive_count,
        stay_negative_count=stay_negative_count,
    )


def format_dry_run_summary(summary: PreprocessingDryRunSummary) -> str:
    return "\n".join(
        [
            "Preprocessing dry run",
            "=====================",
            f"Rows after deduplication: {summary.row_count}",
            f"Unique stays: {summary.stay_count}",
            f"Train stays: {summary.train_stay_count}",
            f"Test stays: {summary.test_stay_count}",
            f"Feature columns imputed: {summary.feature_count}",
            f"Missing feature values before: {summary.missing_before}",
            f"Missing feature values after: {summary.missing_after}",
            (
                "Stay-level outcomes: "
                f"positive={summary.stay_positive_count}, "
                f"negative={summary.stay_negative_count}"
            ),
            "No output files were written.",
        ]
    )
