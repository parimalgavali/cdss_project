from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cdss_project.audit import (
    deduplicate_exact_stay_hours,
    derive_stay_outcomes,
    structural_empty_rows,
)
from cdss_project.features import CATEGORICAL_COLUMNS, INTERPOLATED_FEATURE_COLUMNS


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
    n_folds: int
    test_fold: int
    numeric_feature_count: int
    categorical_feature_count: int
    missing_before: int
    missing_after: int
    padding_row_count: int
    stay_positive_count: int
    stay_negative_count: int
    train_positive_count: int
    test_positive_count: int


@dataclass(frozen=True)
class PreprocessingWriteSummary(PreprocessingDryRunSummary):
    output_path: Path


def assign_stay_folds(
    frame: pd.DataFrame,
    *,
    n_folds: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Assign each stay to a fold, stratified by the observed stay-level outcome."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")

    outcomes = derive_stay_outcomes(frame)
    if len(outcomes) < n_folds:
        raise ValueError(f"Cannot split {len(outcomes)} stays into {n_folds} folds")

    rng = np.random.default_rng(random_state)
    assignments: list[pd.DataFrame] = []
    offset = 0
    for label in sorted(outcomes["stay_has_sepsis"].unique()):
        stay_ids = outcomes.loc[outcomes["stay_has_sepsis"] == label, "stay_id"].to_numpy()
        shuffled = stay_ids.copy()
        rng.shuffle(shuffled)
        folds = (np.arange(len(shuffled)) + offset) % n_folds
        offset = (offset + len(shuffled)) % n_folds
        assignments.append(pd.DataFrame({"stay_id": shuffled, "fold": folds}))

    folds_frame = pd.concat(assignments, ignore_index=True)
    folds_frame["fold"] = folds_frame["fold"].astype(int)
    return folds_frame.sort_values("stay_id").reset_index(drop=True)


def split_stay_ids(
    frame: pd.DataFrame,
    *,
    n_folds: int = 5,
    test_fold: int = 0,
    random_state: int = 42,
) -> StaySplit:
    if not 0 <= test_fold < n_folds:
        raise ValueError(f"test_fold must be between 0 and {n_folds - 1}")
    folds = assign_stay_folds(frame, n_folds=n_folds, random_state=random_state)
    return _split_from_folds(folds, test_fold=test_fold)


def fit_feature_means(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = INTERPOLATED_FEATURE_COLUMNS,
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


def fit_categorical_modes(
    frame: pd.DataFrame,
    *,
    categorical_columns: Sequence[str] = CATEGORICAL_COLUMNS,
) -> dict[str, Any]:
    missing = [column for column in categorical_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing categorical columns: {', '.join(missing)}")

    modes: dict[str, Any] = {}
    for column in categorical_columns:
        mode = frame[column].mode(dropna=True)
        if mode.empty:
            raise ValueError(f"Cannot compute mode for entirely missing column: {column}")
        modes[column] = mode.sort_values().iloc[0]
    return modes


def transform_with_professor_imputation(
    frame: pd.DataFrame,
    *,
    feature_means: Mapping[str, float],
    feature_columns: Sequence[str] = INTERPOLATED_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Interpolate numeric features within each stay, then fill gaps with train means."""
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


def fill_categorical_columns(
    frame: pd.DataFrame,
    *,
    categorical_modes: Mapping[str, Any],
    categorical_columns: Sequence[str] = CATEGORICAL_COLUMNS,
) -> pd.DataFrame:
    """Fill categoricals from the same stay (ffill/bfill), then with the train-set mode."""
    missing = [column for column in categorical_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing categorical columns: {', '.join(missing)}")

    required_modes = [column for column in categorical_columns if column not in categorical_modes]
    if required_modes:
        raise KeyError(f"Missing categorical modes: {', '.join(required_modes)}")

    transformed = frame.copy()
    original_order = transformed.index
    transformed = transformed.sort_values(["stay_id", "hour_before"], kind="mergesort")

    for column in categorical_columns:
        grouped = transformed.groupby("stay_id", group_keys=False)[column]
        filled = grouped.ffill()
        filled = filled.fillna(filled.groupby(transformed["stay_id"]).bfill())
        transformed[column] = filled.fillna(categorical_modes[column])

    return transformed.loc[original_order].reset_index(drop=True)


def preprocess_to_dataframe(
    frame: pd.DataFrame,
    *,
    feature_means: Mapping[str, float],
    feature_columns: Sequence[str] = INTERPOLATED_FEATURE_COLUMNS,
    categorical_modes: Mapping[str, Any] | None = None,
    categorical_columns: Sequence[str] = CATEGORICAL_COLUMNS,
    folds: pd.DataFrame | None = None,
    test_fold: int = 0,
) -> pd.DataFrame:
    deduplicated = deduplicate_exact_stay_hours(frame)
    present_categoricals = [column for column in categorical_columns if column in deduplicated.columns]

    padding = structural_empty_rows(
        deduplicated,
        feature_columns=[*feature_columns, *present_categoricals],
    )
    transformed = transform_with_professor_imputation(
        deduplicated,
        feature_means=feature_means,
        feature_columns=feature_columns,
    )
    if present_categoricals:
        if categorical_modes is None:
            raise KeyError("categorical_modes are required when categorical columns are present")
        transformed = fill_categorical_columns(
            transformed,
            categorical_modes=categorical_modes,
            categorical_columns=present_categoricals,
        )
    transformed["is_padding"] = padding.to_numpy().astype(int)

    outcomes = derive_stay_outcomes(transformed)
    processed = transformed.merge(outcomes, on="stay_id", how="left", validate="many_to_one")
    if folds is not None:
        processed = processed.merge(
            folds[["stay_id", "fold"]], on="stay_id", how="left", validate="many_to_one"
        )
        if processed["fold"].isna().any():
            raise ValueError("Some stays have no fold assignment")
        processed["fold"] = processed["fold"].astype(int)
        processed["split"] = np.where(processed["fold"] == test_fold, "test", "train")
    return processed


def run_preprocessing_dry_run(
    csv_path: Path,
    *,
    n_folds: int = 5,
    test_fold: int = 0,
    random_state: int = 42,
    feature_columns: Sequence[str] = INTERPOLATED_FEATURE_COLUMNS,
    categorical_columns: Sequence[str] = CATEGORICAL_COLUMNS,
) -> PreprocessingDryRunSummary:
    _, summary = _run_pipeline(
        pd.read_csv(csv_path),
        n_folds=n_folds,
        test_fold=test_fold,
        random_state=random_state,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
    )
    return summary


def write_preprocessed_csv(
    csv_path: Path,
    output_path: Path,
    *,
    n_folds: int = 5,
    test_fold: int = 0,
    random_state: int = 42,
    feature_columns: Sequence[str] = INTERPOLATED_FEATURE_COLUMNS,
    categorical_columns: Sequence[str] = CATEGORICAL_COLUMNS,
    overwrite: bool = False,
) -> PreprocessingWriteSummary:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use overwrite=True to replace it."
        )

    processed, summary = _run_pipeline(
        pd.read_csv(csv_path),
        n_folds=n_folds,
        test_fold=test_fold,
        random_state=random_state,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    return PreprocessingWriteSummary(**vars(summary), output_path=output_path)


def format_dry_run_summary(summary: PreprocessingDryRunSummary) -> str:
    lines = _summary_lines(summary, title="Preprocessing dry run")
    lines.append("No output files were written.")
    return "\n".join(lines)


def format_write_summary(summary: PreprocessingWriteSummary) -> str:
    lines = _summary_lines(summary, title="Preprocessing output")
    lines.append(f"Output file: {summary.output_path}")
    return "\n".join(lines)


def _split_from_folds(folds: pd.DataFrame, *, test_fold: int) -> StaySplit:
    is_test = folds["fold"] == test_fold
    return StaySplit(
        train_stay_ids=tuple(sorted(int(value) for value in folds.loc[~is_test, "stay_id"])),
        test_stay_ids=tuple(sorted(int(value) for value in folds.loc[is_test, "stay_id"])),
    )


def _run_pipeline(
    frame: pd.DataFrame,
    *,
    n_folds: int,
    test_fold: int,
    random_state: int,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> tuple[pd.DataFrame, PreprocessingDryRunSummary]:
    if not 0 <= test_fold < n_folds:
        raise ValueError(f"test_fold must be between 0 and {n_folds - 1}")

    deduplicated = deduplicate_exact_stay_hours(frame)
    present_categoricals = [column for column in categorical_columns if column in deduplicated.columns]
    folds = assign_stay_folds(deduplicated, n_folds=n_folds, random_state=random_state)
    split = _split_from_folds(folds, test_fold=test_fold)
    train = deduplicated[deduplicated["stay_id"].isin(split.train_stay_ids)]

    means = fit_feature_means(train, feature_columns=feature_columns)
    modes = (
        fit_categorical_modes(train, categorical_columns=present_categoricals)
        if present_categoricals
        else None
    )
    all_features = [*feature_columns, *present_categoricals]
    missing_before = int(deduplicated[all_features].isna().sum().sum())
    processed = preprocess_to_dataframe(
        deduplicated,
        feature_means=means,
        feature_columns=feature_columns,
        categorical_modes=modes,
        categorical_columns=present_categoricals,
        folds=folds,
        test_fold=test_fold,
    )
    missing_after = int(processed[all_features].isna().sum().sum())

    stays = processed.drop_duplicates("stay_id")
    stay_positive_count = int(stays["stay_has_sepsis"].sum())
    summary = PreprocessingDryRunSummary(
        row_count=len(processed),
        stay_count=len(stays),
        train_stay_count=len(split.train_stay_ids),
        test_stay_count=len(split.test_stay_ids),
        n_folds=n_folds,
        test_fold=test_fold,
        numeric_feature_count=len(feature_columns),
        categorical_feature_count=len(present_categoricals),
        missing_before=missing_before,
        missing_after=missing_after,
        padding_row_count=int(processed["is_padding"].sum()),
        stay_positive_count=stay_positive_count,
        stay_negative_count=len(stays) - stay_positive_count,
        train_positive_count=int(stays.loc[stays["split"] == "train", "stay_has_sepsis"].sum()),
        test_positive_count=int(stays.loc[stays["split"] == "test", "stay_has_sepsis"].sum()),
    )
    return processed, summary


def _summary_lines(summary: PreprocessingDryRunSummary, *, title: str) -> list[str]:
    return [
        title,
        "=" * len(title),
        f"Rows after deduplication: {summary.row_count}",
        f"Unique stays: {summary.stay_count}",
        f"Folds: {summary.n_folds} (stratified by stay_has_sepsis), test fold: {summary.test_fold}",
        f"Train stays: {summary.train_stay_count} (positive={summary.train_positive_count})",
        f"Test stays: {summary.test_stay_count} (positive={summary.test_positive_count})",
        f"Numeric feature columns imputed: {summary.numeric_feature_count}",
        f"Categorical feature columns filled: {summary.categorical_feature_count}",
        f"Missing feature values before: {summary.missing_before}",
        f"Missing feature values after: {summary.missing_after}",
        f"Padding rows flagged (is_padding=1): {summary.padding_row_count}",
        (
            "Stay-level outcomes: "
            f"positive={summary.stay_positive_count}, "
            f"negative={summary.stay_negative_count}"
        ),
    ]
