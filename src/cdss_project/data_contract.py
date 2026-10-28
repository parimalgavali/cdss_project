from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from cdss_project.audit import (
    derive_stay_outcomes,
    duplicate_summary,
    structural_empty_rows,
)
from cdss_project.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, REQUIRED_COLUMNS

DATASET_PATH = Path("data/mimic_sepsis_onset_prediction.csv")


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    message: str


@dataclass(frozen=True)
class DataContractSummary:
    row_count: int
    column_count: int
    stay_count: int | None = None
    hour_before_min: int | None = None
    hour_before_max: int | None = None
    sepsis_counts: dict[Any, int] = field(default_factory=dict)
    sepsis_nan_count: int = 0
    duplicate_stay_hour_row_count: int = 0
    duplicate_stay_hour_group_count: int = 0
    structural_empty_row_count: int = 0
    stay_positive_count: int = 0
    stay_negative_count: int = 0
    stay_rows_summary: dict[str, float] = field(default_factory=dict)
    structural_empty_by_hour: dict[int, int] = field(default_factory=dict)
    categorical_value_types: dict[str, str] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)


def validate_csv_path(path: Path = DATASET_PATH) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the CSV there before running validation."
        )
    if not path.is_file():
        raise FileNotFoundError(f"Dataset path is not a file: {path}")
    return path


def inspect_csv(path: Path = DATASET_PATH) -> DataContractSummary:
    csv_path = validate_csv_path(path)
    return inspect_dataframe(pd.read_csv(csv_path))


def inspect_dataframe(frame: pd.DataFrame) -> DataContractSummary:
    issues: list[ValidationIssue] = []
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        issues.append(
            ValidationIssue(
                severity="error",
                message=f"Missing required columns: {', '.join(missing_columns)}",
            )
        )

    stay_count = _nunique_int(frame, "stay_id")
    hour_min = _series_min_int(frame, "hour_before")
    hour_max = _series_max_int(frame, "hour_before")
    sepsis_counts: dict[Any, int] = {}
    sepsis_nan_count = 0
    stay_rows_summary: dict[str, float] = {}
    duplicate_stay_hour_row_count = 0
    duplicate_stay_hour_group_count = 0
    structural_empty_row_count = 0
    structural_empty_by_hour: dict[int, int] = {}
    categorical_value_types = {
        column: _value_type(frame[column]) for column in CATEGORICAL_COLUMNS if column in frame.columns
    }
    stay_positive_count = 0
    stay_negative_count = 0

    if {"stay_id", "hour_before"}.issubset(frame.columns):
        duplicates = duplicate_summary(frame)
        duplicate_stay_hour_row_count = duplicates.duplicate_row_count
        duplicate_stay_hour_group_count = duplicates.duplicate_group_count
        if duplicate_stay_hour_row_count:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=(
                        f"{duplicate_stay_hour_row_count} rows belong to "
                        f"{duplicate_stay_hour_group_count} duplicate stay-hour groups"
                    ),
                )
            )

    if "hour_before" in frame.columns:
        invalid_hours = frame["hour_before"].dropna()
        invalid_hours = invalid_hours[(invalid_hours < 1) | (invalid_hours > 30)]
        if not invalid_hours.empty:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=(
                        f"{len(invalid_hours)} rows have hour_before outside the "
                        "expected 1..30 range"
                    ),
                )
            )

    if "stay_id" in frame.columns:
        rows_per_stay = frame.groupby("stay_id", dropna=False).size()
        stay_rows_summary = {
            "min": float(rows_per_stay.min()),
            "median": float(rows_per_stay.median()),
            "max": float(rows_per_stay.max()),
        }
        off_contract_stays = int((rows_per_stay != 30).sum())
        if off_contract_stays:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=f"{off_contract_stays} stays do not have exactly 30 rows",
                )
            )

    if "sepsis" in frame.columns:
        sepsis_nan_count = int(frame["sepsis"].isna().sum())
        raw_counts = frame["sepsis"].value_counts(dropna=True).sort_index()
        sepsis_counts = {key: int(value) for key, value in raw_counts.items()}

        invalid_labels = frame["sepsis"].dropna()
        invalid_labels = invalid_labels[~invalid_labels.isin([0, 1, 0.0, 1.0])]
        if not invalid_labels.empty:
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"{len(invalid_labels)} rows have sepsis labels outside 0/1",
                )
            )
        if sepsis_nan_count:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=(
                        f"{sepsis_nan_count} rows have NaN sepsis labels; current policy "
                        "treats rows with no clinical feature values as structural padding"
                    ),
                )
            )
        empty_mask = structural_empty_rows(frame, feature_columns=FEATURE_COLUMNS)
        structural_empty_row_count = int(empty_mask.sum())
        if "hour_before" in frame.columns:
            by_hour = (
                empty_mask.groupby(frame["hour_before"]).sum().sort_index(ascending=False)
            )
            structural_empty_by_hour = {int(hour): int(count) for hour, count in by_hour.items()}
        try:
            stay_outcomes = derive_stay_outcomes(frame)
            stay_positive_count = int(stay_outcomes["stay_has_sepsis"].sum())
            stay_negative_count = int(len(stay_outcomes) - stay_positive_count)
        except ValueError as error:
            issues.append(ValidationIssue(severity="error", message=str(error)))

    return DataContractSummary(
        row_count=len(frame),
        column_count=len(frame.columns),
        stay_count=stay_count,
        hour_before_min=hour_min,
        hour_before_max=hour_max,
        sepsis_counts=sepsis_counts,
        sepsis_nan_count=sepsis_nan_count,
        duplicate_stay_hour_row_count=duplicate_stay_hour_row_count,
        duplicate_stay_hour_group_count=duplicate_stay_hour_group_count,
        structural_empty_row_count=structural_empty_row_count,
        stay_positive_count=stay_positive_count,
        stay_negative_count=stay_negative_count,
        stay_rows_summary=stay_rows_summary,
        structural_empty_by_hour=structural_empty_by_hour,
        categorical_value_types=categorical_value_types,
        issues=issues,
    )


def format_summary(summary: DataContractSummary) -> str:
    lines = [
        "Dataset inspection",
        "==================",
        f"Rows: {summary.row_count}",
        f"Columns: {summary.column_count}",
    ]

    if summary.stay_count is not None:
        lines.append(f"Unique stays: {summary.stay_count}")
    if summary.hour_before_min is not None and summary.hour_before_max is not None:
        lines.append(
            f"hour_before range: {summary.hour_before_min}..{summary.hour_before_max}"
        )
    if summary.stay_rows_summary:
        rows = summary.stay_rows_summary
        lines.append(
            "Rows per stay: "
            f"min={rows['min']:.0f}, median={rows['median']:.0f}, max={rows['max']:.0f}"
        )
    if summary.sepsis_counts:
        counts = ", ".join(
            f"{label}={count}" for label, count in summary.sepsis_counts.items()
        )
        lines.append(f"Sepsis label counts: {counts}")
    lines.append(f"NaN sepsis labels: {summary.sepsis_nan_count}")
    lines.append(f"Structural empty rows: {summary.structural_empty_row_count}")
    lines.append(
        "Duplicate stay-hour rows: "
        f"{summary.duplicate_stay_hour_row_count} "
        f"across {summary.duplicate_stay_hour_group_count} groups"
    )
    if summary.structural_empty_by_hour:
        lines.append("Structural empty rows by hour_before (30 = first ICU hour):")
        for hour, count in summary.structural_empty_by_hour.items():
            lines.append(f"  hour_before={hour:>2}: {count}")
    if summary.categorical_value_types:
        types = ", ".join(
            f"{column}={kind}" for column, kind in summary.categorical_value_types.items()
        )
        lines.append(f"Categorical value types: {types}")
    if summary.stay_positive_count or summary.stay_negative_count:
        lines.append(
            "Stay-level outcomes: "
            f"positive={summary.stay_positive_count}, negative={summary.stay_negative_count}"
        )

    if summary.issues:
        lines.extend(["", "Issues:"])
        for issue in summary.issues:
            lines.append(f"- [{issue.severity.upper()}] {issue.message}")
    else:
        lines.extend(["", "Issues: none"])

    return "\n".join(lines)


def _nunique_int(frame: pd.DataFrame, column: str) -> int | None:
    if column not in frame.columns:
        return None
    return int(frame[column].nunique(dropna=True))


def _series_min_int(frame: pd.DataFrame, column: str) -> int | None:
    if column not in frame.columns or frame[column].dropna().empty:
        return None
    return int(frame[column].min())


def _series_max_int(frame: pd.DataFrame, column: str) -> int | None:
    if column not in frame.columns or frame[column].dropna().empty:
        return None
    return int(frame[column].max())


def _value_type(series: pd.Series) -> str:
    values = series.dropna()
    if values.empty:
        return "empty"
    distinct = int(values.nunique())
    kind = "numeric" if pd.api.types.is_numeric_dtype(values) else "text"
    return f"{kind} ({distinct} distinct)"
