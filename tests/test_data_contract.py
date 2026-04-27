from pathlib import Path

import pandas as pd
import pytest

from cdss_project.data_contract import (
    REQUIRED_COLUMNS,
    ValidationIssue,
    inspect_dataframe,
    validate_csv_path,
)


def make_valid_frame() -> pd.DataFrame:
    rows = []
    for stay_id in [100, 200]:
        for hour in range(1, 31):
            rows.append(
                {
                    "stay_id": stay_id,
                    "hour_before": hour,
                    "gender": "F",
                    "ethnicity": "UNKNOWN",
                    "admission_age": 70.0,
                    "height": 165.0,
                    "weight": 72.0,
                    "diabetes": 0,
                    "platelet": 200.0,
                    "leukocytes": 8.0,
                    "po2": 95.0,
                    "fio2": 21.0,
                    "lactate": 1.2,
                    "creatinine": 1.0,
                    "bilirubin_total": 0.7,
                    "gcs": 15.0,
                    "crp": 3.0,
                    "sofa_24hours": 2.0,
                    "dbp": 70.0,
                    "glucose": 110.0,
                    "heart_rate": 85.0,
                    "mbp": 85.0,
                    "resp_rate": 18.0,
                    "sbp": 120.0,
                    "temperature": 37.0,
                    "spo2": 98.0,
                    "urineoutput": 50.0,
                    "ventilation_status": 0,
                    "last_careunit": "MICU",
                    "sepsis": 0.0,
                }
            )
    frame = pd.DataFrame(rows)
    assert set(REQUIRED_COLUMNS).issubset(frame.columns)
    return frame


def test_inspect_dataframe_reports_expected_contract_for_valid_data():
    summary = inspect_dataframe(make_valid_frame())

    assert summary.row_count == 60
    assert summary.stay_count == 2
    assert summary.hour_before_min == 1
    assert summary.hour_before_max == 30
    assert summary.sepsis_counts[0.0] == 60
    assert summary.issues == []


def test_inspect_dataframe_reports_missing_required_columns():
    frame = make_valid_frame().drop(columns=["hour_before", "sepsis"])

    summary = inspect_dataframe(frame)

    assert ValidationIssue(
        severity="error",
        message="Missing required columns: hour_before, sepsis",
    ) in summary.issues


def test_inspect_dataframe_warns_about_nan_sepsis_labels():
    frame = make_valid_frame()
    frame.loc[:4, "sepsis"] = pd.NA

    summary = inspect_dataframe(frame)

    assert any(
        issue.severity == "warning" and "5 rows have NaN sepsis labels" in issue.message
        for issue in summary.issues
    )
    assert summary.sepsis_nan_count == 5


def test_inspect_dataframe_warns_when_stays_do_not_have_30_hourly_rows():
    frame = make_valid_frame().query("not (stay_id == 200 and hour_before == 30)")

    summary = inspect_dataframe(frame)

    assert any(
        issue.severity == "warning"
        and "1 stays do not have exactly 30 rows" in issue.message
        for issue in summary.issues
    )


def test_inspect_dataframe_reports_duplicate_structural_and_stay_label_counts():
    frame = make_valid_frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    frame.loc[0, "sepsis"] = pd.NA
    for column in REQUIRED_COLUMNS:
        if column not in {"stay_id", "hour_before", "sepsis"}:
            frame.loc[0, column] = pd.NA
    frame.loc[30:59, "sepsis"] = 1.0

    summary = inspect_dataframe(frame)

    assert summary.duplicate_stay_hour_row_count == 2
    assert summary.duplicate_stay_hour_group_count == 1
    assert summary.structural_empty_row_count == 1
    assert summary.stay_positive_count == 1
    assert summary.stay_negative_count == 1


def test_validate_csv_path_rejects_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        validate_csv_path(tmp_path / "missing.csv")
