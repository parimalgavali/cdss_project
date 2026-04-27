import pandas as pd
import pytest

from cdss_project.audit import (
    ConflictingDuplicateRowsError,
    deduplicate_exact_stay_hours,
    derive_stay_outcomes,
    structural_empty_rows,
)


def test_structural_empty_rows_require_nan_label_and_no_clinical_features():
    frame = pd.DataFrame(
        {
            "stay_id": [1, 1, 2],
            "hour_before": [1, 2, 1],
            "heart_rate": [pd.NA, 90.0, pd.NA],
            "lactate": [pd.NA, pd.NA, pd.NA],
            "sepsis": [pd.NA, pd.NA, 0.0],
        }
    )

    result = structural_empty_rows(frame, feature_columns=["heart_rate", "lactate"])

    assert result.tolist() == [True, False, False]


def test_derive_stay_outcomes_uses_observed_labels_only():
    frame = pd.DataFrame(
        {
            "stay_id": [10, 10, 10, 20, 20],
            "hour_before": [1, 2, 3, 1, 2],
            "sepsis": [pd.NA, 1.0, pd.NA, 0.0, pd.NA],
        }
    )

    outcomes = derive_stay_outcomes(frame)

    assert outcomes.set_index("stay_id").loc[10, "stay_has_sepsis"] == 1
    assert outcomes.set_index("stay_id").loc[20, "stay_has_sepsis"] == 0
    assert outcomes.set_index("stay_id").loc[10, "observed_label_count"] == 1
    assert outcomes.set_index("stay_id").loc[20, "observed_label_count"] == 1


def test_derive_stay_outcomes_rejects_stays_without_observed_labels():
    frame = pd.DataFrame(
        {
            "stay_id": [10, 10],
            "hour_before": [1, 2],
            "sepsis": [pd.NA, pd.NA],
        }
    )

    with pytest.raises(ValueError, match="no observed sepsis labels"):
        derive_stay_outcomes(frame)


def test_deduplicate_exact_stay_hours_drops_identical_duplicate_rows():
    frame = pd.DataFrame(
        {
            "stay_id": [10, 10, 10],
            "hour_before": [1, 1, 2],
            "heart_rate": [80.0, 80.0, 81.0],
            "sepsis": [0.0, 0.0, 0.0],
        }
    )

    deduped = deduplicate_exact_stay_hours(frame)

    assert len(deduped) == 2
    assert not deduped.duplicated(["stay_id", "hour_before"]).any()
    assert deduped.iloc[0]["heart_rate"] == 80.0


def test_deduplicate_exact_stay_hours_rejects_conflicting_duplicate_rows():
    frame = pd.DataFrame(
        {
            "stay_id": [10, 10],
            "hour_before": [1, 1],
            "heart_rate": [80.0, 95.0],
            "sepsis": [0.0, 0.0],
        }
    )

    with pytest.raises(ConflictingDuplicateRowsError, match="10"):
        deduplicate_exact_stay_hours(frame)
