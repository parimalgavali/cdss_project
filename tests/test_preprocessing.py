import pandas as pd

from cdss_project.preprocessing import (
    fit_feature_means,
    split_stay_ids,
    transform_with_professor_imputation,
)


def test_split_stay_ids_keeps_stays_disjoint_and_deterministic():
    frame = pd.DataFrame({"stay_id": [1, 1, 2, 2, 3, 3, 4, 4]})

    first = split_stay_ids(frame, test_size=0.25, random_state=7)
    second = split_stay_ids(frame, test_size=0.25, random_state=7)

    assert first == second
    assert set(first.train_stay_ids).isdisjoint(first.test_stay_ids)
    assert len(first.test_stay_ids) == 1


def test_fit_feature_means_uses_training_rows_only():
    train = pd.DataFrame({"stay_id": [1, 1], "heart_rate": [80.0, 100.0]})

    means = fit_feature_means(train, feature_columns=["heart_rate"])

    assert means["heart_rate"] == 90.0


def test_transform_interpolates_internal_patient_gaps():
    frame = pd.DataFrame(
        {
            "stay_id": [1, 1, 1],
            "hour_before": [1, 2, 3],
            "heart_rate": [80.0, pd.NA, 100.0],
            "sepsis": [0.0, pd.NA, 0.0],
        }
    )

    transformed = transform_with_professor_imputation(
        frame,
        feature_means={"heart_rate": 70.0},
        feature_columns=["heart_rate"],
    )

    assert transformed["heart_rate"].tolist() == [80.0, 90.0, 100.0]
    assert transformed["sepsis"].isna().sum() == 1


def test_transform_uses_train_mean_for_leading_trailing_and_entirely_missing_values():
    frame = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 2, 2],
            "hour_before": [1, 2, 3, 1, 2],
            "lactate": [pd.NA, 2.0, pd.NA, pd.NA, pd.NA],
            "sepsis": [0.0, 0.0, pd.NA, 1.0, pd.NA],
        }
    )

    transformed = transform_with_professor_imputation(
        frame,
        feature_means={"lactate": 5.0},
        feature_columns=["lactate"],
    )

    assert transformed["lactate"].tolist() == [5.0, 2.0, 5.0, 5.0, 5.0]
    assert transformed["sepsis"].isna().sum() == 2


def test_transform_does_not_fit_new_means_from_validation_data():
    frame = pd.DataFrame(
        {
            "stay_id": [99, 99],
            "hour_before": [1, 2],
            "glucose": [pd.NA, pd.NA],
            "sepsis": [0.0, pd.NA],
        }
    )

    transformed = transform_with_professor_imputation(
        frame,
        feature_means={"glucose": 123.0},
        feature_columns=["glucose"],
    )

    assert transformed["glucose"].tolist() == [123.0, 123.0]
