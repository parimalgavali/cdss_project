import pandas as pd
import pytest

from cdss_project.preprocessing import (
    assign_stay_folds,
    fill_categorical_columns,
    fit_categorical_modes,
    fit_feature_means,
    preprocess_to_dataframe,
    run_preprocessing_dry_run,
    split_stay_ids,
    transform_with_professor_imputation,
    write_preprocessed_csv,
)


def make_labeled_stays(positive: int, negative: int) -> pd.DataFrame:
    rows = []
    for index in range(positive + negative):
        label = 1.0 if index < positive else 0.0
        rows.append({"stay_id": 1000 + index, "hour_before": 1, "sepsis": label})
        rows.append({"stay_id": 1000 + index, "hour_before": 2, "sepsis": pd.NA})
    return pd.DataFrame(rows)


def test_split_stay_ids_keeps_stays_disjoint_and_deterministic():
    frame = make_labeled_stays(positive=4, negative=4)

    first = split_stay_ids(frame, n_folds=4, test_fold=1, random_state=7)
    second = split_stay_ids(frame, n_folds=4, test_fold=1, random_state=7)

    assert first == second
    assert set(first.train_stay_ids).isdisjoint(first.test_stay_ids)
    assert len(first.test_stay_ids) == 2
    assert len(first.train_stay_ids) == 6


def test_split_stay_ids_rejects_out_of_range_test_fold():
    with pytest.raises(ValueError, match="test_fold"):
        split_stay_ids(make_labeled_stays(positive=2, negative=3), n_folds=5, test_fold=5)


def test_assign_stay_folds_is_stratified_by_stay_outcome():
    frame = make_labeled_stays(positive=10, negative=40)

    folds = assign_stay_folds(frame, n_folds=5, random_state=42)
    assert folds.equals(assign_stay_folds(frame, n_folds=5, random_state=42))
    outcomes = frame.dropna(subset=["sepsis"]).set_index("stay_id")["sepsis"]
    folds["label"] = folds["stay_id"].map(outcomes)

    assert folds["stay_id"].is_unique
    assert set(folds["stay_id"]) == set(frame["stay_id"])
    per_fold = folds.groupby("fold")["label"].agg(["size", "sum"])
    assert per_fold["size"].tolist() == [10] * 5
    assert per_fold["sum"].tolist() == [2.0] * 5


def test_assign_stay_folds_rejects_too_few_stays():
    with pytest.raises(ValueError, match="Cannot split"):
        assign_stay_folds(make_labeled_stays(positive=1, negative=1), n_folds=5)


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


def test_preprocess_to_dataframe_preserves_sepsis_and_adds_stay_labels():
    frame = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 2, 2],
            "hour_before": [1, 2, 3, 1, 2],
            "heart_rate": [80.0, pd.NA, 100.0, pd.NA, pd.NA],
            "sepsis": [0.0, pd.NA, 0.0, 1.0, pd.NA],
        }
    )

    processed = preprocess_to_dataframe(
        frame,
        feature_means={"heart_rate": 70.0},
        feature_columns=["heart_rate"],
    )

    assert processed["heart_rate"].tolist() == [80.0, 90.0, 100.0, 70.0, 70.0]
    assert processed["sepsis"].isna().sum() == 2
    assert processed.loc[processed["stay_id"] == 1, "stay_has_sepsis"].unique().tolist() == [0]
    assert processed.loc[processed["stay_id"] == 2, "stay_has_sepsis"].unique().tolist() == [1]


def test_write_preprocessed_csv_creates_output_without_modifying_input(tmp_path):
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "processed" / "preprocessed.csv"
    raw = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 2, 2, 2],
            "hour_before": [1, 2, 3, 1, 2, 3],
            "heart_rate": [80.0, pd.NA, 100.0, 70.0, pd.NA, 90.0],
            "sepsis": [0.0, pd.NA, 0.0, 1.0, pd.NA, 1.0],
        }
    )
    raw.to_csv(input_path, index=False)

    summary = write_preprocessed_csv(
        input_path,
        output_path,
        n_folds=2,
        test_fold=0,
        feature_columns=["heart_rate"],
        overwrite=False,
    )

    assert input_path.read_text() == raw.to_csv(index=False)
    written = pd.read_csv(output_path)
    assert written["heart_rate"].tolist() == [80.0, 90.0, 100.0, 70.0, 80.0, 90.0]
    assert "stay_has_sepsis" in written.columns
    assert written["is_padding"].tolist() == [0, 1, 0, 0, 1, 0]
    assert written.groupby("stay_id")["fold"].nunique().tolist() == [1, 1]
    assert sorted(written.drop_duplicates("stay_id")["split"]) == ["test", "train"]
    assert summary.output_path == output_path
    assert summary.row_count == 6
    assert summary.padding_row_count == 2
    assert summary.train_stay_count == 1
    assert summary.test_stay_count == 1


def test_write_preprocessed_csv_refuses_to_overwrite_existing_output(tmp_path):
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "out.csv"
    pd.DataFrame({"stay_id": [1], "hour_before": [1], "heart_rate": [1.0], "sepsis": [0.0]}).to_csv(
        input_path, index=False
    )
    output_path.write_text("existing")

    with pytest.raises(FileExistsError):
        write_preprocessed_csv(input_path, output_path, feature_columns=["heart_rate"])
    assert output_path.read_text() == "existing"


def test_fill_categorical_columns_uses_stay_values_then_train_mode():
    frame = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 2, 2],
            "hour_before": [1, 2, 3, 1, 2],
            "gender": [pd.NA, "F", pd.NA, pd.NA, pd.NA],
            "last_careunit": [1.0, pd.NA, 3.0, pd.NA, pd.NA],
        }
    )

    filled = fill_categorical_columns(
        frame,
        categorical_modes={"gender": "M", "last_careunit": 7.0},
        categorical_columns=["gender", "last_careunit"],
    )

    assert filled["gender"].tolist() == ["F", "F", "F", "M", "M"]
    assert filled["last_careunit"].tolist() == [1.0, 1.0, 3.0, 7.0, 7.0]


def test_fit_categorical_modes_rejects_entirely_missing_column():
    frame = pd.DataFrame({"gender": [pd.NA, pd.NA]})

    with pytest.raises(ValueError, match="entirely missing"):
        fit_categorical_modes(frame, categorical_columns=["gender"])


def test_preprocess_requires_categorical_modes_when_categoricals_present():
    frame = pd.DataFrame(
        {"stay_id": [1], "hour_before": [1], "heart_rate": [80.0], "gender": ["F"], "sepsis": [0.0]}
    )

    with pytest.raises(KeyError, match="categorical_modes"):
        preprocess_to_dataframe(
            frame,
            feature_means={"heart_rate": 80.0},
            feature_columns=["heart_rate"],
            categorical_columns=["gender"],
        )


def test_padding_flag_is_computed_before_imputation():
    frame = pd.DataFrame(
        {
            "stay_id": [1, 1, 1],
            "hour_before": [1, 2, 3],
            "heart_rate": [80.0, pd.NA, pd.NA],
            "gender": ["F", "F", pd.NA],
            "sepsis": [0.0, 0.0, pd.NA],
        }
    )

    processed = preprocess_to_dataframe(
        frame,
        feature_means={"heart_rate": 70.0},
        feature_columns=["heart_rate"],
        categorical_modes={"gender": "M"},
        categorical_columns=["gender"],
    )

    assert processed["is_padding"].tolist() == [0, 0, 1]
    assert processed["heart_rate"].tolist() == [80.0, 70.0, 70.0]
    assert processed["gender"].tolist() == ["F", "F", "F"]


def test_dry_run_fits_statistics_on_train_folds_only(tmp_path):
    input_path = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "stay_id": [1, 1, 2, 2],
            "hour_before": [1, 2, 1, 2],
            "heart_rate": [pd.NA, pd.NA, 60.0, 80.0],
            "gender": [pd.NA, pd.NA, "F", "F"],
            "sepsis": [0.0, pd.NA, 1.0, 1.0],
        }
    ).to_csv(input_path, index=False)
    output_path = tmp_path / "out.csv"

    summary = write_preprocessed_csv(
        input_path,
        output_path,
        n_folds=2,
        test_fold=0,
        feature_columns=["heart_rate"],
        categorical_columns=["gender"],
    )
    written = pd.read_csv(output_path)
    test_stay = written.loc[written["split"] == "test", "stay_id"].unique().tolist()

    assert test_stay == [1]
    assert written.loc[written["stay_id"] == 1, "heart_rate"].tolist() == [70.0, 70.0]
    assert written.loc[written["stay_id"] == 1, "gender"].tolist() == ["F", "F"]
    assert summary.missing_after == 0
    assert summary.categorical_feature_count == 1
    dry = run_preprocessing_dry_run(
        input_path, n_folds=2, test_fold=0, feature_columns=["heart_rate"], categorical_columns=["gender"]
    )
    assert dry.missing_before == summary.missing_before
