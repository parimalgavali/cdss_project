import numpy as np
import pandas as pd
import pytest

from cdss_project.windowing import (
    build_expanding_windows,
    build_fixed_windows,
    summarize_window_plan,
    write_window_dataset,
)


def _thirty_hour_frame(stay_id: int = 10, *, label: int = 1) -> pd.DataFrame:
    hours = list(range(30, 0, -1))
    return pd.DataFrame(
        {
            "stay_id": [stay_id] * 30,
            "hour_before": hours,
            "heart_rate": [float(hour) for hour in hours],
            "sbp": [float(hour * 10) for hour in hours],
            "stay_has_sepsis": [label] * 30,
        }
    )


def test_fixed_windows_create_expected_count_and_shape_for_six_hours():
    dataset = build_fixed_windows(
        _thirty_hour_frame(),
        window_length=6,
        feature_columns=["heart_rate", "sbp"],
    )

    assert dataset.X.shape == (25, 6, 2)
    assert dataset.X.dtype == np.float32
    assert dataset.y.tolist() == [1] * 25
    assert dataset.metadata["window_length"].tolist() == [6] * 25


def test_fixed_windows_preserve_chronological_hour_order_and_metadata():
    dataset = build_fixed_windows(
        _thirty_hour_frame(),
        window_length=6,
        feature_columns=["heart_rate"],
    )

    assert dataset.X[0, :, 0].tolist() == [30.0, 29.0, 28.0, 27.0, 26.0, 25.0]
    first = dataset.metadata.iloc[0]
    assert first["stay_id"] == 10
    assert first["window_start_hour_before"] == 30
    assert first["window_end_hour_before"] == 25
    assert first["prediction_hour_before"] == 25
    assert first["window_type"] == "fixed"


def test_fixed_windows_create_expected_count_for_three_hours():
    dataset = build_fixed_windows(
        _thirty_hour_frame(label=0),
        window_length=3,
        feature_columns=["heart_rate"],
    )

    assert dataset.X.shape == (28, 3, 1)
    assert dataset.y.tolist() == [0] * 28
    assert dataset.metadata["prediction_hour_before"].tolist() == list(range(28, 0, -1))


def test_fixed_windows_skip_stays_shorter_than_window_length():
    frame = _thirty_hour_frame().head(2)

    dataset = build_fixed_windows(frame, window_length=3, feature_columns=["heart_rate"])

    assert dataset.X.shape == (0, 3, 1)
    assert dataset.y.shape == (0,)
    assert dataset.metadata.empty


def test_fixed_partial_windows_predict_from_first_hour_with_padding_and_mask():
    dataset = build_fixed_windows(
        _thirty_hour_frame().head(4),
        window_length=3,
        feature_columns=["heart_rate"],
        include_partial=True,
    )

    assert dataset.X.shape == (4, 3, 1)
    assert dataset.mask is not None
    assert dataset.X[0, :, 0].tolist() == [30.0, 0.0, 0.0]
    assert dataset.mask[0].tolist() == [True, False, False]
    assert dataset.X[1, :, 0].tolist() == [30.0, 29.0, 0.0]
    assert dataset.mask[1].tolist() == [True, True, False]
    assert dataset.X[2, :, 0].tolist() == [30.0, 29.0, 28.0]
    assert dataset.mask[2].tolist() == [True, True, True]
    assert dataset.X[3, :, 0].tolist() == [29.0, 28.0, 27.0]
    assert dataset.metadata["prediction_hour_before"].tolist() == [30, 29, 28, 27]
    assert dataset.metadata["observed_window_length"].tolist() == [1, 2, 3, 3]


def test_expanding_windows_grow_from_minimum_length_and_use_mask():
    dataset = build_expanding_windows(
        _thirty_hour_frame(),
        min_window_length=3,
        feature_columns=["heart_rate"],
    )

    assert dataset.X.shape == (28, 30, 1)
    assert dataset.mask is not None
    assert dataset.metadata["window_length"].tolist() == list(range(3, 31))
    assert dataset.X[0, :3, 0].tolist() == [30.0, 29.0, 28.0]
    assert dataset.X[0, 3:, 0].tolist() == [0.0] * 27
    assert dataset.mask[0].tolist() == [True, True, True] + [False] * 27
    assert dataset.X[-1, :, 0].tolist() == [float(hour) for hour in range(30, 0, -1)]
    assert dataset.mask[-1].all()


def test_windows_reject_missing_label_column():
    frame = _thirty_hour_frame().drop(columns=["stay_has_sepsis"])

    with pytest.raises(KeyError, match="stay_has_sepsis"):
        build_fixed_windows(frame, window_length=6, feature_columns=["heart_rate"])


def test_write_window_dataset_saves_arrays_and_metadata(tmp_path):
    output_path = tmp_path / "windows.npz"
    dataset = build_fixed_windows(
        _thirty_hour_frame(),
        window_length=6,
        feature_columns=["heart_rate"],
    )

    write_window_dataset(dataset, output_path)

    written = np.load(output_path, allow_pickle=True)
    assert written["X"].shape == (25, 6, 1)
    assert written["y"].tolist() == [1] * 25
    assert written["feature_columns"].tolist() == ["heart_rate"]
    assert "metadata_json" in written.files


def test_summarize_window_plan_counts_windows_without_building_arrays():
    frame = pd.concat(
        [
            _thirty_hour_frame(10, label=1),
            _thirty_hour_frame(20, label=0).head(5),
        ],
        ignore_index=True,
    )

    summary = summarize_window_plan(
        frame,
        window_type="fixed",
        window_length=6,
        feature_columns=["heart_rate"],
    )

    assert summary.window_count == 25
    assert summary.positive_window_count == 25
    assert summary.negative_window_count == 0


def test_summarize_window_plan_counts_partial_fixed_windows_per_hour():
    summary = summarize_window_plan(
        _thirty_hour_frame(),
        window_type="fixed",
        window_length=6,
        feature_columns=["heart_rate"],
        include_partial=True,
    )

    assert summary.window_count == 30
    assert summary.positive_window_count == 30
