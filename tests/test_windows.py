import numpy as np
import pandas as pd
import pytest

from cdss_project.windows import (
    build_windows,
    format_window_summary,
    load_windows,
    save_windows,
    summarize_windows,
)


def make_hourly_frame(
    stays: dict[int, int], *, hours: int = 30, padding: dict[int, int] | None = None
) -> pd.DataFrame:
    """Rows in reverse-chronological order to check that windowing re-sorts them."""
    padding = padding or {}
    rows = []
    for stay_id, label in stays.items():
        for hour_before in range(1, hours + 1):
            hour_after_admission = hours - hour_before
            rows.append(
                {
                    "stay_id": stay_id,
                    "hour_before": hour_before,
                    "heart_rate": float(stay_id * 100 + hour_after_admission),
                    "gender": 1.0,
                    "stay_has_sepsis": label,
                    "is_padding": int(hour_after_admission < padding.get(stay_id, 0)),
                    "fold": stay_id % 2,
                    "split": "test" if stay_id % 2 == 0 else "train",
                }
            )
    return pd.DataFrame(rows)


def test_build_windows_matches_outline_horizons_and_order():
    dataset = build_windows(
        make_hourly_frame({1: 0, 2: 1}), feature_columns=["heart_rate", "gender"]
    )

    assert dataset.X.shape == (50, 6, 2)
    assert dataset.horizon[:25].tolist() == list(range(25, 0, -1))
    first = dataset.X[0, :, 0]
    last = dataset.X[24, :, 0]
    assert first.tolist() == [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    assert last.tolist() == [124.0, 125.0, 126.0, 127.0, 128.0, 129.0]
    assert dataset.stay_id.tolist() == [1] * 25 + [2] * 25
    assert dataset.y.tolist() == [0] * 25 + [1] * 25
    assert dataset.fold.tolist() == [1] * 25 + [0] * 25
    assert set(dataset.split[25:]) == {"test"}


def test_windows_never_cross_stays():
    dataset = build_windows(make_hourly_frame({1: 0, 2: 1}), feature_columns=["heart_rate"])

    per_window_stay = (dataset.X[:, :, 0] // 100).astype(int)
    assert (per_window_stay == dataset.stay_id[:, None]).all()


def test_observed_hours_counts_non_padding_rows():
    dataset = build_windows(
        make_hourly_frame({1: 0}, padding={1: 8}), feature_columns=["heart_rate"]
    )

    # Hours 0..7 are padding: window k covers hours k..k+5.
    assert dataset.observed_hours[:4].tolist() == [0, 0, 0, 1]
    assert dataset.observed_hours[8:].tolist() == [6] * 17


def test_observed_mask_marks_recorded_hours_per_window(tmp_path):
    dataset = build_windows(
        make_hourly_frame({1: 0}, padding={1: 8}), feature_columns=["heart_rate"]
    )

    assert dataset.observed_mask.shape == (25, 6)
    assert dataset.observed_mask[3].tolist() == [0, 0, 0, 0, 0, 1]
    assert (dataset.observed_mask.sum(axis=1) == dataset.observed_hours).all()
    reloaded = load_windows(save_windows(dataset, tmp_path / "w.npz"))
    assert (reloaded.observed_mask == dataset.observed_mask).all()


def test_min_observed_hours_filters_windows():
    dataset = build_windows(
        make_hourly_frame({1: 0}, padding={1: 8}),
        feature_columns=["heart_rate"],
        min_observed_hours=1,
    )

    assert len(dataset) == 22
    assert dataset.horizon.max() == 22
    assert dataset.observed_hours.min() == 1


def test_min_observed_hours_requires_padding_flag():
    frame = make_hourly_frame({1: 0}).drop(columns="is_padding")

    with pytest.raises(KeyError, match="is_padding"):
        build_windows(frame, feature_columns=["heart_rate"], min_observed_hours=1)


def test_custom_window_size_changes_horizon_range():
    dataset = build_windows(make_hourly_frame({1: 0}), feature_columns=["heart_rate"], window_size=5)

    assert dataset.X.shape == (26, 5, 1)
    assert dataset.horizon.tolist() == list(range(26, 0, -1))


def test_rejects_stays_with_missing_hours():
    frame = make_hourly_frame({1: 0, 2: 1})
    frame = frame[~((frame["stay_id"] == 2) & (frame["hour_before"] == 7))]

    with pytest.raises(ValueError, match="exactly 30 hourly rows: 2=29"):
        build_windows(frame, feature_columns=["heart_rate"])


def test_rejects_duplicate_hours_even_when_count_matches():
    frame = make_hourly_frame({1: 0})
    frame.loc[frame["hour_before"] == 7, "hour_before"] = 8

    with pytest.raises(ValueError, match="hour_before 30..1"):
        build_windows(frame, feature_columns=["heart_rate"])


def test_rejects_unimputed_or_text_features():
    frame = make_hourly_frame({1: 0})
    frame.loc[0, "heart_rate"] = np.nan
    with pytest.raises(ValueError, match="run preprocessing first"):
        build_windows(frame, feature_columns=["heart_rate"])

    frame = make_hourly_frame({1: 0})
    frame["gender"] = "F"
    with pytest.raises(TypeError, match="numeric-encoded"):
        build_windows(frame, feature_columns=["gender"])


def test_rejects_inconsistent_stay_label():
    frame = make_hourly_frame({1: 0})
    frame.loc[0, "stay_has_sepsis"] = 1

    with pytest.raises(ValueError, match="constant within each stay"):
        build_windows(frame, feature_columns=["heart_rate"])


def test_summary_reports_per_horizon_counts():
    dataset = build_windows(
        make_hourly_frame({1: 0, 2: 1}, padding={1: 8}), feature_columns=["heart_rate"]
    )

    summary = summarize_windows(dataset)
    table = summary.horizon_table.set_index("horizon")

    assert summary.window_count == 50
    assert summary.positive_window_count == 25
    assert table.loc[25, "windows"] == 2
    assert table.loc[25, "positives"] == 1
    assert table.loc[25, "fully_padded_windows"] == 1
    assert table.loc[25, "mean_observed_hours"] == 3.0
    assert "Per horizon" in format_window_summary(summary)


def test_save_and_load_round_trip(tmp_path):
    dataset = build_windows(make_hourly_frame({1: 0, 2: 1}), feature_columns=["heart_rate", "gender"])
    path = tmp_path / "windows.npz"

    save_windows(dataset, path)
    loaded = load_windows(path)

    assert np.array_equal(loaded.X, dataset.X)
    assert np.array_equal(loaded.horizon, dataset.horizon)
    assert loaded.feature_columns == ("heart_rate", "gender")
    assert loaded.split.tolist() == dataset.split.tolist()
    with pytest.raises(FileExistsError):
        save_windows(dataset, path)
