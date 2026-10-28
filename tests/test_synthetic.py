import numpy as np
import pytest

from cdss_project.audit import structural_empty_rows
from cdss_project.data_contract import inspect_dataframe
from cdss_project.features import REQUIRED_COLUMNS
from cdss_project.synthetic import HOURS_PER_STAY, make_synthetic_frame


def test_synthetic_frame_follows_data_contract():
    frame = make_synthetic_frame(200, seed=1)

    assert set(REQUIRED_COLUMNS) <= set(frame.columns)
    assert len(frame) == 200 * HOURS_PER_STAY
    summary = inspect_dataframe(frame)
    assert not summary.has_errors
    assert summary.hour_before_min == 1
    assert summary.hour_before_max == HOURS_PER_STAY


def test_padding_is_structural_and_leading():
    frame = make_synthetic_frame(100, max_padding_hours=10, seed=2)
    empty = structural_empty_rows(frame)

    assert (empty == frame["sepsis"].isna()).all()
    assert not empty[frame["hour_before"] <= HOURS_PER_STAY - 10].any()
    # Padding is contiguous from hour_before=30 downwards within each stay.
    for _, stay in frame.assign(empty=empty).groupby("stay_id"):
        flags = stay.sort_values("hour_before", ascending=False)["empty"].to_numpy()
        assert (np.diff(flags.astype(int)) <= 0).all()


def test_labels_are_constant_per_stay_and_seeded():
    frame = make_synthetic_frame(300, positive_rate=0.3, seed=3)

    per_stay = frame.dropna(subset=["sepsis"]).groupby("stay_id")["sepsis"].nunique()
    assert (per_stay == 1).all()
    rate = frame.groupby("stay_id")["sepsis"].max().mean()
    assert 0.2 < rate < 0.4
    assert make_synthetic_frame(300, positive_rate=0.3, seed=3).equals(frame)


def test_positive_drift_grows_towards_decision_point():
    frame = make_synthetic_frame(2000, positive_rate=0.5, max_padding_hours=0, seed=4)
    gap = frame.groupby(["hour_before", "sepsis"])["heart_rate"].mean().unstack()
    difference = gap[1.0] - gap[0.0]

    assert difference.loc[1] > 0.7
    assert abs(difference.loc[HOURS_PER_STAY]) < 0.2


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        make_synthetic_frame(0)
    with pytest.raises(ValueError):
        make_synthetic_frame(10, max_padding_hours=HOURS_PER_STAY)
