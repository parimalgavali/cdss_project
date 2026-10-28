import math

import numpy as np
import pandas as pd
import pytest

from cdss_project.evaluation import (
    binary_metrics,
    early_detection_advantage,
    early_detection_summary,
    first_alert_horizons,
    metrics_by_horizon,
    select_threshold,
)


def test_binary_metrics_counts():
    y = np.array([1, 1, 0, 0, 0])
    score = np.array([0.9, 0.4, 0.6, 0.2, 0.1])

    metrics = binary_metrics(y, score, threshold=0.5)

    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["auroc"] == pytest.approx(5 / 6)
    assert metrics["windows"] == 5 and metrics["positives"] == 2


def test_binary_metrics_single_class_gives_nan_auc():
    metrics = binary_metrics(np.zeros(3), np.array([0.1, 0.7, 0.2]), threshold=0.5)

    assert math.isnan(metrics["auroc"]) and math.isnan(metrics["sensitivity"])
    assert metrics["specificity"] == pytest.approx(2 / 3)


def test_metrics_by_horizon_has_overall_row_then_descending_horizons():
    predictions = pd.DataFrame(
        {"horizon": [1, 2, 1, 2], "y": [1, 1, 0, 0], "score": [0.9, 0.8, 0.1, 0.3]}
    )

    table = metrics_by_horizon(predictions, threshold=0.5)

    assert table["horizon"].tolist() == ["all", 2, 1]
    assert (table["threshold"] == 0.5).all()
    assert table.loc[0, "auroc"] == 1.0


def test_select_threshold_max_f1_and_target_sensitivity():
    y = np.array([1, 1, 1, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.15, 0.7, 0.2, 0.1])

    assert select_threshold(y, score) == pytest.approx(0.8)
    assert select_threshold(y, score, target_sensitivity=1.0) == pytest.approx(0.15)
    with pytest.raises(ValueError, match="both classes"):
        select_threshold(np.ones(3), score[:3])


def test_first_alert_horizon_is_earliest_window_over_threshold():
    predictions = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 2, 2, 2],
            "horizon": [3, 2, 1, 3, 2, 1],
            "y": [1, 1, 1, 0, 0, 0],
            "score": [0.2, 0.9, 0.4, 0.1, 0.2, 0.3],
        }
    )

    alerts = first_alert_horizons(predictions, threshold=0.5)

    assert alerts.set_index("stay_id")["first_alert_horizon"].to_dict() == {1: 2, 2: 0}
    summary = early_detection_summary(alerts)
    assert summary["stay_detection_rate"] == 1.0
    assert summary["mean_warning_hours"] == 2.0
    assert summary["negative_stays_alerted"] == 0


def test_persistent_alert_needs_consecutive_windows():
    predictions = pd.DataFrame(
        {
            "stay_id": [1] * 6 + [2] * 6 + [3] * 3,
            "horizon": [6, 5, 4, 3, 2, 1] * 2 + [5, 3, 1],
            "y": [1] * 15,
            "score": [
                *[0.9, 0.1, 0.9, 0.9, 0.9, 0.1],  # single spike at 6, then run 4..2
                *[0.9, 0.1, 0.9, 0.1, 0.9, 0.1],  # never two in a row
                *[0.9, 0.9, 0.9],  # horizons not contiguous (windows filtered out)
            ],
        }
    ).sample(frac=1, random_state=0)

    def first(persistence):
        alerts = first_alert_horizons(predictions, threshold=0.5, persistence=persistence)
        return alerts.set_index("stay_id")["first_alert_horizon"].to_dict()

    assert first(1) == {1: 6, 2: 6, 3: 5}
    assert first(2) == {1: 3, 2: 0, 3: 0}
    assert first(3) == {1: 2, 2: 0, 3: 0}
    assert first(4) == {1: 0, 2: 0, 3: 0}
    with pytest.raises(ValueError, match="at least 1"):
        first(0)


def test_early_detection_advantage_is_paired_mean_on_positive_stays():
    model = pd.DataFrame({"stay_id": [1, 2, 3], "y": [1, 1, 0], "first_alert_horizon": [10, 4, 7]})
    reference = pd.DataFrame(
        {"stay_id": [1, 2, 3], "y": [1, 1, 0], "first_alert_horizon": [6, 0, 0]}
    )

    assert early_detection_advantage(model, reference) == 4.0
    with pytest.raises(ValueError, match="same stays"):
        early_detection_advantage(model, reference.iloc[:2])
