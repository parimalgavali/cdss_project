"""Window-level metrics, threshold selection, and early-detection helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

METRIC_COLUMNS: tuple[str, ...] = (
    "windows",
    "positives",
    "auroc",
    "auprc",
    "f1",
    "sensitivity",
    "specificity",
)


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, *, threshold: float) -> dict[str, float]:
    """AUROC/AUPRC are NaN when only one class is present."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = y_score >= threshold
    positives = y_true == 1
    tp = int((y_pred & positives).sum())
    fp = int((y_pred & ~positives).sum())
    fn = int((~y_pred & positives).sum())
    tn = int((~y_pred & ~positives).sum())
    both_classes = 0 < positives.sum() < len(y_true)
    return {
        "windows": len(y_true),
        "positives": int(positives.sum()),
        "auroc": float(roc_auc_score(y_true, y_score)) if both_classes else math.nan,
        "auprc": float(average_precision_score(y_true, y_score)) if both_classes else math.nan,
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
    }


def metrics_by_horizon(predictions: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    """One `all` row followed by one row per horizon (25..1); needs y, score, horizon."""
    rows = [{"horizon": "all", **binary_metrics(predictions["y"], predictions["score"], threshold=threshold)}]
    for horizon, group in sorted(predictions.groupby("horizon"), key=lambda item: -item[0]):
        rows.append(
            {"horizon": int(horizon), **binary_metrics(group["y"], group["score"], threshold=threshold)}
        )
    table = pd.DataFrame(rows, columns=["horizon", *METRIC_COLUMNS])
    table.insert(1, "threshold", threshold)
    return table


def select_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    target_sensitivity: float | None = None,
) -> float:
    """Pick a threshold on the precision-recall curve.

    Default: the threshold with the highest F1. With `target_sensitivity`: the
    highest threshold whose sensitivity (recall) is at least the target.
    """
    y_true = np.asarray(y_true).astype(int)
    if not 0 < y_true.sum() < len(y_true):
        raise ValueError("Threshold selection needs both classes in the validation data")
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    precision, recall = precision[:-1], recall[:-1]
    if target_sensitivity is None:
        denominator = precision + recall
        f1 = np.divide(
            2 * precision * recall,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        return float(thresholds[int(np.argmax(f1))])
    if not 0 < target_sensitivity <= 1:
        raise ValueError("target_sensitivity must be in (0, 1]")
    reaching = np.flatnonzero(recall >= target_sensitivity)
    return float(thresholds[reaching.max()])


def first_alert_horizons(
    predictions: pd.DataFrame, *, threshold: float, persistence: int = 1
) -> pd.DataFrame:
    """Per stay: the earliest horizon at which the alert has fired for `persistence` windows.

    With `persistence=1` this is the largest horizon whose window score reaches
    the threshold. With `persistence=p` the alert must hold for p consecutive
    windows (horizons h+p-1 .. h, i.e. p consecutive hours), and it is raised at
    the last of them, h. `first_alert_horizon` is the warning time in hours
    before the decision point (0 when the alert never fires).
    """
    if persistence < 1:
        raise ValueError("persistence must be at least 1")
    alerts = predictions.assign(alert=predictions["score"] >= threshold).sort_values(
        ["stay_id", "horizon"], ascending=[True, False]
    )
    if persistence > 1:
        grouped = alerts.groupby("stay_id", sort=False)
        held = grouped["alert"].rolling(persistence).min().to_numpy() == 1
        span = (grouped["horizon"].shift(persistence - 1) - alerts["horizon"]).to_numpy()
        alerts["alert"] = held & (span == persistence - 1)
    fired = alerts[alerts["alert"]].groupby("stay_id")["horizon"].max()
    stays = alerts.groupby("stay_id")["y"].first().rename("y").to_frame()
    stays["first_alert_horizon"] = fired.reindex(stays.index).fillna(0).astype(int)
    return stays.reset_index()


def early_detection_summary(alerts: pd.DataFrame) -> dict[str, float]:
    positive = alerts[alerts["y"] == 1]
    warned = positive["first_alert_horizon"] > 0
    return {
        "positive_stays": len(positive),
        "detected_stays": int(warned.sum()),
        "stay_detection_rate": _ratio(int(warned.sum()), len(positive)),
        "mean_warning_hours": float(positive["first_alert_horizon"].mean())
        if len(positive)
        else math.nan,
        "median_warning_hours": float(positive["first_alert_horizon"].median())
        if len(positive)
        else math.nan,
        "negative_stays_alerted": int((alerts.loc[alerts["y"] == 0, "first_alert_horizon"] > 0).sum()),
    }


def early_detection_advantage(alerts: pd.DataFrame, reference_alerts: pd.DataFrame) -> float:
    """Mean extra warning hours of a model over a reference, paired on positive stays."""
    paired = alerts.merge(reference_alerts, on=["stay_id", "y"], suffixes=("", "_reference"))
    if len(paired) != len(alerts) or len(paired) != len(reference_alerts):
        raise ValueError("Alert tables must cover the same stays with the same labels")
    positive = paired[paired["y"] == 1]
    if positive.empty:
        return math.nan
    lead = positive["first_alert_horizon"] - positive["first_alert_horizon_reference"]
    return float(lead.mean())


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else math.nan
