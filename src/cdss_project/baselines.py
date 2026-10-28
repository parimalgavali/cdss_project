"""Logistic Regression and Random Forest baselines on aggregated window features."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cdss_project.evaluation import (
    early_detection_summary,
    first_alert_horizons,
    metrics_by_horizon,
    select_threshold,
)
from cdss_project.features import CATEGORICAL_COLUMNS
from cdss_project.windows import WindowDataset

MODEL_NAMES: tuple[str, ...] = ("logistic_regression", "random_forest", "missingness_only")
AGGREGATIONS: tuple[str, ...] = ("mean", "min", "max")


@dataclass(frozen=True)
class AggregatedFeatures:
    values: np.ndarray
    names: tuple[str, ...]
    n_numeric: int

    @property
    def n_categorical(self) -> int:
        return len(self.names) - self.n_numeric


@dataclass(frozen=True)
class BaselineResult:
    model_name: str
    threshold: float
    validation_fold: int
    test_fold: int
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    alerts: pd.DataFrame
    early_detection: dict[str, float]
    params: dict[str, Any] = field(default_factory=dict)


def aggregate_windows(
    X: np.ndarray,
    feature_columns: Sequence[str],
    *,
    categorical_columns: Sequence[str] = CATEGORICAL_COLUMNS,
) -> AggregatedFeatures:
    """Mean/min/max per numeric feature; categorical codes taken from the last hour."""
    feature_columns = tuple(feature_columns)
    numeric = [i for i, name in enumerate(feature_columns) if name not in categorical_columns]
    categorical = [i for i, name in enumerate(feature_columns) if name in categorical_columns]
    numeric_values = X[:, :, numeric]
    blocks = [
        numeric_values.mean(axis=1),
        numeric_values.min(axis=1),
        numeric_values.max(axis=1),
        X[:, -1, categorical],
    ]
    names = [
        f"{feature_columns[i]}_{aggregation}" for aggregation in AGGREGATIONS for i in numeric
    ] + [f"{feature_columns[i]}_last" for i in categorical]
    return AggregatedFeatures(
        values=np.concatenate(blocks, axis=1).astype(np.float32),
        names=tuple(names),
        n_numeric=len(numeric) * len(AGGREGATIONS),
    )


def missingness_features(dataset: WindowDataset) -> AggregatedFeatures:
    """Which hours of each window are recorded (1) or structural padding (0), nothing else."""
    if dataset.observed_mask is None:
        raise ValueError("missingness_only needs observed_mask; rebuild the windows file")
    width = dataset.observed_mask.shape[1]
    return AggregatedFeatures(
        values=dataset.observed_mask.astype(np.float32),
        names=tuple(f"observed_hour_{width - i}" for i in range(width)),
        n_numeric=width,
    )


def make_model(
    model_name: str,
    *,
    n_numeric: int,
    n_categorical: int,
    random_state: int = 42,
    rf_n_estimators: int = 200,
    rf_min_samples_leaf: int = 20,
    n_jobs: int = -1,
) -> Pipeline:
    if model_name in ("logistic_regression", "missingness_only"):
        numeric_step: Any = StandardScaler()
        estimator: Any = LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=random_state
        )
    elif model_name == "random_forest":
        numeric_step = "passthrough"
        estimator = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            min_samples_leaf=rf_min_samples_leaf,
            class_weight="balanced_subsample",
            n_jobs=n_jobs,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model {model_name!r}; expected one of {MODEL_NAMES}")

    transformers: list[tuple[str, Any, slice]] = [("numeric", numeric_step, slice(0, n_numeric))]
    if n_categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                slice(n_numeric, n_numeric + n_categorical),
            )
        )
    return Pipeline(
        [("features", ColumnTransformer(transformers)), ("classifier", estimator)]
    )


def choose_validation_fold(fold: np.ndarray, split: np.ndarray) -> tuple[int, int]:
    """Return (test_fold, validation_fold); validation is the next training fold cyclically."""
    test_folds = np.unique(fold[split == "test"])
    train_folds = np.unique(fold[split == "train"])
    if len(test_folds) != 1:
        raise ValueError(f"Expected exactly one test fold, found {test_folds.tolist()}")
    if len(train_folds) < 2:
        raise ValueError("Need at least two training folds to hold one out for threshold tuning")
    test_fold = int(test_folds[0])
    later = train_folds[train_folds > test_fold]
    validation_fold = int(later[0] if len(later) else train_folds[0])
    return test_fold, validation_fold


def run_baseline(
    dataset: WindowDataset,
    model_name: str,
    *,
    target_sensitivity: float | None = None,
    alert_persistence: int = 1,
    refit: bool = True,
    features: AggregatedFeatures | None = None,
    **model_kwargs: Any,
) -> BaselineResult:
    """Fit on training stays, tune the threshold on a held-out training fold, score the test fold."""
    if dataset.fold is None or dataset.split is None:
        raise ValueError("Windows need fold and split arrays; rebuild them from the processed CSV")
    split = dataset.split.astype(str)
    test_fold, validation_fold = choose_validation_fold(dataset.fold, split)
    train = split == "train"
    test = split == "test"
    inner_train = train & (dataset.fold != validation_fold)
    validation = train & (dataset.fold == validation_fold)

    if model_name == "missingness_only":
        features = missingness_features(dataset)
    else:
        features = features or aggregate_windows(dataset.X, dataset.feature_columns)

    def new_model() -> Pipeline:
        return make_model(
            model_name,
            n_numeric=features.n_numeric,
            n_categorical=features.n_categorical,
            **model_kwargs,
        )

    model = new_model().fit(features.values[inner_train], dataset.y[inner_train])
    validation_scores = model.predict_proba(features.values[validation])[:, 1]
    threshold = select_threshold(
        dataset.y[validation], validation_scores, target_sensitivity=target_sensitivity
    )
    if refit:
        model = new_model().fit(features.values[train], dataset.y[train])

    predictions = pd.DataFrame(
        {
            "stay_id": dataset.stay_id[test],
            "horizon": dataset.horizon[test],
            "y": dataset.y[test],
            "score": model.predict_proba(features.values[test])[:, 1],
        }
    )
    if dataset.observed_hours is not None:
        predictions["observed_hours"] = dataset.observed_hours[test]

    metrics = metrics_by_horizon(predictions, threshold=threshold)
    metrics.insert(0, "model", model_name)
    alerts = first_alert_horizons(
        predictions, threshold=threshold, persistence=alert_persistence
    )
    return BaselineResult(
        model_name=model_name,
        threshold=threshold,
        validation_fold=validation_fold,
        test_fold=test_fold,
        predictions=predictions,
        metrics=metrics,
        alerts=alerts,
        early_detection=early_detection_summary(alerts),
        params={
            "target_sensitivity": target_sensitivity,
            "alert_persistence": alert_persistence,
            "refit": refit,
            **model_kwargs,
        },
    )
