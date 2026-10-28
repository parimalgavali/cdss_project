"""Synthetic hourly ICU data that follows the real CSV contract.

Only for tests and pipeline smoke runs: values are random, not clinical.
Each stay has leading structural padding (``sepsis`` and every feature NaN)
and, for sepsis-positive stays, a drift in a few vitals that grows towards
the decision point so that short horizons are easier to predict than long ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cdss_project.features import (
    CATEGORICAL_COLUMNS,
    DYNAMIC_NUMERIC_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
)

HOURS_PER_STAY = 30
FIRST_STAY_ID = 30_000_000
CATEGORY_LEVELS = {"gender": 2, "ethnicity": 5, "last_careunit": 8}
DRIFT_COLUMNS = ("heart_rate", "resp_rate", "lactate", "temperature", "sofa_24hours")


def make_synthetic_frame(
    n_stays: int = 2000,
    *,
    positive_rate: float = 0.17,
    max_padding_hours: int = 24,
    missing_rate: float = 0.4,
    signal: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Return ``n_stays * 30`` rows with the required columns, ordered like the real CSV."""
    if n_stays < 1:
        raise ValueError("n_stays must be positive")
    if not 0 <= max_padding_hours < HOURS_PER_STAY:
        raise ValueError(f"max_padding_hours must be in [0, {HOURS_PER_STAY - 1}]")
    rng = np.random.default_rng(seed)
    n_rows = n_stays * HOURS_PER_STAY

    stay_index = np.repeat(np.arange(n_stays), HOURS_PER_STAY)
    hour_before = np.tile(np.arange(HOURS_PER_STAY, 0, -1), n_stays)
    hour_after_admission = HOURS_PER_STAY - hour_before

    label = (rng.random(n_stays) < positive_rate).astype(float)
    padding_hours = rng.integers(0, max_padding_hours + 1, size=n_stays)
    is_padding = hour_after_admission < padding_hours[stay_index]
    row_label = label[stay_index]

    frame = pd.DataFrame({"stay_id": FIRST_STAY_ID + stay_index, "hour_before": hour_before})
    static = {
        "admission_age": rng.normal(65, 15, n_stays).clip(18, 100),
        "height": rng.normal(170, 10, n_stays),
        "weight": rng.normal(80, 15, n_stays).clip(35, 200),
        "diabetes": rng.integers(0, 2, n_stays).astype(float),
    }
    for column, levels in CATEGORY_LEVELS.items():
        static[column] = rng.integers(0, levels, n_stays).astype(float)
    for column, values in static.items():
        frame[column] = values[stay_index]

    # Drift ramps from 0 at hour_before=30 to `signal` standard deviations at hour_before=1.
    ramp = signal * (HOURS_PER_STAY - hour_before) / (HOURS_PER_STAY - 1)
    for column in DYNAMIC_NUMERIC_COLUMNS:
        values = rng.normal(0.0, 1.0, n_rows)
        if column in DRIFT_COLUMNS:
            values += row_label * ramp
        values[rng.random(n_rows) < missing_rate] = np.nan
        frame[column] = values

    feature_columns = [column for column in FEATURE_COLUMNS if column in frame.columns]
    frame.loc[is_padding, feature_columns] = np.nan
    frame[LABEL_COLUMN] = np.where(is_padding, np.nan, row_label)
    missing_features = set(FEATURE_COLUMNS) - set(frame.columns) - set(CATEGORICAL_COLUMNS)
    if missing_features:
        raise RuntimeError(f"Synthetic generator is missing features: {sorted(missing_features)}")
    return frame
