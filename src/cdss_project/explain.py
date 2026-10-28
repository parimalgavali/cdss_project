"""TimeSHAP-style Shapley explanations for window models (outline 3.6).

Event level: exact Shapley values over the time steps of a window.
Feature level: permutation-sampling Shapley over original features (one-hot groups
kept together). Absent players take the value of a baseline (average training window).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence

import numpy as np

from cdss_project.rnn import SequenceEncoder

Predictor = Callable[[np.ndarray], np.ndarray]


def feature_groups(
    encoder: SequenceEncoder, feature_columns: Sequence[str]
) -> list[tuple[str, np.ndarray]]:
    """Map encoded columns back to original feature names, in encoder column order."""
    names = tuple(feature_columns)
    groups = [(names[i], np.array([position])) for position, i in enumerate(encoder.numeric_index)]
    start = len(encoder.numeric_index)
    for index, values in zip(encoder.categorical_index, encoder.categories, strict=True):
        groups.append((names[index], np.arange(start, start + len(values))))
        start += len(values)
    return groups


def mean_encoded_window(
    encoder: SequenceEncoder, X: np.ndarray, *, chunk_size: int = 100_000
) -> np.ndarray:
    """Average encoded window (time_steps, encoded_features), computed in chunks."""
    total = np.zeros((X.shape[1], encoder.n_features), dtype=np.float64)
    for start in range(0, len(X), chunk_size):
        total += encoder.transform(X[start : start + chunk_size]).sum(axis=0, dtype=np.float64)
    return (total / len(X)).astype(np.float32)


def event_shapley(predict: Predictor, X: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Exact Shapley value of each time step, shape (n_windows, time_steps)."""
    n_windows, n_steps, _ = X.shape
    masks = np.array(list(itertools.product((False, True), repeat=n_steps)))
    codes = masks @ (1 << np.arange(n_steps))
    values = np.empty((1 << n_steps, n_windows))
    for mask, code in zip(masks, codes, strict=True):
        values[code] = predict(np.where(mask[None, :, None], X, baseline[None]))

    phi = np.zeros((n_windows, n_steps))
    for mask, code in zip(masks, codes, strict=True):
        size = int(mask.sum())
        if size == n_steps:
            continue
        weight = math.factorial(size) * math.factorial(n_steps - size - 1) / math.factorial(n_steps)
        for step in np.flatnonzero(~mask):
            phi[:, step] += weight * (values[code | (1 << step)] - values[code])
    return phi


def feature_shapley(
    predict: Predictor,
    X: np.ndarray,
    baseline: np.ndarray,
    groups: Sequence[tuple[str, np.ndarray]],
    *,
    n_permutations: int = 32,
    seed: int = 42,
) -> np.ndarray:
    """Permutation-sampling Shapley value per feature group, shape (n_windows, n_groups)."""
    rng = np.random.default_rng(seed)
    phi = np.zeros((len(X), len(groups)))
    start = np.broadcast_to(baseline, X.shape).copy()
    start_value = predict(start)
    for _ in range(n_permutations):
        current = start.copy()
        previous = start_value
        for group in rng.permutation(len(groups)):
            columns = groups[group][1]
            current[:, :, columns] = X[:, :, columns]
            value = predict(current)
            phi[:, group] += value - previous
            previous = value
    return phi / n_permutations
