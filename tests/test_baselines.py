import numpy as np
import pytest

from cdss_project.baselines import (
    aggregate_windows,
    choose_validation_fold,
    make_model,
    run_baseline,
)
from cdss_project.windows import WindowDataset


def make_dataset(n_stays: int = 200, seed: int = 0) -> WindowDataset:
    """Positive stays have rising heart rate; signal grows toward the decision point."""
    rng = np.random.default_rng(seed)
    hours, window = 30, 6
    labels = (np.arange(n_stays) % 4 == 0).astype(np.int8)
    trend = np.linspace(0, 1, hours)
    heart_rate = 80 + rng.normal(0, 1, (n_stays, hours)) + labels[:, None] * 20 * trend
    gender = np.repeat(rng.integers(0, 2, n_stays)[:, None], hours, axis=1)
    values = np.stack([heart_rate, gender], axis=2).astype(np.float32)
    X = np.lib.stride_tricks.sliding_window_view(values, window, axis=1).transpose(0, 1, 3, 2)
    windows_per_stay = hours - window + 1
    fold = np.arange(n_stays) % 5
    return WindowDataset(
        X=np.ascontiguousarray(X).reshape(-1, window, 2),
        y=np.repeat(labels, windows_per_stay),
        horizon=np.tile(np.arange(windows_per_stay, 0, -1), n_stays),
        stay_id=np.repeat(np.arange(n_stays), windows_per_stay),
        feature_columns=("heart_rate", "gender"),
        observed_hours=np.full(n_stays * windows_per_stay, 6),
        fold=np.repeat(fold, windows_per_stay),
        split=np.repeat(np.where(fold == 0, "test", "train"), windows_per_stay),
    )


def test_aggregate_windows_mean_min_max_and_last_categorical():
    X = np.array([[[1.0, 0.0], [2.0, 0.0], [6.0, 1.0]]])

    features = aggregate_windows(X, ["heart_rate", "gender"])

    assert features.names == ("heart_rate_mean", "heart_rate_min", "heart_rate_max", "gender_last")
    assert features.values.tolist() == [[3.0, 1.0, 6.0, 1.0]]
    assert features.n_numeric == 3 and features.n_categorical == 1


def test_choose_validation_fold_is_next_training_fold():
    fold = np.array([0, 1, 2, 3, 4])
    assert choose_validation_fold(fold, np.array(["test"] + ["train"] * 4)) == (0, 1)
    assert choose_validation_fold(fold, np.array(["train"] * 4 + ["test"])) == (4, 0)
    with pytest.raises(ValueError, match="exactly one test fold"):
        choose_validation_fold(fold, np.array(["test", "test", "train", "train", "train"]))


def test_make_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown model"):
        make_model("svm", n_numeric=3, n_categorical=0)


@pytest.mark.parametrize("model_name", ["logistic_regression", "random_forest"])
def test_run_baseline_scores_only_test_fold_and_learns_signal(model_name):
    dataset = make_dataset()
    kwargs = {"rf_n_estimators": 20, "n_jobs": 1} if model_name == "random_forest" else {}

    result = run_baseline(dataset, model_name, **kwargs)

    assert (result.test_fold, result.validation_fold) == (0, 1)
    test_stays = set(dataset.stay_id[dataset.split == "test"])
    assert set(result.predictions["stay_id"]) == test_stays
    assert len(result.predictions) == len(test_stays) * 25
    overall = result.metrics.iloc[0]
    assert overall["horizon"] == "all" and overall["model"] == model_name
    assert overall["auroc"] > 0.9
    by_horizon = result.metrics.iloc[1:].set_index("horizon")["auroc"]
    assert by_horizon[1] >= by_horizon[25]
    assert result.early_detection["positive_stays"] == 10
    assert set(result.alerts["stay_id"]) == test_stays


def test_run_baseline_requires_folds():
    dataset = make_dataset(n_stays=20)
    without_folds = WindowDataset(
        X=dataset.X,
        y=dataset.y,
        horizon=dataset.horizon,
        stay_id=dataset.stay_id,
        feature_columns=dataset.feature_columns,
    )
    with pytest.raises(ValueError, match="fold and split"):
        run_baseline(without_folds, "logistic_regression")
