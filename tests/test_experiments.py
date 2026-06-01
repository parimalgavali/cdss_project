import json

import numpy as np
import pandas as pd
import torch

from cdss_project.experiments import (
    _choose_torch_device,
    compute_binary_metrics,
    flatten_for_classical_models,
    load_window_dataset,
    run_classical_experiment,
    run_neural_experiment,
    split_indices_by_stay,
    write_predictions_by_hour,
)
from cdss_project.windowing import WindowDataset, write_window_dataset


def _window_dataset() -> WindowDataset:
    metadata = pd.DataFrame(
        {
            "stay_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "prediction_hour_before": [30, 29, 30, 29, 30, 29, 30, 29],
            "label": [0, 0, 1, 1, 0, 0, 1, 1],
        }
    )
    return WindowDataset(
        X=np.arange(8 * 2 * 2, dtype=np.float32).reshape(8, 2, 2),
        y=np.array([0, 0, 1, 1, 0, 0, 1, 1]),
        metadata=metadata,
        feature_columns=("a", "b"),
        window_type="fixed",
        mask=np.array([[True, False], [True, True]] * 4),
    )


def _separable_window_dataset() -> WindowDataset:
    rows = []
    X = []
    y = []
    for stay_id in range(1, 9):
        label = int(stay_id % 2 == 0)
        for hour in [30, 29, 28]:
            rows.append(
                {
                    "stay_id": stay_id,
                    "prediction_hour_before": hour,
                    "label": label,
                    "window_type": "fixed",
                    "window_length": 2,
                }
            )
            X.append(np.full((2, 2), float(label), dtype=np.float32))
            y.append(label)
    return WindowDataset(
        X=np.array(X, dtype=np.float32),
        y=np.array(y, dtype=np.int64),
        metadata=pd.DataFrame(rows),
        feature_columns=("a", "b"),
        window_type="fixed",
        mask=np.ones((len(y), 2), dtype=bool),
    )


def test_load_window_dataset_round_trips_npz(tmp_path):
    path = tmp_path / "windows.npz"
    write_window_dataset(_window_dataset(), path)

    loaded = load_window_dataset(path)

    assert loaded.X.shape == (8, 2, 2)
    assert loaded.y.tolist() == [0, 0, 1, 1, 0, 0, 1, 1]
    assert loaded.feature_columns == ("a", "b")
    assert loaded.mask is not None
    assert loaded.metadata["stay_id"].tolist() == [1, 1, 2, 2, 3, 3, 4, 4]


def test_split_indices_by_stay_keeps_patient_windows_together():
    dataset = _window_dataset()

    split = split_indices_by_stay(dataset.metadata, test_size=0.25, val_size=0.25, random_state=7)

    train_stays = set(dataset.metadata.iloc[split.train_indices]["stay_id"])
    val_stays = set(dataset.metadata.iloc[split.val_indices]["stay_id"])
    test_stays = set(dataset.metadata.iloc[split.test_indices]["stay_id"])
    assert train_stays.isdisjoint(val_stays)
    assert train_stays.isdisjoint(test_stays)
    assert val_stays.isdisjoint(test_stays)
    assert len(split.train_indices) + len(split.val_indices) + len(split.test_indices) == 8


def test_flatten_for_classical_models_appends_mask_features():
    dataset = _window_dataset()

    flattened = flatten_for_classical_models(dataset.X, dataset.mask)

    assert flattened.shape == (8, 6)
    assert flattened[0].tolist() == [0.0, 1.0, 2.0, 3.0, 1.0, 0.0]


def test_compute_binary_metrics_uses_probabilities_and_threshold():
    metrics = compute_binary_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.4, 0.6, 0.9]),
        threshold=0.5,
    )

    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0


def test_choose_torch_device_prefers_mps_when_available(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert _choose_torch_device().type == "mps"


def test_choose_torch_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert _choose_torch_device().type == "cpu"


def test_write_predictions_by_hour_creates_one_file_per_hour(tmp_path):
    metadata = _window_dataset().metadata.copy()
    probabilities = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.4, 0.9, 0.6])

    write_predictions_by_hour(
        metadata,
        np.array(metadata["label"]),
        probabilities,
        output_dir=tmp_path,
        threshold=0.5,
    )

    hour_30 = tmp_path / "hour_30.csv"
    hour_29 = tmp_path / "hour_29.csv"
    assert hour_30.exists()
    assert hour_29.exists()
    rows = pd.read_csv(hour_30)
    assert rows["predicted_probability"].tolist() == [0.1, 0.8, 0.3, 0.9]
    config = json.loads((tmp_path / "prediction_schema.json").read_text())
    assert config["threshold"] == 0.5


def test_run_classical_experiment_writes_metrics_and_predictions(tmp_path):
    result = run_classical_experiment(
        _separable_window_dataset(),
        model_name="logistic_regression",
        output_dir=tmp_path,
        random_state=3,
    )

    assert result["model"] == "logistic_regression"
    assert (tmp_path / "metrics_overall.csv").exists()
    assert (tmp_path / "metrics_by_hour.csv").exists()
    assert (tmp_path / "predictions_by_hour" / "hour_30.csv").exists()
    assert (tmp_path / "model_config.json").exists()


def test_run_neural_experiment_writes_metrics_and_predictions(tmp_path):
    result = run_neural_experiment(
        _separable_window_dataset(),
        model_name="lstm",
        output_dir=tmp_path,
        epochs=1,
        batch_size=4,
        random_state=3,
    )

    assert result["model"] == "lstm"
    assert (tmp_path / "metrics_overall.csv").exists()
    assert (tmp_path / "metrics_by_hour.csv").exists()
    assert (tmp_path / "predictions_by_hour" / "hour_30.csv").exists()
    assert (tmp_path / "model_config.json").exists()
