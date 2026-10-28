import numpy as np
import pytest
import torch
from test_baselines import make_dataset

from cdss_project.rnn import (
    SepsisRNN,
    SequenceEncoder,
    TrainConfig,
    run_rnn,
    train_model,
)


def test_encoder_standardizes_numeric_and_one_hots_categorical():
    X = np.array([[[1.0, 0.0, 5.0], [3.0, 1.0, 5.0]]], dtype=np.float32)
    encoder = SequenceEncoder.fit(X, ["heart_rate", "gender", "height"])

    encoded = encoder.transform(np.array([[[3.0, 1.0, 7.0], [1.0, 2.0, 5.0]]], dtype=np.float32))

    assert encoder.n_features == 4
    assert encoded[0, :, 0].tolist() == [1.0, -1.0]
    assert encoded[0, :, 1].tolist() == [2.0, 0.0]  # zero std falls back to 1
    assert encoded[0, 0, 2:].tolist() == [0.0, 1.0]
    assert encoded[0, 1, 2:].tolist() == [0.0, 0.0]  # unseen category -> all zeros


@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_model_outputs_one_logit_per_window(cell):
    model = SepsisRNN(5, cell=cell).eval()
    assert model(torch.zeros(4, 6, 5)).shape == (4,)


def test_gru_has_fewer_parameters_than_lstm():
    def count(model):
        return sum(p.numel() for p in model.parameters())

    assert count(SepsisRNN(26, cell="gru")) < count(SepsisRNN(26, cell="lstm"))
    with pytest.raises(ValueError, match="Unknown cell"):
        SepsisRNN(5, cell="rnn")


def test_early_stopping_stops_after_patience_without_improvement():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6, 3)).astype(np.float32)
    y = np.tile([0, 1], 20)
    config = TrainConfig(max_epochs=50, patience=2, learning_rate=0.0, min_delta=10.0)

    history = train_model(SepsisRNN(3), X, y, X, y, config)

    assert len(history) == 3


@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_run_rnn_learns_signal_and_scores_only_test_fold(cell):
    dataset = make_dataset()
    config = TrainConfig(max_epochs=15, patience=5, device="cpu")

    result = run_rnn(dataset, cell, config=config)

    test_stays = set(dataset.stay_id[dataset.split == "test"])
    assert set(result.predictions["stay_id"]) == test_stays
    assert (result.test_fold, result.validation_fold) == (0, 1)
    assert result.metrics.iloc[0]["auroc"] > 0.85
    assert result.metrics.iloc[0]["model"] == cell
    assert 1 <= len(result.history) <= 15
    assert result.params["best_epoch"] <= result.params["epochs_trained"]
