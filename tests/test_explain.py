import numpy as np
import pytest
import torch

from cdss_project.explain import (
    event_shapley,
    feature_groups,
    feature_shapley,
    mean_encoded_window,
)
from cdss_project.rnn import (
    SepsisRNN,
    SequenceEncoder,
    load_rnn,
    predict_scores,
    save_rnn,
)

COLUMNS = ["heart_rate", "gender", "sbp"]


@pytest.fixture
def encoder():
    X = np.array([[[1.0, 0.0, 5.0], [3.0, 1.0, 7.0]]] * 2, dtype=np.float32)
    return SequenceEncoder.fit(X, COLUMNS)


def test_feature_groups_keep_one_hot_columns_together(encoder):
    groups = feature_groups(encoder, COLUMNS)
    assert [name for name, _ in groups] == ["heart_rate", "sbp", "gender"]
    assert groups[2][1].tolist() == [2, 3]


def test_mean_encoded_window_matches_direct_mean(encoder):
    X = np.random.default_rng(0).normal(size=(7, 2, 3)).astype(np.float32)
    X[:, :, 1] = np.random.default_rng(1).integers(0, 2, size=(7, 2))
    direct = encoder.transform(X).mean(axis=0)
    assert np.allclose(mean_encoded_window(encoder, X, chunk_size=3), direct, atol=1e-6)


def test_shapley_is_exact_for_additive_model():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 6, 4))
    baseline = rng.normal(size=(6, 4))
    weights = rng.normal(size=(6, 4))

    def predict(batch):
        return (batch * weights).sum(axis=(1, 2))

    contribution = (X - baseline) * weights
    events = event_shapley(predict, X, baseline)
    groups = [("a", np.array([0, 1])), ("b", np.array([2])), ("c", np.array([3]))]
    features = feature_shapley(predict, X, baseline, groups, n_permutations=3)

    assert np.allclose(events, contribution.sum(axis=2))
    assert np.allclose(features[:, 0], contribution[:, :, :2].sum(axis=(1, 2)))
    assert np.allclose(features[:, 2], contribution[:, :, 3].sum(axis=1))


def test_shapley_efficiency_on_rnn_and_save_load_round_trip(encoder, tmp_path):
    torch.manual_seed(0)
    model = SepsisRNN(encoder.n_features, cell="gru").eval()
    save_rnn(tmp_path, "gru", model, encoder)
    loaded, loaded_encoder = load_rnn(tmp_path, "gru")

    X = np.random.default_rng(2).normal(size=(4, 6, encoder.n_features)).astype(np.float32)
    device = torch.device("cpu")
    assert np.allclose(
        predict_scores(model, X, device=device), predict_scores(loaded, X, device=device)
    )
    assert np.array_equal(loaded_encoder.categories[0], encoder.categories[0])

    def predict(batch):
        return predict_scores(loaded, batch.astype(np.float32), device=device)

    baseline = np.zeros((6, encoder.n_features), dtype=np.float32)
    total = predict(X) - predict(baseline[None])
    events = event_shapley(predict, X, baseline)
    features = feature_shapley(
        predict, X, baseline, feature_groups(loaded_encoder, COLUMNS), n_permutations=4
    )
    assert np.allclose(events.sum(axis=1), total, atol=1e-5)
    assert np.allclose(features.sum(axis=1), total, atol=1e-5)
