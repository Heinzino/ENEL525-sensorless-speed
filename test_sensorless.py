"""Checks for the polynomial basis, the split, and one gradient step."""

import numpy as np

from sensorless.data import TRAIN_END_S, Standardizer, split_development
from sensorless.features import polynomial_features, polynomial_powers
from sensorless.metrics import mse
from sensorless.models import fit_mlp, fit_ols, fit_sgd, predict_linear, predict_mlp


def test_monomial_counts():
    assert len(polynomial_powers(2)) == 10
    assert len(polynomial_powers(5)) == 56
    assert polynomial_powers(1)[0] == (0, 0, 0)


def test_ols_recovers_a_quadratic():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 3))
    phi, powers = polynomial_features(x, 2)
    true = np.zeros(phi.shape[1])
    true[powers.index((0, 0, 0))] = 1.5
    true[powers.index((1, 0, 0))] = -0.7
    true[powers.index((1, 1, 0))] = 0.25
    y = phi @ true
    coef, info = fit_ols(phi, y)
    assert info["rank"] == phi.shape[1]
    assert np.allclose(coef, true, atol=1e-8)
    assert mse(y, predict_linear(phi, coef)) < 1e-16


def test_standardizer_uses_the_training_slice_only():
    recording = {
        "t": np.array([0.0, 31.0, 32.0, 40.0]),
        "X": np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [100.0, 100.0, 100.0], [100.0, 100.0, 100.0]]),
        "y": np.zeros(4),
    }
    train, val = split_development(recording, TRAIN_END_S)
    scaler = Standardizer().fit(recording["X"][train])
    assert np.allclose(scaler.mean_, [1.0, 1.0, 1.0])
    assert np.any(np.abs(scaler.transform(recording["X"][val]).mean(axis=0)) > 1.0)


def test_sgd_and_mlp_reduce_mse_on_a_linear_target():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(600, 3))
    y = 0.4 * x[:, 0] - 0.2 * x[:, 1] + 0.1 * x[:, 2]
    phi, _ = polynomial_features(x[:500], 1)
    phi_val, _ = polynomial_features(x[500:], 1)
    start = mse(y[:500], np.zeros(500))
    fitted = fit_sgd(phi, y[:500], phi_val, y[500:], lr=0.05, epochs=30, batch_size=64, seed=525)
    assert fitted is not None
    assert mse(y[:500], predict_linear(phi, fitted["coef"])) < start
    net = fit_mlp(x[:500], y[:500], x[500:], y[500:], hidden=16, lr=0.05, epochs=40, batch_size=64, seed=525)
    assert net is not None
    assert mse(y[:500], predict_mlp(x[:500], net["W1"], net["b1"], net["W2"], net["b2"])) < start


if __name__ == "__main__":
    test_monomial_counts()
    test_ols_recovers_a_quadratic()
    test_standardizer_uses_the_training_slice_only()
    test_sgd_and_mlp_reduce_mse_on_a_linear_target()
    print("tests passed")
