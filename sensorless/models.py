"""Baseline, ordinary least squares, plain SGD, and a one-hidden-layer network."""

import numpy as np

from sensorless.metrics import mae, mse, rmse


def _finite(arr):
    return np.all(np.isfinite(arr))


def fit_ols(phi, y):
    coef, residuals, rank, singular = np.linalg.lstsq(phi, y, rcond=None)
    cond = float(singular[0] / singular[-1]) if singular.size and singular[-1] > 0 else float("inf")
    return coef, {"rank": int(rank), "condition_number": cond}


def predict_linear(phi, coef):
    return phi @ coef


def _epoch_metrics(phi, y, phi_val, y_val, w):
    pred = phi @ w
    pred_val = phi_val @ w
    if not _finite(pred) or not _finite(pred_val):
        return None
    return {
        "train_rmse": rmse(y, pred),
        "val_rmse": rmse(y_val, pred_val),
    }


def fit_sgd(phi, y, phi_val, y_val, lr, epochs, batch_size, seed):
    """Plain minibatch SGD. Column 0 is the bias and stays initialized at 0.

    Returns None when a step diverges.
    """
    rng = np.random.default_rng(seed)
    n, n_features = phi.shape
    w = np.zeros(n_features, dtype=np.float64)
    fan_in = max(n_features - 1, 1)
    w[1:] = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=fan_in)
    history = []
    for _epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = order[start:start + batch_size]
            xb = phi[batch]
            err = xb @ w - y[batch]
            grad = (2.0 / batch.size) * (xb.T @ err)
            w = w - lr * grad
            if not _finite(w):
                return None
        step = _epoch_metrics(phi, y, phi_val, y_val, w)
        if step is None:
            return None
        history.append(step)
    return {"coef": w, "history": history}


def _relu(z):
    return np.maximum(z, 0.0)


def fit_mlp(X, y, X_val, y_val, hidden, lr, epochs, batch_size, seed):
    """One hidden ReLU layer, linear output, mean-squared-error loss.

    Returns None when a step diverges.
    """
    rng = np.random.default_rng(seed)
    n, n_in = X.shape
    W1 = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(hidden, n_in))
    b1 = np.zeros(hidden, dtype=np.float64)
    W2 = rng.normal(0.0, np.sqrt(2.0 / hidden), size=(hidden,))
    b2 = 0.0
    history = []
    for _epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = order[start:start + batch_size]
            xb = X[batch]
            yb = y[batch]
            z1 = xb @ W1.T + b1
            h = _relu(z1)
            pred = h @ W2 + b2
            if not _finite(pred):
                return None
            bsz = batch.size
            delta2 = (2.0 / bsz) * (pred - yb)
            dW2 = h.T @ delta2
            db2 = float(np.sum(delta2))
            dh = delta2[:, None] * W2[None, :]
            dz = dh * (z1 > 0.0)
            dW1 = dz.T @ xb
            db1 = np.sum(dz, axis=0)
            W1 = W1 - lr * dW1
            b1 = b1 - lr * db1
            W2 = W2 - lr * dW2
            b2 = b2 - lr * db2
            if not (_finite(W1) and _finite(b1) and _finite(W2) and np.isfinite(b2)):
                return None
        pred_tr = predict_mlp(X, W1, b1, W2, b2)
        pred_va = predict_mlp(X_val, W1, b1, W2, b2)
        if not _finite(pred_tr) or not _finite(pred_va):
            return None
        history.append({"train_rmse": rmse(y, pred_tr), "val_rmse": rmse(y_val, pred_va)})
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "history": history}


def predict_mlp(X, W1, b1, W2, b2):
    h = _relu(X @ W1.T + b1)
    return h @ W2 + b2


def regression_scores(y_true, y_pred):
    return {"mse": mse(y_true, y_pred), "rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred)}
