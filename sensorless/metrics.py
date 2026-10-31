"""Regression metrics in radians per second."""

import numpy as np


def mse(y_true, y_pred):
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.mean(err ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred):
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.mean(np.abs(err)))


def metric_triple(y_true, y_pred):
    return {"mse": mse(y_true, y_pred), "rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred)}
