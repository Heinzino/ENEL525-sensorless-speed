"""Load the Simulink CSVs and align them onto the 1 ms model grid."""

from pathlib import Path

import numpy as np

MEASURED_DT = 5e-5
MODEL_DT = 1e-3
TRAIN_END_S = 32.0

FEATURE_NAMES = ("ia_measured", "i_model", "w_model")
TARGET_NAME = "w_measured"


def _load_csv(path):
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.array([arr], dtype=arr.dtype)
    return arr


def _pair(measured_path, model_path):
    measured = _load_csv(measured_path)
    model = _load_csv(model_path)
    t_m = np.asarray(measured["time"], dtype=np.float64)
    # genfromtxt renames a repeated header time, time_1, ...
    t_model = np.asarray(model["time"], dtype=np.float64)
    idx = np.rint(t_model / MEASURED_DT).astype(np.int64)
    if np.any(idx < 0) or np.any(idx >= len(t_m)):
        raise ValueError(f"model time falls outside measured recording: {measured_path.name}")
    t_err = np.max(np.abs(t_m[idx] - t_model))
    if t_err > 1e-9:
        raise ValueError(f"alignment error {t_err} s in {measured_path.name}")
    ia = np.asarray(measured["ia_measured"], dtype=np.float64)[idx]
    w_measured = np.asarray(measured["w_measured"], dtype=np.float64)[idx]
    i_model = np.asarray(model["i_model"], dtype=np.float64)
    w_model = np.asarray(model["w_model"], dtype=np.float64)
    features = np.column_stack([ia, i_model, w_model])
    return {
        "t": t_model,
        "X": features,
        "y": w_measured,
        "alignment_error_s": float(t_err),
    }


def load_recordings(data_dir):
    root = Path(data_dir)
    development = _pair(root / "MeasuredV4.csv", root / "ModelV4.csv")
    test = _pair(root / "TestMeasuredV4.csv", root / "TestModelV4.csv")
    return development, test


def split_development(recording, train_end_s=TRAIN_END_S):
    t = recording["t"]
    train = t < train_end_s
    val = ~train
    if not np.any(train) or not np.any(val):
        raise ValueError("train or validation slice is empty")
    return train, val


class Standardizer:
    """Feature standardization fit on the training slice only."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_ = np.where(self.std_ == 0.0, 1.0, self.std_)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_
