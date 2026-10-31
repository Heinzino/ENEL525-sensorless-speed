"""Full polynomial expansion of the three standardized inputs."""

import numpy as np


def polynomial_powers(degree, n_vars=3):
    """Exponent tuples (a, b, c, ...) with total degree from 0 through `degree`."""
    if degree < 0:
        raise ValueError("degree must be non-negative")
    powers = []

    def rec(var_index, remaining, acc):
        if var_index == n_vars - 1:
            powers.append(tuple(acc + [remaining]))
            return
        for k in range(remaining + 1):
            rec(var_index + 1, remaining - k, acc + [k])

    for total in range(degree + 1):
        rec(0, total, [])
    return powers


def polynomial_features(X, degree):
    """Design matrix with a leading bias column (the zero exponent)."""
    X = np.asarray(X, dtype=np.float64)
    powers = polynomial_powers(degree, n_vars=X.shape[1])
    cols = []
    for exp in powers:
        col = np.ones(X.shape[0], dtype=np.float64)
        for j, p in enumerate(exp):
            if p:
                col = col * X[:, j] ** p
        cols.append(col)
    return np.column_stack(cols), powers
