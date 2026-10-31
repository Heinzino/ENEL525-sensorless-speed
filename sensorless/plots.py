"""Figures for the report and the notebook."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def _history(row):
    if not row or not row.get("history"):
        return None
    train = [step["train_rmse"] for step in row["history"]]
    val = [step["val_rmse"] for step in row["history"]]
    return np.arange(1, len(train) + 1), np.asarray(train), np.asarray(val)


def save_figures(prepared, search, tested, fig_dir):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    _eda(prepared, fig_dir)
    _val_bars(search, fig_dir)
    _curves(search, fig_dir)
    label = _pretty(search["best_overall"])
    _test_trace(tested, fig_dir, label)
    _residual(tested, fig_dir, label)


def _eda(prepared, fig_dir):
    t = prepared["t_dev"]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    axes[0].plot(t, prepared["y_dev"], color="#222222", lw=0.6, label="measured speed")
    axes[0].plot(t, prepared["w_model_dev"], color="#0072B2", lw=0.6, label="state-space speed")
    axes[0].axvline(32.0, color="#D55E00", ls="--", lw=1, label="train / validation cut")
    axes[0].set_ylabel("speed (rad/s)")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Development recording")
    axes[1].plot(t, prepared["X_dev_raw"][:, 0], color="#222222", lw=0.6, label="measured current")
    axes[1].plot(t, prepared["X_dev_raw"][:, 1], color="#009E73", lw=0.6, label="state-space current")
    axes[1].axvline(32.0, color="#D55E00", ls="--", lw=1)
    axes[1].set_ylabel("current (A)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_development.png")
    plt.close(fig)


def _val_bars(search, fig_dir):
    rows = [row for row in search["rows"] if np.isfinite(row["val_rmse"])]
    labels = [row["name"].replace("-", "\n") for row in rows]
    values = [row["val_rmse"] for row in rows]
    colors = {"baseline": "#666666", "ols": "#0072B2", "sgd": "#E69F00", "mlp": "#009E73"}
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.bar(np.arange(len(rows)), values, color=[colors[row["family"]] for row in rows])
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("validation RMSE (rad/s)")
    ax.set_title("Validation RMSE by configuration")
    fig.tight_layout()
    fig.savefig(fig_dir / "validation_rmse.png")
    plt.close(fig)


def _pretty(row):
    if row is None:
        return "selected model"
    if row["family"] == "sgd":
        return f"Degree-{row['degree']} SGD, learning rate {row['lr']:g}"
    if row["family"] == "mlp":
        return f"{row['hidden']}-unit network, learning rate {row['lr']:g}"
    if row["family"] == "ols":
        return f"Degree-{row['degree']} least squares"
    return row["name"]


def _curves(search, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=False)
    panels = (search["best_sgd"], search["best_mlp"])
    for ax, row in zip(axes, panels):
        parsed = _history(row)
        if parsed is None:
            ax.set_title("No finite run")
            continue
        epochs, train, val = parsed
        ax.plot(epochs, train, color="#0072B2", label="train")
        ax.plot(epochs, val, color="#D55E00", label="validation")
        ax.set_title(_pretty(row))
        ax.set_xlabel("epoch")
        ax.set_ylabel("RMSE (rad/s)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "learning_curves.png")
    plt.close(fig)


def _zoom(t, *series):
    mask = (t >= 25.0) & (t <= 35.0)
    return (t[mask],) + tuple(s[mask] for s in series)


def _test_trace(tested, fig_dir, winner_label):
    pred = tested["predictions"]
    t, y, base, winner = pred["t"], pred["y"], pred["baseline"], pred["winner"]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=False)
    axes[0].plot(t, y, color="#222222", lw=0.5, label="measured")
    axes[0].plot(t, base, color="#0072B2", lw=0.5, label="state-space")
    if winner is not None:
        axes[0].plot(t, winner, color="#D55E00", lw=0.5, label=winner_label)
    axes[0].set_ylabel("speed (rad/s)")
    axes[0].set_xlabel("time (s)")
    axes[0].set_title("Locked test recording")
    tz, yz, bz = _zoom(t, y, base)
    axes[1].plot(tz, yz, color="#222222", lw=1.0)
    axes[1].plot(tz, bz, color="#0072B2", lw=1.0)
    if winner is not None:
        wz = _zoom(t, winner)[1]
        axes[1].plot(tz, wz, color="#D55E00", lw=1.0)
    axes[1].set_ylabel("speed (rad/s)")
    axes[1].set_xlabel("time (s)")
    axes[1].set_title("Test recording, 25 s to 35 s")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(fig_dir / "test_trace.png")
    plt.close(fig)


def _residual(tested, fig_dir, winner_label):
    pred = tested["predictions"]
    winner = pred["winner"]
    if winner is None:
        return
    residual = winner - pred["y"]
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.plot(pred["t"], residual, color="#222222", lw=0.5)
    ax.axhline(0.0, color="#0072B2", lw=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("residual (rad/s)")
    ax.set_title(f"Test residual, {winner_label}")
    fig.tight_layout()
    fig.savefig(fig_dir / "test_residual.png")
    plt.close(fig)
