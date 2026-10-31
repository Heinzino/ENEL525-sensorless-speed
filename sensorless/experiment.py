"""Locked ENEL 525 experiment: tune on validation, then touch the test files once."""

import json
from pathlib import Path

import numpy as np

from sensorless.data import FEATURE_NAMES, TRAIN_END_S, Standardizer, load_recordings, split_development
from sensorless.features import polynomial_features
from sensorless.metrics import metric_triple
from sensorless.models import fit_mlp, fit_ols, fit_sgd, predict_linear, predict_mlp

DEGREES = (2, 3, 4, 5)
LEARNING_RATES = (1e-3, 1e-2, 1e-1)
HIDDEN_WIDTHS = (16, 32)
EPOCHS = 100
BATCH_SIZE = 256
SELECTION_SEED = 525
UNCERTAINTY_SEEDS = (525, 526, 527, 528, 529)


def _scores(y, pred):
    return metric_triple(y, pred)


def _prepare(data_dir):
    development, test = load_recordings(data_dir)
    train_mask, val_mask = split_development(development)
    scaler = Standardizer().fit(development["X"][train_mask])
    dev_x = scaler.transform(development["X"])
    test_x = scaler.transform(test["X"])
    prepared = {
        "t_dev": development["t"],
        "y_dev": development["y"],
        "w_model_dev": development["X"][:, 2],
        "X_dev_raw": development["X"],
        "X_dev": dev_x,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "t_test": test["t"],
        "y_test": test["y"],
        "w_model_test": test["X"][:, 2],
        "X_test_raw": test["X"],
        "X_test": test_x,
        "scaler_mean": scaler.mean_,
        "scaler_std": scaler.std_,
        "alignment_error_s": development["alignment_error_s"],
    }
    return prepared


def _select(rows):
    """Lowest validation RMSE, then the simpler setting."""
    finite = [row for row in rows if np.isfinite(row["val_rmse"])]
    if not finite:
        return None
    return min(finite, key=lambda row: (row["val_rmse"], row["simplicity"], row["name"]))


def run_search(prepared, epochs=EPOCHS, batch_size=BATCH_SIZE, seed=SELECTION_SEED, progress=print):
    """Fit every agreed configuration. The test arrays are not read here."""
    train = prepared["train_mask"]
    val = prepared["val_mask"]
    x_train = prepared["X_dev"][train]
    y_train = prepared["y_dev"][train]
    x_val = prepared["X_dev"][val]
    y_val = prepared["y_dev"][val]
    baseline_pred_val = prepared["w_model_dev"][val]
    rows = [{
        "family": "baseline",
        "name": "state-space w_model",
        "degree": None,
        "lr": None,
        "hidden": None,
        "val_rmse": _scores(y_val, baseline_pred_val)["rmse"],
        "val_metrics": _scores(y_val, baseline_pred_val),
        "train_metrics": _scores(prepared["y_dev"][train], prepared["w_model_dev"][train]),
        "simplicity": 0,
        "diverged": False,
        "history": None,
        "condition_number": None,
    }]
    designs = {}
    for degree in DEGREES:
        phi_train, powers = polynomial_features(x_train, degree)
        phi_val, _ = polynomial_features(x_val, degree)
        designs[degree] = powers
        coef, info = fit_ols(phi_train, y_train)
        pred_val = predict_linear(phi_val, coef)
        rows.append({
            "family": "ols",
            "name": f"ols-degree-{degree}",
            "degree": degree,
            "lr": None,
            "hidden": None,
            "n_terms": int(phi_train.shape[1]),
            "val_rmse": _scores(y_val, pred_val)["rmse"],
            "val_metrics": _scores(y_val, pred_val),
            "train_metrics": _scores(y_train, predict_linear(phi_train, coef)),
            "simplicity": 10 + degree,
            "diverged": False,
            "history": None,
            "condition_number": info["condition_number"],
            "coef": coef,
        })
        progress(f"OLS degree {degree}: val RMSE {rows[-1]['val_rmse']:.4f}, cond {info['condition_number']:.3g}")
        for lr in LEARNING_RATES:
            progress(f"SGD degree {degree}, lr {lr}")
            fitted = fit_sgd(phi_train, y_train, phi_val, y_val, lr, epochs, batch_size, seed)
            if fitted is None:
                rows.append({
                    "family": "sgd",
                    "name": f"sgd-degree-{degree}-lr-{lr}",
                    "degree": degree,
                    "lr": lr,
                    "hidden": None,
                    "n_terms": int(phi_train.shape[1]),
                    "val_rmse": float("inf"),
                    "val_metrics": None,
                    "train_metrics": None,
                    "simplicity": 20 + degree,
                    "diverged": True,
                    "history": None,
                    "condition_number": None,
                })
                progress("  diverged")
                continue
            final = fitted["history"][-1]
            rows.append({
                "family": "sgd",
                "name": f"sgd-degree-{degree}-lr-{lr}",
                "degree": degree,
                "lr": lr,
                "hidden": None,
                "n_terms": int(phi_train.shape[1]),
                "val_rmse": final["val_rmse"],
                "val_metrics": _scores(y_val, predict_linear(phi_val, fitted["coef"])),
                "train_metrics": _scores(y_train, predict_linear(phi_train, fitted["coef"])),
                "simplicity": 20 + degree + lr,
                "diverged": False,
                "history": fitted["history"],
                "condition_number": None,
                "coef": fitted["coef"],
            })
            progress(f"  val RMSE {final['val_rmse']:.4f}")
    for hidden in HIDDEN_WIDTHS:
        for lr in LEARNING_RATES:
            progress(f"MLP width {hidden}, lr {lr}")
            fitted = fit_mlp(x_train, y_train, x_val, y_val, hidden, lr, epochs, batch_size, seed)
            if fitted is None:
                rows.append({
                    "family": "mlp",
                    "name": f"mlp-width-{hidden}-lr-{lr}",
                    "degree": None,
                    "lr": lr,
                    "hidden": hidden,
                    "val_rmse": float("inf"),
                    "val_metrics": None,
                    "train_metrics": None,
                    "simplicity": 30 + hidden,
                    "diverged": True,
                    "history": None,
                    "condition_number": None,
                })
                progress("  diverged")
                continue
            final = fitted["history"][-1]
            pred_val = predict_mlp(x_val, fitted["W1"], fitted["b1"], fitted["W2"], fitted["b2"])
            pred_train = predict_mlp(x_train, fitted["W1"], fitted["b1"], fitted["W2"], fitted["b2"])
            rows.append({
                "family": "mlp",
                "name": f"mlp-width-{hidden}-lr-{lr}",
                "degree": None,
                "lr": lr,
                "hidden": hidden,
                "val_rmse": final["val_rmse"],
                "val_metrics": _scores(y_val, pred_val),
                "train_metrics": _scores(y_train, pred_train),
                "simplicity": 30 + hidden + lr,
                "diverged": False,
                "history": fitted["history"],
                "condition_number": None,
                "pack": fitted,
            })
            progress(f"  val RMSE {final['val_rmse']:.4f}")
    best_ols = _select([row for row in rows if row["family"] == "ols"])
    best_sgd = _select([row for row in rows if row["family"] == "sgd"])
    best_mlp = _select([row for row in rows if row["family"] == "mlp"])
    best_overall = _select(rows)
    return {
        "rows": rows,
        "best_ols": best_ols,
        "best_sgd": best_sgd,
        "best_mlp": best_mlp,
        "best_overall": best_overall,
        "powers": {str(degree): designs[degree] for degree in designs},
    }


def _refit_sgd(prepared, degree, lr, seed, epochs, batch_size):
    train = prepared["train_mask"]
    val = prepared["val_mask"]
    phi_train, _ = polynomial_features(prepared["X_dev"][train], degree)
    phi_val, _ = polynomial_features(prepared["X_dev"][val], degree)
    phi_test, _ = polynomial_features(prepared["X_test"], degree)
    fitted = fit_sgd(
        phi_train, prepared["y_dev"][train], phi_val, prepared["y_dev"][val],
        lr, epochs, batch_size, seed,
    )
    if fitted is None:
        return None
    pred_test = predict_linear(phi_test, fitted["coef"])
    return fitted, pred_test


def _refit_mlp(prepared, hidden, lr, seed, epochs, batch_size):
    train = prepared["train_mask"]
    val = prepared["val_mask"]
    fitted = fit_mlp(
        prepared["X_dev"][train], prepared["y_dev"][train],
        prepared["X_dev"][val], prepared["y_dev"][val],
        hidden, lr, epochs, batch_size, seed,
    )
    if fitted is None:
        return None
    pred_test = predict_mlp(prepared["X_test"], fitted["W1"], fitted["b1"], fitted["W2"], fitted["b2"])
    return fitted, pred_test


def evaluate_test(prepared, search, epochs=EPOCHS, batch_size=BATCH_SIZE, seeds=UNCERTAINTY_SEEDS, progress=print):
    """One locked test evaluation of the validation winners."""
    y_test = prepared["y_test"]
    baseline = _scores(y_test, prepared["w_model_test"])
    ols = search["best_ols"]
    train = prepared["train_mask"]
    phi_train, _ = polynomial_features(prepared["X_dev"][train], ols["degree"])
    phi_test, _ = polynomial_features(prepared["X_test"], ols["degree"])
    coef, info = fit_ols(phi_train, prepared["y_dev"][train])
    ols_pred = predict_linear(phi_test, coef)
    ols_test = _scores(y_test, ols_pred)
    ols_test["degree"] = ols["degree"]
    ols_test["condition_number"] = info["condition_number"]

    sgd_runs = []
    sgd_pred_seed = None
    sgd_spec = search["best_sgd"]
    for seed in seeds:
        progress(f"test SGD seed {seed}")
        refit = _refit_sgd(prepared, sgd_spec["degree"], sgd_spec["lr"], seed, epochs, batch_size)
        if refit is None:
            sgd_runs.append({"seed": seed, "diverged": True})
            continue
        _fitted, pred = refit
        sgd_runs.append({"seed": seed, "diverged": False, **_scores(y_test, pred)})
        if seed == SELECTION_SEED:
            sgd_pred_seed = pred

    mlp_runs = []
    mlp_pred_seed = None
    mlp_spec = search["best_mlp"]
    for seed in seeds:
        progress(f"test MLP seed {seed}")
        refit = _refit_mlp(prepared, mlp_spec["hidden"], mlp_spec["lr"], seed, epochs, batch_size)
        if refit is None:
            mlp_runs.append({"seed": seed, "diverged": True})
            continue
        _fitted, pred = refit
        mlp_runs.append({"seed": seed, "diverged": False, **_scores(y_test, pred)})
        if seed == SELECTION_SEED:
            mlp_pred_seed = pred

    def _summary(runs):
        vals = [run["rmse"] for run in runs if not run["diverged"]]
        return {
            "runs": runs,
            "rmse_mean": float(np.mean(vals)) if vals else None,
            "rmse_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "mse_mean": float(np.mean([run["mse"] for run in runs if not run["diverged"]])) if vals else None,
            "mae_mean": float(np.mean([run["mae"] for run in runs if not run["diverged"]])) if vals else None,
        }

    candidates = {
        "baseline": (baseline["rmse"], prepared["w_model_test"]),
        "ols": (ols_test["rmse"], ols_pred),
    }
    if sgd_pred_seed is not None:
        candidates["sgd"] = (_scores(y_test, sgd_pred_seed)["rmse"], sgd_pred_seed)
    if mlp_pred_seed is not None:
        candidates["mlp"] = (_scores(y_test, mlp_pred_seed)["rmse"], mlp_pred_seed)
    # Residual figure uses the validation winner, scored here only for display.
    winner_name = search["best_overall"]["family"]
    if winner_name == "baseline":
        winner_pred = prepared["w_model_test"]
    elif winner_name == "ols":
        winner_pred = ols_pred
    elif winner_name == "sgd":
        winner_pred = sgd_pred_seed
    else:
        winner_pred = mlp_pred_seed
    return {
        "baseline": baseline,
        "ols": ols_test,
        "sgd": {**_summary(sgd_runs), "degree": sgd_spec["degree"], "lr": sgd_spec["lr"]},
        "mlp": {**_summary(mlp_runs), "hidden": mlp_spec["hidden"], "lr": mlp_spec["lr"]},
        "winner_family": winner_name,
        "winner_name": search["best_overall"]["name"],
        "predictions": {
            "t": prepared["t_test"],
            "y": y_test,
            "baseline": prepared["w_model_test"],
            "ols": ols_pred,
            "sgd": sgd_pred_seed,
            "mlp": mlp_pred_seed,
            "winner": winner_pred,
        },
    }


def _jsonable_row(row):
    history = row["history"]
    return {
        "family": row["family"],
        "name": row["name"],
        "degree": row["degree"],
        "lr": row["lr"],
        "hidden": row["hidden"],
        "n_terms": row.get("n_terms"),
        "val_rmse": None if not np.isfinite(row["val_rmse"]) else row["val_rmse"],
        "val_metrics": row["val_metrics"],
        "train_metrics": row["train_metrics"],
        "diverged": row["diverged"],
        "condition_number": row["condition_number"],
        "history": history,
    }


def run_experiment(data_dir, out_dir, progress=print):
    out = Path(out_dir)
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    prepared = _prepare(data_dir)
    search = run_search(prepared, progress=progress)
    # Test labels are used only after selection.
    tested = evaluate_test(prepared, search, progress=progress)
    from sensorless.plots import save_figures
    save_figures(prepared, search, tested, fig)
    np.savez_compressed(
        out / "test_predictions.npz",
        t=tested["predictions"]["t"],
        y=tested["predictions"]["y"],
        baseline=tested["predictions"]["baseline"],
        ols=tested["predictions"]["ols"],
        sgd=tested["predictions"]["sgd"] if tested["predictions"]["sgd"] is not None else np.array([]),
        mlp=tested["predictions"]["mlp"] if tested["predictions"]["mlp"] is not None else np.array([]),
        winner=tested["predictions"]["winner"] if tested["predictions"]["winner"] is not None else np.array([]),
    )
    summary = {
        "train_end_s": TRAIN_END_S,
        "n_train": int(np.sum(prepared["train_mask"])),
        "n_val": int(np.sum(prepared["val_mask"])),
        "n_test": int(prepared["y_test"].shape[0]),
        "feature_names": list(FEATURE_NAMES),
        "scaler_mean": prepared["scaler_mean"].tolist(),
        "scaler_std": prepared["scaler_std"].tolist(),
        "alignment_error_s": prepared["alignment_error_s"],
        "selection_seed": SELECTION_SEED,
        "uncertainty_seeds": list(UNCERTAINTY_SEEDS),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rates": list(LEARNING_RATES),
        "degrees": list(DEGREES),
        "hidden_widths": list(HIDDEN_WIDTHS),
        "rows": [_jsonable_row(row) for row in search["rows"]],
        "best_ols": search["best_ols"]["name"],
        "best_sgd": search["best_sgd"]["name"] if search["best_sgd"] else None,
        "best_mlp": search["best_mlp"]["name"] if search["best_mlp"] else None,
        "best_overall": search["best_overall"]["name"],
        "test": {
            "baseline": tested["baseline"],
            "ols": tested["ols"],
            "sgd": {k: v for k, v in tested["sgd"].items() if k != "runs"} | {"runs": tested["sgd"]["runs"]},
            "mlp": {k: v for k, v in tested["mlp"].items() if k != "runs"} | {"runs": tested["mlp"]["runs"]},
            "winner_family": tested["winner_family"],
            "winner_name": tested["winner_name"],
        },
    }
    # The merge above is fine in 3.9+; keep it explicit for readability if needed.
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_val_table(summary["rows"], out / "validation_results.csv")
    progress(f"wrote {out / 'summary.json'}")
    return summary


def _write_val_table(rows, path):
    lines = ["family,name,degree,hidden,lr,val_rmse,val_mse,val_mae,train_rmse,diverged,condition_number"]
    for row in rows:
        val = row["val_metrics"] or {}
        train = row["train_metrics"] or {}
        lines.append(",".join([
            row["family"],
            row["name"],
            "" if row["degree"] is None else str(row["degree"]),
            "" if row["hidden"] is None else str(row["hidden"]),
            "" if row["lr"] is None else repr(row["lr"]),
            "" if row["val_rmse"] is None else f"{row['val_rmse']:.8g}",
            "" if val.get("mse") is None else f"{val['mse']:.8g}",
            "" if val.get("mae") is None else f"{val['mae']:.8g}",
            "" if train.get("rmse") is None else f"{train['rmse']:.8g}",
            str(row["diverged"]),
            "" if row["condition_number"] is None else f"{row['condition_number']:.8g}",
        ]))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
