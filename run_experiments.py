"""Run the locked sensorless-speed experiment and write results/."""

from pathlib import Path

from sensorless.experiment import run_experiment

ROOT = Path(__file__).resolve().parent


def main():
    summary = run_experiment(ROOT, ROOT / "results")
    print("best overall:", summary["best_overall"])
    print("test baseline RMSE:", summary["test"]["baseline"]["rmse"])
    print("test OLS RMSE:", summary["test"]["ols"]["rmse"])
    print("test SGD RMSE mean:", summary["test"]["sgd"]["rmse_mean"])
    print("test MLP RMSE mean:", summary["test"]["mlp"]["rmse_mean"])


if __name__ == "__main__":
    main()
