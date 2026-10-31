# ENEL 525 course project

Sensorless shaft-speed estimation for a brushed DC motor. The code compares a state-space baseline, full polynomials of degree 2 through 5, and a one-hidden-layer network. Mohammad Elmougey and Heinz Veintimilla.

## Reproduce

From this directory, with Python 3.13:

```text
python -m pip install -r requirements.txt
python test_sensorless.py
python run_experiments.py
```

`run_experiments.py` writes `results/summary.json`, `results/validation_results.csv`, `results/test_predictions.npz`, and the figures in `results/figures/`. The test files are scored only after the validation winner is chosen. The selection seed is 525. Uncertainty seeds are 525 through 529. Training is `t < 32` s on the development recording.

The notebook `ENEL525_sensorless_speed.ipynb` loads those results. Set `RECOMPUTE = True` in the first code cell to fit the models again.

The IEEE report is `report/main.tex`. With [Tectonic](https://tectonic-typesetting.github.io/) on `PATH`:

```text
tectonic report/main.tex
```

The compiled PDF is written next to the source. The contribution statement is the last page and is outside the report page count.

The presentation is `slides/ENEL525_Project_Presentation.pptx`. To rebuild it, run `npm install pptxgenjs` inside `slides/`, then `node presentation.js`.
