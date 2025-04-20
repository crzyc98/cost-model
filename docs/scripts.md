# Available Scripts

This document lists all key scripts in the `scripts/` folder and utility scripts, with descriptions and run commands.

---

## scripts/run_retirement_plan_abm.py
Runs the Agent-Based Retirement Plan simulation using Mesa.

Usage:
```
python3 scripts/run_retirement_plan_abm.py \
  --config data/config.yaml \
  --census data/census_data.csv \
  --output output/<prefix>
```

- `--config`: path to scenario config YAML file.
- `--census`: path to initial census CSV.
- `--output`: prefix for output CSVs (model & agent results).

---

## scripts/monte_carlo_experiments.py
Performs Monte Carlo runs by jittering attrition/growth parameters and aggregating KPIs.

Usage:
```
python3 scripts/monte_carlo_experiments.py \
  --config data/config.yaml \
  --census data/census_data.csv \
  --runs 100 \
  --output_dir output/monte_carlo
```

- `--runs`: number of iterations.
- `--output_dir`: directory for run outputs & summary.

---

## scripts/generate_census.py
Generates dummy census data for initial and subsequent years.

Usage:
```
python3 scripts/generate_census.py
```

Outputs year-by-year CSVs in `output/` by default.

---

## scripts/plot_survival.py
Plots Kaplan–Meier survival curves by tenure cohort from historical turnover data.

Usage:
```
python3 scripts/plot_survival.py
```

Requires `data/historical_turnover.csv` in project root.

---

## scripts/report_dynamics.py
Visualizes ABM workforce dynamics: net growth decomposition, cohort counts, and hires vs. terminations.

Usage:
```
python3 scripts/report_dynamics.py \
  --model_csv output/<model_results>.csv \
  --agent_csv output/<agent_results>.csv \
  [--output_dir output]
```

- `--model_csv`: model-level yearly metrics CSV.
- `--agent_csv`: agent-level yearly metrics CSV.
- `--output_dir`: directory for plots (default `output/`).

---

## scripts/fit_hazard_models.py
Fits Kaplan–Meier and Cox hazard models on historical turnover data and exports parameters.

Usage:
```
python3 scripts/fit_hazard_models.py \
  --historical data/historical_turnover.csv \
  --output data/hazard_model_params.yaml
```

---

## scripts/fix_survival_notebook.py
Patches `notebooks/survival_plots.ipynb` to include execution metadata and correct CSV paths.

Usage:
```
python3 scripts/fix_survival_notebook.py
```
