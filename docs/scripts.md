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

## scripts/run_projection.py
Runs the scenario-driven retirement plan projection simulation and outputs scenario results to Excel.

Usage:
```
python3 scripts/run_projection.py <input_census.csv> --config configs/config.yaml --output <output_prefix> [--raw-output]
```
- `<input_census.csv>`: Path to the initial census CSV file.
- `--config`: Path to the scenario YAML configuration file (see `configs/`).
- `--output`: (optional) Base path/name for output Excel files (scenario name and .xlsx will be appended).
- `--raw-output`: (optional) Save raw agent-level results to Excel.

### Logging
- All logs (level DEBUG and above) are written to `output/projection.log` in the output directory for each run (overwrites each time).
- Warnings and errors (level WARNING and above) also appear in the terminal.
- To change log verbosity or log file location, edit the `logging.basicConfig(...)` call in `scripts/run_projection.py`.

### Developer Note
- Plan rule logic is now centralized in `utils/rules/`.
- Use the new `utils/plan_rules.py` facade for all plan rule imports:
  ```python
  from utils.plan_rules import apply_eligibility, apply_auto_enrollment, ...
  ```
  This ensures all rule logic is modular, testable, and stable for future expansion.

---

## utils/preprocess_census.py
Preprocess utility for census data: adds derived fields (status, eligibility, dates) for simulation.

Usage:
```
python3 utils/preprocess_census.py \
  --input data/census_data.csv \
  --output data/census_preprocessed.csv \
  --config configs/config.yaml \
  --year 2025 \
  [--parquet]
```

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

---

## scripts/sanity_check.py
Sanity check headcounts & total compensation between HR snapshots and plan outputs.

Usage:
```
python3 scripts/sanity_check.py \
  --snapshots snapshots \
  --plan_outputs plan_outputs
```

---

## scripts/run_hr_snapshots.py
Script to export one shared HR snapshot per projection year (Phase I).

Usage:
```
python3 scripts/run_hr_snapshots.py \
  --config configs/config.yaml \
  --census data/census_data.csv \
  --output output/hr_snapshots \
  --seed 42
```

---

## scripts/run_plan_rules.py
Script to apply plan rules for each scenario & year (Phase II).

Usage:
```
python3 scripts/run_plan_rules.py \
  --config configs/config.yaml \
  --snapshots-dir output/hr_snapshots \
  --output-dir output/plan_outputs
```
