# Available Scripts and Commands

This document lists all key commands and scripts in the cost-model package, with descriptions and usage examples.

---

## Core Commands

### Multi-Year Projection CLI

Runs the scenario-driven retirement plan projection simulation using the modular CLI.

Usage:
```bash
python -m cost_model.projections.cli \
  --config config/config.yaml \
  --scenario baseline \
  --census data/census_data.csv \
  --output output/
```

Parameters:
- `--config`: Path to the scenario YAML configuration file.
- `--scenario`: Name of the scenario to run from the config file.
- `--census`: Path to the initial census CSV file.
- `--output`: Directory for output files.
- `--raw-output`: (optional) Save raw agent-level results to Excel.

### Agent-Based Model Simulation

Runs the Agent-Based Retirement Plan simulation using Mesa.

Usage:
```bash
python -m cost_model.abm.run_abm_simulation \
  --config config/config.yaml \
  --scenario baseline \
  --census data/census_data.csv \
  --output output/abm_results/
```

Parameters:
- `--config`: Path to scenario config YAML file.
- `--scenario`: Name of the scenario to run.
- `--census`: Path to initial census CSV.
- `--output`: Directory for output CSVs (model & agent results).

---

## Helper Scripts

### scripts/generate_census.py

Generates dummy census data for initial and subsequent years.

Usage:
```bash
python scripts/generate_census.py \
  --output data/dev_tiny/ \
  --employees 100 \
  --start_year 2025
```

Parameters:
- `--output`: Directory for year-by-year CSVs.
- `--employees`: Number of employees to generate.
- `--start_year`: Starting year for the census data.

### scripts/monte_carlo_experiments.py

Performs Monte Carlo runs by jittering attrition/growth parameters and aggregating KPIs.

Usage:
```bash
python scripts/monte_carlo_experiments.py \
  --config config/config.yaml \
  --census data/census_data.csv \
  --runs 100 \
  --output_dir output/monte_carlo
```

Parameters:
- `--config`: Path to scenario config YAML file.
- `--census`: Path to initial census CSV.
- `--runs`: Number of iterations.
- `--output_dir`: Directory for run outputs & summary.

### scripts/plot_survival.py

Plots Kaplan–Meier survival curves by tenure cohort from historical turnover data.

Usage:
```bash
python scripts/plot_survival.py \
  --data data/historical_turnover.csv \
  --output output/survival_curves/
```

Parameters:
- `--data`: Path to historical turnover data CSV.
- `--output`: Directory for output plots.

### scripts/report_dynamics.py

Visualizes workforce dynamics: net growth decomposition, cohort counts, and hires vs. terminations.

Usage:
```bash
python scripts/report_dynamics.py \
  --model_csv output/abm_results/model_results.csv \
  --agent_csv output/abm_results/agent_results.csv \
  --output_dir output/dynamics_reports/
```

Parameters:
- `--model_csv`: Model-level yearly metrics CSV.
- `--agent_csv`: Agent-level yearly metrics CSV.
- `--output_dir`: Directory for plots.

---

## Utility Scripts

### scripts/preprocess_census.py

Preprocess utility for census data: adds derived fields (status, eligibility, dates) for simulation.

Usage:
```bash
python scripts/preprocess_census.py \
  --input data/census_data.csv \
  --output data/census_preprocessed.parquet \
  --config config/config.yaml \
  --year 2025 \
  --parquet
```

Parameters:
- `--input`: Path to raw census CSV.
- `--output`: Path for preprocessed output file.
- `--config`: Path to config YAML file.
- `--year`: Reference year for preprocessing.
- `--parquet`: Save output as Parquet (default is CSV).

### scripts/sanity_check.py

Sanity check headcounts & total compensation between snapshots and outputs.

Usage:
```bash
python scripts/sanity_check.py \
  --snapshots output/snapshots \
  --outputs output/projection_results \
  --tolerance 0.01
```

Parameters:
- `--snapshots`: Directory containing snapshot files.
- `--outputs`: Directory containing projection output files.
- `--tolerance`: Acceptable difference tolerance (default: 0.01).

---

## Event Log and Snapshot Utilities

### scripts/extract_event_log.py

Extracts and formats the event log for analysis.

Usage:
```bash
python scripts/extract_event_log.py \
  --input output/projection_results/event_log.parquet \
  --output output/event_analysis/ \
  --format csv
```

Parameters:
- `--input`: Path to event log Parquet file.
- `--output`: Directory for output files.
- `--format`: Output format (csv, excel, parquet).

### scripts/rebuild_snapshot.py

Rebuilds a point-in-time snapshot from the event log.

Usage:
```bash
python scripts/rebuild_snapshot.py \
  --event_log output/projection_results/event_log.parquet \
  --date 2026-06-30 \
  --output output/snapshots/
```

Parameters:
- `--event_log`: Path to event log Parquet file.
- `--date`: Date for the snapshot (YYYY-MM-DD).
- `--output`: Directory for output snapshot file.
