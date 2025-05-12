# Retirement Plan Cost Model Runbook

This runbook provides a step-by-step guide for working with the cost-model codebase. Follow these steps when setting up a new environment, making changes to rules, or testing new scenarios.

## 1. Environment Setup

### Initial Setup (one-time per laptop / CI runner)

| Step | Command | Purpose |
|------|---------|--------|
| 1.1 | `python -m venv .venv && source .venv/bin/activate` | Create isolated Python environment |
| 1.2 | `pip install -r requirements.txt -r requirements-dev.txt` | Install all dependencies |
| 1.3 | `pre-commit install` | Set up auto-formatting & linting for commits |
| 1.4 | `export PYTHONPATH=$PWD` (add to shell profile) | Make repo modules resolvable everywhere |

## 2. Quick Sanity Checks

### 2.1 Fast-fail tests (< 30 seconds)

```bash
# Run quick tests to verify basic functionality
pytest -q tests/quick
```

What it tells you: Code imports correctly, key rules run, output folders are created.

### 2.2 Event Log Verification

```bash
# Verify event log is being generated correctly
python scripts/extract_event_log.py \
  --input output_dev/smoke_test/baseline/event_log.parquet \
  --output output_dev/smoke_test/event_analysis/ \
  --format csv
```

Check that events are being properly recorded with timestamps and UUIDs.

## 3. Full Simulation Run

### 3.1 Run Multi-Year Projection

```bash
# Run a full projection with a realistic dataset
python -m cost_model.projections.cli \
  --config config/config.yaml \
  --scenario baseline \
  --census data/census_data.csv \
  --output output/projection_results/ \
  | tee output/projection_logs/projection_cli_run.log
```

Pro-tip: Control logging verbosity with environment variables:
```bash
LOG_LEVEL=INFO python -m cost_model.projections.cli ...
```

### 3.2 Run Agent-Based Model Simulation

```bash
python -m cost_model.abm.run_abm_simulation \
  --config config/config.yaml \
  --scenario baseline \
  --census data/census_data.csv \
  --output output/abm_results/
```

### 3.3 Automated Quality Checks

Run the quality check script to verify key metrics:

```bash
python scripts/sanity_check.py \
  --snapshots output/snapshots \
  --outputs output/projection_results \
  --tolerance 0.01
```

Key metrics to verify (✅/❌):

| Metric | Test |
|--------|------|
| Headcount | Matches growth assumptions ±0.25% |
| Eligible % | Stays within 2pp of prior census |
| Participation % | Same logic |
| Avg deferral % | Non-zero when AE on; monotonic if AI on |
| Total plan cost | Not NaN/inf; monotonic with match increases |
| Comp growth | Row-level median ≈ configured comp increase |

## 4. Analysis and Visualization

### 4.1 Interactive Notebooks

Spin up JupyterLab for interactive analysis:

```bash
jupyter lab --notebook-dir notebooks
```

Key notebooks to explore:

1. **Population Analysis** (`notebooks/01_population_analysis.ipynb`)
   - Loads yearly snapshots from the event log
   - Visualizes demographics (age vs. tenure scatter plots)
   - Shows churn waterfall diagrams
   - Displays headcount growth over time

2. **Financial Analysis** (`notebooks/02_financial_analysis.ipynb`)
   - Loads projection results
   - Creates side-by-side scenario comparison tables
   - Generates heat maps of deferral rates vs. compensation quantiles
   - Tracks new-hire cohort participation over time

3. **Event Analysis** (`notebooks/03_event_analysis.ipynb`)
   - Analyzes the event log to track key events
   - Visualizes event timelines
   - Identifies patterns in enrollment and contribution changes

### 4.2 Log Analysis

Use the log analysis script to quickly identify behavioral patterns and potential issues:

```bash
python scripts/analyze_logs.py \
  --log output/projection_logs/projection_cli_run.log \
  --events "auto_enrollment,eligibility,contribution_change" \
  --output output/log_analysis/
```

This will extract key events from the logs and generate summary statistics and visualizations.

## 5. Debug and Iteration Workflow

### 5.1 Debugging Process

When you encounter issues, follow this debugging workflow:

1. **Population Dynamics Issues** (e.g., headcount growth too low)
   - Extract and analyze the event log to identify patterns
   - Adjust termination rate or hiring parameters in the config
   - Re-run the simulation with modified parameters

2. **Plan Rule Issues** (e.g., unexpected participation rates)
   - Check logs for auto-enrollment or eligibility events
   - Verify rule application in `cost_model/plan_rules/` modules
   - Modify rule implementation and re-run

3. **Financial Calculation Issues** (e.g., contribution errors)
   - Review `cost_model/rules/contributions.py` implementation
   - Check for FutureWarnings related to DataFrame concatenation with empty entries
   - Ensure IRS limits are being correctly applied from config

Each iteration cycle should take less than 2 minutes since you only need to re-run the simulation, not reinstall dependencies.

### 5.2 CI/PR Gate

Add a GitHub Actions workflow for continuous integration:

```yaml
# .github/workflows/regression.yml
name: regression
on: [pull_request]
jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: pytest -q tests/quick
      - run: python -m cost_model.projections.cli \
              --config config/ci.yaml \
              --scenario baseline \
              --census data/sample_census.csv \
              --output output/ci_results
      - run: python scripts/sanity_check.py \
              --snapshots output/ci_results/snapshots \
              --outputs output/ci_results/baseline \
              --tolerance 0.01
```

The workflow should fail if any quality check returns ❌.

## 6. Documentation and Collaboration

### 6.1 Documentation Best Practices

- Keep this runbook updated with any workflow changes
- Link to this runbook from README.md#quick-start
- Add visual documentation (screenshots or GIFs) showing notebooks in action
- Document any new constants or configuration parameters

### 6.2 Collaboration Workflow

1. **Run the pipeline** (use `make run_dev` if you've wrapped the commands in a Makefile)
2. **Review quality checks** - if any ❌ appears, examine the generated reports
3. **Analyze with notebooks** - filter data (e.g., `employee_role == "Staff" & year == 2027`) to identify specific issues
4. **File GitHub issues** - attach relevant data slices as CSV + screenshots, use appropriate labels (e.g., `bug:auto_enrollment`)
5. **Track fixes** - verify that CI passes after fixes are implemented

## 7. Checklist for Quality Assurance

| Status | Item |
|--------|------|
| ☐ | Constants from `utils/columns.py` used consistently throughout codebase |
| ☐ | Quality checks return all ✅ on baseline scenario |
| ☐ | Notebooks render without manual path edits |
| ☐ | CI workflow passes on pull requests |
| ☐ | Event log properly records all employee actions |
| ☐ | FutureWarnings for DataFrame concatenation with empty entries addressed |
| ☐ | IRS limits correctly applied from configuration |

By following this runbook, you'll have a repeatable, efficient workflow to validate the model's behavior and provide clear documentation for other team members.