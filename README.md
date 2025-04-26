# Retirement Plan Cost Model (Projection Tool)

## Project Directory Structure

```
cost-model/
├── agents/                  # Agent-based model logic (Mesa, agent classes, behaviors)
├── configs/                 # YAML scenario/config files
├── data/                    # Input census data, reference data
├── docs/                    # Documentation and design notes
├── engine/                  # Core config loader, orchestration logic
├── loaders/                 # Data loading utilities
├── model/                   # Core model logic
├── notebooks/               # Jupyter/analysis notebooks
├── output/                  # Output artifacts and results
├── scripts/                 # Main entry points and batch scripts
│   ├── run_projection.py
│   ├── generate_census.py
│   ├── train_termination_model.py
│   └── ...
├── tests/                   # Unit and integration tests
├── utils/                   # Plan rules, helpers, ML logic, utilities
│   ├── rules/               # Modular plan rule implementations
│   │   ├── eligibility.py
│   │   ├── auto_enrollment.py
│   │   ├── auto_increase.py
│   │   ├── contributions.py
│   │   ├── response.py
│   │   └── formula_parsers.py
│   ├── plan_rules.py        # Facade: stable import for all plan rules
│   ├── ml_logic.py
│   ├── projection_utils.py
│   ├── sandbox_utils.py
│   └── ...
├── requirements.txt         # Python dependencies
└── README.md
```


A scenario-driven projection engine for retirement plan outcomes. Customize plan rules, demographic assumptions, and IRS limits, then generate both summary and detailed agent‑level outputs over multiple years.

## Onboarding: Plan Rules and Utilities

### Plan Rule Usage Example

All plan rule logic is modularized in `utils/rules/` and can be accessed via the stable facade `utils/plan_rules.py`.

**Importing and using plan rules:**
```python
from utils.plan_rules import apply_eligibility, apply_auto_enrollment, apply_auto_increase, apply_contributions

df = ...  # your census DataFrame
scenario_config = ...  # scenario dict from YAML
simulation_year_end_date = ...  # datetime object

# Apply eligibility rule
eligible_df = apply_eligibility(df, scenario_config['plan_rules'], simulation_year_end_date)

# Apply auto-enrollment
enrolled_df = apply_auto_enrollment(eligible_df, scenario_config['plan_rules'], simulation_year_end_date)

# Apply auto-increase
increased_df = apply_auto_increase(enrolled_df, scenario_config['plan_rules'], simulation_year_end_date)

# Calculate contributions
contrib_df = apply_contributions(increased_df, scenario_config, year, year_start, year_end)
```

### Extending Plan Rules
- To add or modify a rule, edit or add a module in `utils/rules/` (e.g., `eligibility.py`, `contributions.py`).
- To expose a new rule for project-wide use, add it to `utils/plan_rules.py` for stable imports.
- All rule modules are designed for easy testing and extension.

---

## Features

- Configurable scenarios via `data/config.yaml` (start year, projection length, comp increases, hire/term rates).
- Plan rules: eligibility, auto‑enrollment (AE), auto‑increase (AI), employer match/NEC formulas.
- Population dynamics: hires and terminations (rule‑based or ML‑based).
- Precise financials with `Decimal`, pandas and NumPy.
- Outputs:
  - Summary Excel per scenario (yearly metrics).
  - Combined summary across scenarios (`*_all_summaries.xlsx`).
  - Raw agent‑level Excel (`--raw-output`): each year sheet + `Combined_Raw` sheet with **Year** column.

## Requirements

- Python 3.8+
- pandas
- numpy
- scipy
- joblib

Install via:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `data/config.yaml` to define scenarios. Key fields:

```yaml
scenarios:
  - scenario_name: Baseline
    start_year: 2025
    projection_years: 5
    comp_increase_rate: 0.03
    hire_rate: 0.10
    termination_rate: 0.08
    maintain_headcount: false
    plan_rules:
      eligibility:
        min_age: 21
        min_service_months: 0
      auto_enrollment:
        enabled: true
        default_rate: 0.02
        window_days: 35
      auto_increase:
        enabled: false
      employer_match_formula: "50% up to 6%"
      employer_non_elective_formula: "0%"
      match_change_response:
        enabled: true
        increase_probability: 0.25  # Probability of bump to optimal
        increase_target: optimal    # Only 'optimal' supported currently
    irs_limits:
      ...
    use_ml_turnover: true
    ml_model_path: termination_model_pipeline.joblib
    model_features_path: termination_model_features.joblib
```

Additional settings include new‑hire generation parameters, veteran attrition toggles, and hazard model options under `hazard_model_params`.

## Data

Provide initial census CSV (e.g., `census_data.csv`) with columns:
- `ssn` (or employee_id)
- `birth_date`
- `hire_date`
- `gross_compensation`
- `pre_tax_deferral_percentage` (optional)
- any other attributes (role, salary, etc.)

## Usage

```bash
python scripts/run_projection.py <census_csv> --output <base_name> [--raw-output]
```

- `<census_csv>`: path to initial census file.
- `--output`: base path/name for Excel outputs (appends scenario and extension).
- `--raw-output`: include detailed agent‑level results in `<base>_<scenario>_raw.xlsx`.

Examples:
```bash
python scripts/run_projection.py census_data.csv --output projection_results --raw-output
```

After running, you’ll have:
- `projection_results_Baseline.xlsx`, `projection_results_AIP_New_Hires.xlsx`, etc.
- `projection_results_all_summaries.xlsx`
- `projection_results_<scenario>_raw.xlsx` (per‑scenario raw data).

│   ├── plan_rules.py           # Façade to rule modules
│   └── rules/                  # Rule implementations
│       ├── eligibility.py      # Eligibility logic
│       ├── auto_enrollment.py  # Auto-enrollment logic
│       ├── auto_increase.py    # Auto-increase logic
│       ├── formula_parsers.py  # Match/NEC parsing helpers
│       ├── contributions.py    # Contributions and limit engine
│       └── response.py         # Plan-change deferral response
├── data/
│   └── config.yaml             # Scenario configurations and IRS limits
├── docs/                       # Documentation and debugging notes
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview and instructions
```

## Contributing

Feedback and pull requests are welcome. Please open issues for major features or bug reports.

## License

This project is released under the MIT License.
