# Retirement Plan Cost Model (Projection Tool)

A scenario-driven projection engine for retirement plan outcomes. Customize plan rules, demographic assumptions, and IRS limits, then generate both summary and detailed agent‑level outputs over multiple years.

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
python sandbox/run_projection.py <census_csv> --output <base_name> [--raw-output]
```

- `<census_csv>`: path to initial census file.
- `--output`: base path/name for Excel outputs (appends scenario and extension).
- `--raw-output`: include detailed agent‑level results in `<base>_<scenario>_raw.xlsx`.

Examples:
```bash
python sandbox/run_projection.py census_data.csv --output projection_results --raw-output
```

After running, you’ll have:
- `projection_results_Baseline.xlsx`, `projection_results_AIP_New_Hires.xlsx`, etc.
- `projection_results_all_summaries.xlsx`
- `projection_results_<scenario>_raw.xlsx` (per‑scenario raw data).

## Project Structure

```
cost-model/
├── sandbox/
│   ├── run_projection.py       # Main projection runner
│   └── projection_utils.py     # Core simulation logic
├── utils/
│   ├── date_utils.py           # Age and tenure functions
│   └── plan_rules.py           # Eligibility, AE/AI, contributions
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
