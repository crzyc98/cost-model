# Retirement Plan Cost Model (Projection Tool)

Retirement Plan Cost Model (Projection Tool)
What is this tool?
This engine enables organizations, consultants, and plan sponsors to simulate and project the financial and demographic outcomes of retirement plans under customizable rules and assumptions. It supports scenario analysis for plan design, regulatory compliance, and cost forecasting—generating both high-level summaries and detailed agent-level (employee-level) outputs over multiple years.

Key Use Cases
Plan Design Evaluation: Compare the cost and participation impact of different plan rules (eligibility, auto-enrollment, employer match/core, etc.).
Scenario Analysis: Model workforce changes, compensation growth, and turnover using both rule-based and machine learning approaches.
Regulatory & Compliance Forecasting: Assess plan compliance with IRS limits and test the impact of regulatory changes.
Detailed Analytics: Output both summary metrics and granular, agent-level results for further analysis.

This tool is a projection engine for retirement plan outcomes. Customize plan rules, demographic assumptions, and IRS limits, then generate both summary and detailed agent-level outputs over multiple years.

## 🚀 Phase I & Phase II Overview

### Phase I: **HR Snapshots**  
Generate or load “raw” census files (one per plan year) that capture headcount and compensation at year-end.  
- **Scripts**:  
  - `scripts/generate_census.py` (dummy/historical census generator)  
  - `scripts/run_hr_snapshots.py` (reads CSVs, writes Parquet “snapshots/”)  
- **Output**:  
  - Per-scenario folder under `hr_snapshots/`  
  - One Parquet file per year, with **all** employees (active & terminated) at year-end  

**Purpose**: feed stable data into Phase II, and allow sanity checks on headcount & total compensation.

---

### Phase II: **Plan Rules & Projection**  
Take the Phase I snapshots (active population) and apply plan design rules + population dynamics to project out future years.  
- **Scripts**:  
  - `scripts/run_plan_rules.py` (applies eligibility, AE, AI, contributions)  
  - `scripts/run_projection.py` (config-driven multi-scenario projection)  
- **Core Logic**:  
  1. **Eligibility**: age & service entry rules  
  2. **Auto-Enrollment (AE)** & **Auto-Increase (AI)**  
  3. **Contributions**: employee deferral, employer match/NEC  
  4. **Population Dynamics**: hires, terminations (rule- or ML-based)  
  5. **Cost Proration**: prorate comp for mid-year hires/terms  

- **Output**:  
  - Per-scenario **summary** CSV/Excel: yearly headcount, contributions, plan cost metrics  
  - Optional **raw agent-level** Excel: one sheet per year + combined sheet  


## Project Directory Structure

```
cost-model/
├── cost_model/           # Public Python package façade (stable API)
├── engine/               # Core config loader & orchestration logic
├── loaders/              # Data loading utilities (YAML/JSON/CSV)
├── configs/              # Scenario/config YAML files
├── defaults/             # Engine default settings (defaults.yaml)
├── model/                # Core projection & business logic
├── agents/               # ABM agents (Mesa-based EmployeeAgent)
├── utils/                # Helpers, plan_rules façade, ML utilities
│   ├── rules/             # Rule implementations
│   │   ├── eligibility.py  # Eligibility logic
│   │   ├── auto_enrollment.py  # Auto-enrollment logic
│   │   ├── auto_increase.py    # Auto-increase logic
│   │   ├── formula_parsers.py  # Match/NEC parsing helpers
│   │   ├── contributions.py    # Contributions and limit engine
│   │   └── response.py         # Plan-change deferral response
│   └── plan_rules.py        # Facade to rule modules
├── data/                 # Input census & reference data
├── docs/                 # Documentation & design notes
├── notebooks/            # Jupyter analysis notebooks
├── scripts/              # CLI entry points & batch scripts
│   ├── run_projection.py
│   ├── generate_census.py
│   └── ...
├── tests/                # Unit & integration tests
├── output/               # Generated output files & logs
├── requirements.txt      # Python dependencies
└── README.md             # Project overview
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

- Configurable scenarios via `configs/config.yaml` (start year, projection length, comp increases, hire/term rates).
- Plan rules: eligibility, auto‑enrollment (AE), auto‑increase (AI), employer match/NEC formulas.
- Population dynamics: hires and terminations (rule‑based or ML‑based).
- Precise financials with `Decimal`, pandas and NumPy.
- Outputs:
  - Summary Excel per scenario (yearly metrics).
  - Combined summary across scenarios (`*_all_summaries.xlsx`).
  - Raw agent‑level Excel (`--raw-output`): each year sheet + `Combined_Raw` sheet with **Year** column.

## Requirements

- Python 3.8+
- Core dependencies:
  ```
  pandas>=2.2.2
  numpy>=1.26.4
  scipy>=1.13.0
  matplotlib>=3.8.4
  seaborn>=0.13.2
  lifelines>=0.28.0
  openpyxl>=3.1.2
  joblib>=1.4.2
  scikit-learn>=1.4.2
  lightgbm>=4.3.0
  PyYAML>=6.0.1
  pydantic>=1.10.2
  mesa>=2.2.0
  ```

- Development dependencies:
  ```
  pytest>=8.1.1
  mypy>=1.15.0
  flake8>=7.2.0
  structlog>=22.3.0
  loguru>=0.7.2
  tqdm>=4.66.2
  jupyterlab>=4.1.5
  pandas-stubs>=2.2.0
  numpy-stubs>=1.26.0
  ```

Install via:
```bash
pip install -r requirements.txt
# For development:
pip install -r requirements-dev.txt
```

## Configuration

Edit `configs/config.yaml` to define scenarios. Key fields:

```yaml
global_parameters:
  start_year: 2025
  projection_years: 5
  random_seed: 12345
  annual_compensation_increase_rate: 0.03
  annual_termination_rate: 0.13
  new_hire_termination_rate: 0.20
  maintain_headcount: false
  
  role_distribution:
    Staff: 0.8
    Manager: 0.1
    Executive: 0.1
    
  new_hire_compensation_params:
    comp_base_salary: 55000
    comp_std: 10000
    comp_increase_per_age_year: 500
    comp_increase_per_tenure_year: 1000
    comp_min_salary: 40000
    
  role_compensation_params:
    Staff:
      comp_base_salary: 55000
      comp_std: 10000
    Manager:
      comp_base_salary: 85000
      comp_std: 15000
    Executive:
      comp_base_salary: 150000
      comp_std: 20000

scenarios:
  Baseline:
    plan_rules:
      eligibility:
        min_age: 21
        min_service_months: 0
      auto_enrollment:
        enabled: true
        default_rate: 0.02
      employer_match_formula: "50% up to 6%"
      employer_core_formula: "0%"
```

## Data

Provide initial census CSV (e.g., `census_data.csv`) with columns:
- `employee_id`
- `role`  
- `birth_date`
- `hire_date`
- `termination_date` (optional)
- `gross_compensation`
- `plan_year_compensation`
- `capped_compensation`
- `deferral_percentage`
- `employee_contribution`
- `employer_match`
- `employer_core_contribution`
