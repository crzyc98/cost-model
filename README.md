# Workforce Cost Model

## Overview

The Workforce Cost Model is an advanced, event-driven simulation framework for projecting workforce demographics, compensation, and retirement plan costs. It provides deterministic and stochastic modeling capabilities for hiring, terminations, promotions, compensation adjustments, and retirement plan participation. The system is designed for organizations, consultants, and plan sponsors to analyze, audit, and forecast workforce and plan outcomes under customizable business rules and scenarios.

Key goals:
- **Accurate multi-year workforce and cost projections**
- **Deterministic and stochastic simulation support**
- **Full audit trail of all workforce events**
- **Modular, analyst-friendly configuration**
- **Comprehensive snapshot and event log management**

---

## Key Features

- **Event-Driven Simulation Engine**: Models every workforce event (hire, termination, promotion, compensation change, plan eligibility, enrollment, etc.) with full auditability.
- **Modular Business Rule Engines**: Each business rule (compensation, hiring, termination, plan rules, etc.) is implemented as a self-contained, swappable engine.
- **Deterministic & Stochastic Logic**: Supports both deterministic and probabilistic (hazard-based) modeling for terminations, hiring, and other events.
- **Snapshot Management**: Rebuilds workforce state at any point in time for scenario analysis and reporting.
- **Structured Logging**: All significant state changes are logged with rich context for traceability and debugging.
- **Scenario Analysis**: Easily compare alternative plan designs or workforce strategies using YAML/CSV configuration files.
- **Analyst-Friendly Configuration**: All assumptions and rules are stored in human-readable YAML and CSV files.

---

## Project Architecture

```
cost-model/
├── cost_model/
│   ├── simulation.py            # Main simulation entry point
│   ├── state/
│   │   ├── snapshot.py         # Snapshot creation and management
│   │   ├── snapshot_update.py  # Snapshot update logic
│   │   ├── event_log.py        # Event logging
│   │   └── schema.py           # Data schema definitions
│   ├── dynamics/
│   │   ├── engine.py           # Core dynamics orchestration
│   │   ├── hiring.py           # Hiring logic
│   │   ├── termination.py      # Termination logic
│   │   └── compensation.py     # Compensation adjustments
│   ├── plan_rules/             # Plan rules (benefits, contributions)
│   ├── projections/
│   │   ├── hazard.py           # Hazard modeling (stochastic logic)
│   │   └── cli.py              # CLI interface
│   ├── data/                   # Data loading and preprocessing
│   ├── ml/                     # Machine learning models (if any)
│   └── reporting/              # Reporting utilities
├── config/                     # YAML/CSV configuration files
├── tests/                      # Unit and integration tests
├── output_dev/                 # Output logs and reports
└── README.md                   # Project documentation
```

---

## Configuration

All simulation rules and assumptions are defined in YAML and CSV files under `config/`.

- **Plan Rules**: Define employer match, auto-enrollment, auto-increase, IRS limits, etc.
- **Workforce Assumptions**: Turnover rates, hiring rates, compensation growth, etc.
- **Scenarios**: Multiple plan designs or workforce strategies can be defined and compared.

Example YAML snippet:
```yaml
plan_rules:
  auto_enroll:
    enabled: true
    window_days: 90
    default_rate: 0.03
    outcome_distribution:
      prob_opt_out: 0.10
      prob_stay_default: 0.70
      prob_opt_down: 0.05
      prob_increase_to_match: 0.10
      prob_increase_high: 0.05
  auto_increase:
    enabled: true
    increase_rate: 0.01
    cap_rate: 0.10
  employer_match:
    tiers:
      - match_rate: 1.0
        cap_deferral_pct: 0.03
      - match_rate: 0.5
        cap_deferral_pct: 0.02
    dollar_cap: 5000
  employer_nec:
    rate: 0.03
  irs_limits:
    2025:
      compensation_limit: 345000
      deferral_limit: 23000
      catchup_limit: 7500
      catchup_eligibility_age: 50

scenarios:
  Baseline:
    name: "Current Plan Design"
  Enhanced_Match:
    name: "Enhanced Employer Match"
    plan_rules:
      employer_match:
        tiers:
          - match_rate: 1.0
            cap_deferral_pct: 0.04
          - match_rate: 0.5
            cap_deferral_pct: 0.02
```
See `docs/config_documentation.md` for a full reference of configuration options.

---

## Data Requirements

Provide an initial census CSV (e.g., `census_data.csv`) with the following columns:

**Required Columns:**
- `employee_id`: Unique identifier
- `birth_date`: YYYY-MM-DD
- `hire_date`: YYYY-MM-DD
- `gross_compensation`: Annual gross compensation

**Optional Columns:**
- `termination_date`: YYYY-MM-DD (if applicable)
- `role`, `plan_year_compensation`, `capped_compensation`, `deferral_percentage`, `employee_contribution`, `employer_match`, `employer_nec`, `tenure_band`, `age_band`

Missing values will be calculated based on configuration and business rules.

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd cost-model
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Review configuration:**
   - Edit YAML/CSV files in `config/` as needed for your scenarios.

---

## Running Simulations

### Command-Line Interface (CLI)

Run a projection with your chosen scenario:
```bash
python -m cost_model.projections.cli --config config/your_config.yaml --scenario Baseline
```

Outputs (snapshots, event logs, reports) will be saved to `output_dev/`.

---

## Testing & Validation

- Run all tests:
  ```bash
  pytest
  ```
- Review logs in `output_dev/projection_logs/` for detailed traceability.
- Validate output data against schema definitions in `cost_model/state/schema.py`.

---

## Contribution & Support

- Follow PEP 8 and project code conventions.
- Add type hints and docstrings to all new code.
- Include unit tests for new functionality.
- See `docs/` for further documentation.
- For questions or support, open an issue or contact the maintainers.

---

© 2025 Workforce Cost Model Project. All rights reserved.
