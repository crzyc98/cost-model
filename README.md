# Retirement Plan Cost Model (Projection Tool)

**What is this tool?**  
Imagine a system that captures every single action in your workforce—from the moment someone's hired, to every pay bump, every enrollment in benefits, every voluntary or automatic plan increase—and stores it all in an immutable "event log." Now picture being able to replay those events, at any point in time, to see exactly how your headcount, payroll costs, and retirement‐plan participation evolved year by year.

That's what we've built: an event-driven simulation engine that enables organizations, consultants, and plan sponsors to simulate and project the financial and demographic outcomes of retirement plans under customizable rules and assumptions. It supports scenario analysis for plan design, regulatory compliance, and cost forecasting—generating both high-level summaries and detailed agent-level (employee-level) outputs over multiple years.

## Key Features

### 1. Enterprise‐Grade Audit Trail
Every hire, termination, compensation change, plan‐eligibility update, and enrollment decision gets stamped with a UUID and timestamp. You get a complete, tamper-proof history of your entire population.

### 2. Highly Modular "Engines"
Each business rule—compensation increases, voluntary enrollment, auto-escalation of deferrals, proactive contribution changes, you name it—is its own self-contained engine. Want to tweak your auto-enroll parameters? Swap in a new YAML file, rerun, and instantly see the impact—no code changes required.

### 3. On-Demand Snapshots & What-If Scenarios
Need to know your active headcount as of June 30, 2027? Rebuild the state in seconds from the event log. Want to model a more aggressive match policy? Flip a number in the config, rerun a five-year forecast, and compare scenarios in under two seconds.

### 4. Analyst-Friendly Configuration
All your core assumptions—turnover rates, raise percentages, enrollment windows, match tiers—live in human-readable CSVs and YAML files. Finance and HR partners can tune parameters without ever touching Python.

### 5. End-to-End Transparency
From raw events to final summary metrics (headcount, participation rates, contribution dollars), every step is fully tested, documented, and reproducible. Your auditors, regulators, or board members will love the clarity.

## Key Use Cases

- **Plan Design Evaluation:** Compare the cost and participation impact of different plan rules (eligibility, auto-enrollment, employer match/core, etc.).
- **Scenario Analysis:** Model workforce changes, compensation growth, and turnover using both rule-based and machine learning approaches.
- **Regulatory & Compliance Forecasting:** Assess plan compliance with IRS limits and test the impact of regulatory changes.
- **Detailed Analytics:** Output both summary metrics and granular, agent-level results for further analysis.

## Running Simulations

### Production Simulation (Full Dataset)
```bash
python -m cost_model.projections.cli \
  --config config/config.yaml \
  --scenario baseline \
  --census data/census_data.csv \
  --output output/
```

### Development Simulation (Small Test Dataset)
```bash
python -m cost_model.projections.cli \
  --config config/dev_tiny.yaml \
  --scenario baseline \
  --census data/dev_tiny/census_2024.csv \
  --output output_dev/
```

### Agent-Based Model Simulation
```bash
python -m cost_model.abm.run_abm_simulation \
  --config config/dev_tiny.yaml \
  --scenario baseline \
  --census data/dev_tiny/census_2024.csv \
  --output output_dev/abm_results/
```

---

## Project Directory Structure

```
cost-model/
├── config/                # Scenario/config YAML files
├── data/                 # Input census & reference data
├── docs/                 # Documentation & design notes
├── cost_model/           # Core package
│   ├── abm/              # Agent-Based Model implementation
│   │   ├── agent.py      # EmployeeAgent implementation
│   │   └── model.py      # RetirementPlanModel implementation
│   ├── dynamics/         # Population dynamics (hires, terms)
│   ├── engines/          # Core business logic engines
│   ├── plan_rules/       # Plan rule implementations
│   ├── projections/      # Multi-year projection framework
│   │   └── cli.py        # CLI entry point
│   ├── rules/            # Business rule implementations
│   ├── state/            # State management (event log, snapshots)
│   │   ├── event_log.py  # Event logging system
│   │   └── snapshot.py   # Point-in-time snapshots
│   └── utils/            # Utilities and helpers
├── scripts/              # Helper scripts
├── tests/                # Test suite
├── output/               # Generated output files & logs
│   └── output_dev/       # Development output
├── requirements.txt
└── README.md
```

---

## How It Works

In short, we've moved from a brittle, hard-to-change monolith to a nimble, auditable, and extensible event-driven simulation engine. It's like upgrading from a black-box spreadsheet to a flight-recorder-driven "time machine" for your workforce—and everyone from analysts to executives can explore "what happened," "why it happened," and "what could happen" with confidence and speed.

### Event-Driven Architecture
The system is built around an immutable event log that records every significant change in the employee population:

1. **Event Generation**: Events are generated for hires, terminations, compensation changes, eligibility changes, enrollment decisions, etc.
2. **Event Processing**: Events are processed in chronological order to build the state of the system at any point in time.
3. **State Reconstruction**: The system can rebuild the state from the event log for any point in time, enabling precise historical analysis.

### Two Simulation Approaches

#### 1. Traditional Projection Framework
The projection framework processes employees in cohorts, applying rules and generating events for each year in the projection:

- **Engines**: Modular components handle specific business logic (compensation, termination, eligibility, etc.)
- **Rules**: Business rules determine eligibility, auto-enrollment, contributions, etc.
- **Dynamics**: Population dynamics handle hiring and termination patterns
- **Snapshots**: Point-in-time views of the employee population are generated for analysis

#### 2. Agent-Based Model (ABM)
The ABM approach models each employee as an autonomous agent making decisions based on their characteristics and the environment:

- **EmployeeAgent**: Individual employees with state and decision-making logic
- **RetirementPlanModel**: Environment that manages agents and applies global rules
- **Behavioral Modeling**: Probabilistic decision-making based on employee characteristics
- **Emergent Patterns**: System-level patterns emerge from individual agent behaviors

### Output and Analysis
- **Yearly Snapshots**: Complete population state at the end of each projection year
- **Summary Metrics**: Aggregated metrics on participation, contributions, and costs
- **Scenario Comparison**: Side-by-side comparison of different plan design scenarios
- **Detailed Agent Data**: Individual-level data for deeper analysis  

---

## Technical Features

- **Event-Driven Architecture**: Immutable event log for complete audit trail and state reconstruction
- **Dual Simulation Approaches**: Traditional projection framework and agent-based modeling
- **Data Validation**: Comprehensive validation of input data and simulation state. [View data validation documentation](docs/cost_model/10_data_validation_issues.md)
- **Configurable Scenarios**: Via YAML files (start year, projection length, comp increases, hire/term rates)
- **Modular Plan Rules**: Eligibility, auto-enrollment (AE), auto-increase (AI), employer match/NEC formulas
- **Population Dynamics**: Sophisticated models for hires and terminations (rule-based or ML-based)
- **Financial Precision**: Accurate calculations with `Decimal`, pandas, and NumPy
- **Comprehensive Output**: Summary Excel per scenario, combined summaries, and raw agent-level Excel
- **Extensive Testing**: Comprehensive test suite ensures reliability and correctness

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
  typing_extensions>=4.7.1
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

---

## Configuration

Edit `config/config.yaml` to define scenarios. The configuration is structured into three main sections:

1. **global_parameters**: General simulation settings
2. **plan_rules**: Retirement plan specific rules and parameters
3. **scenarios**: Defined scenarios for running simulations

Key fields:
```yaml
global_parameters:
  start_year: 2025
  projection_years: 5
  random_seed: 42
  annual_compensation_increase_rate: 0.03
  annual_termination_rate: 0.15
  new_hire_termination_rate: 0.25
  new_hire_rate: 0.17
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

plan_rules:
  eligibility:
    min_age: 21
    min_service_months: 0
  auto_enrollment:
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

See `docs/config_documentation.md` for a complete reference of all configuration options.

---

## Data

Provide initial census CSV (e.g., `census_data.csv`) with the following columns:

### Required Columns
- `employee_id`: Unique identifier for each employee
- `birth_date`: Employee date of birth (YYYY-MM-DD)
- `hire_date`: Employee hire date (YYYY-MM-DD)
- `termination_date`: Employee termination date if applicable (YYYY-MM-DD)
- `gross_compensation`: Annual gross compensation

### Optional Columns (will be calculated if not provided)
- `role`: Employee role/job title
- `plan_year_compensation`: Compensation for the plan year
- `capped_compensation`: Compensation capped at IRS limit
- `deferral_percentage`: Current employee deferral percentage
- `employee_contribution`: Employee contribution amount
- `employer_match`: Employer match amount
- `employer_nec`: Employer non-elective contribution amount
- `tenure_band`: Tenure category (e.g., "0-1 years", "1-3 years")
- `age_band`: Age category (e.g., "21-30", "31-40")

The system will calculate missing values based on the configuration and business rules.
