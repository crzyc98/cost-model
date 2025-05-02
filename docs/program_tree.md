# Cost Model Project Structure

## Overview
Retirement plan cost simulation system with:
- Population dynamics modeling
- Plan rules engine
- Configuration-driven scenarios
- Statistical modeling components

## Root Directory
```
.
├── config/             # Configuration files
├── cost_model/         # Core simulation package
├── data/               # Input data files
├── docs/               # Documentation
├── output/             # Simulation outputs (generated)
├── results/            # Results storage (generated)
├── scripts/            # Main executable scripts
├── tests/              # Test suite
├── README.md           # Project overview
├── requirements.txt    # Production dependencies
└── requirements-dev.txt # Development dependencies
```

## Configuration (config/)
```
config/
├── config.yaml         # Main simulation configuration
├── defaults.yaml       # Default parameter values
├── dev_tiny.yaml       # Development/test configuration
└── mc_packages.json    # Monte Carlo experiment packages
```

## Core Package (cost_model/)
```
cost_model/
├── config/             # Configuration models and loaders
├── data/               # Data readers/writers
├── dynamics/           # Population dynamics engine
├── ml/                 # Machine learning components
├── reporting/          # Results analysis and metrics
├── rules/              # Plan rules implementation
├── utils/              # Shared utilities
├── simulation.py       # Main simulation orchestrator
└── __init__.py         # Package definition
```

### Key Components

#### Rules Engine (cost_model/rules/)
- `engine.py`: Main rule application logic
- `contributions.py`: Contribution calculations
- `auto_enrollment.py`: Auto-enrollment implementation
- `eligibility.py`: Eligibility determination

#### Dynamics Engine (cost_model/dynamics/)
- Population modeling (hiring, termination, compensation)
- Headcount projections
- Statistical survival models

#### Data Handling (cost_model/data/)
- Census data readers (`readers.py`)
- Result writers (`writers.py`)
- Data preprocessing utilities

## Scripts (scripts/)
```
scripts/
├── run_simulation.py       # Main simulation runner
├── generate_census.py      # Census data generation
├── run_monte_carlo.py      # Monte Carlo experiments
├── train_termination_model.py # ML model training
├── plot_survival.py        # Survival analysis visualization
└── ... (20+ utility scripts)
```

## Data Files (data/)
```
data/
├── census_data.csv         # Employee census data
├── historical_turnover.csv # Turnover statistics
├── hazard_model_params.yaml # Survival model parameters
└── test_census/            # Test datasets
```

## Simulation Flow
1. Configuration loaded from `config/*.yaml`
2. Initial population loaded from `data/census_data.csv`
3. Yearly dynamics applied (terminations, hires, compensation)
4. Plan rules executed (eligibility, contributions)
5. Results saved to `results/<scenario>/`

## Key Dependencies
- Python 3.9+
- Pandas/Numpy (data processing)
- Mesa (agent-based modeling)
- Pydantic (configuration validation)
- Scikit-learn (ML components)
