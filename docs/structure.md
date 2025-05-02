cost-model/
├── configs/              # Scenario/config YAML files (Good, keep as is)
├── data/                 # Input census & reference data (Good, keep as is)
│   └── dev_tiny/         # Example data subdirectory
├── docs/                 # Documentation & design notes (Good, keep as is)
├── output/               # Generated output files & logs (Good, keep as is)
│   └── hr_snapshots-dev/ # Example output subdirectory
├── scripts/              # Thin CLI wrappers (Good, keep but simplify scripts)
│   ├── generate_census.py   # Calls cost_model.utils.census_generation_helpers
│   └── run_simulation.py    # NEW: Orchestrates simulation via cost_model.simulation
├── cost_model/           # <--- Core library/package
│   ├── __init__.py       # Makes 'cost_model' a package
│   ├── config/           # Configuration loading, validation, access
│   │   ├── __init__.py
│   │   ├── loaders.py    # Functions to load/parse YAML
│   │   ├── models.py     # Pydantic models for config structure validation
│   │   └── accessors.py  # Helper functions to get specific config values
│   ├── data/             # Data input/output handling
│   │   ├── __init__.py
│   │   ├── readers.py    # Functions to read census CSV/Parquet
│   │   ├── writers.py    # Functions to write snapshots, results, summaries
│   │   └── validation.py # Optional: Data validation logic
│   ├── dynamics/         # Population dynamics & HR simulation ("Phase 1")
│   │   ├── __init__.py
│   │   ├── compensation.py # Comp increase logic (_apply_comp_bump)
│   │   ├── hiring.py       # New hire generation logic (from generate_new_hires)
│   │   ├── termination.py  # Termination logic (_apply_turnover, sampling)
│   │   └── engine.py       # Orchestrates dynamics steps (replaces project_hr)
│   ├── ml/               # OPTIONAL: Machine Learning components
│   │   ├── __init__.py
│   │   └── predictors.py # Loading models, making predictions
│   ├── rules/            # Plan rule implementations & application ("Phase 2")
│   │   ├── __init__.py
│   │   ├── eligibility.py
│   │   ├── auto_enrollment.py
│   │   ├── auto_increase.py
│   │   ├── contributions.py
│   │   ├── validators.py # Pydantic models for rule configs
│   │   └── engine.py     # Applies rules (replaces apply_plan_rules)
│   ├── reporting/        # Generating output summaries and metrics
│   │   ├── __init__.py
│   │   └── metrics.py    # Functions to calculate/aggregate results
│   ├── utils/            # Generic utilities & specific helpers
│   │   ├── __init__.py
│   │   ├── census_generation_helpers.py # For generate_census.py script
│   │   ├── columns.py    # Constants for column names (or define in relevant modules)
│   │   ├── constants.py  # Other constants (e.g., ACTIVE_STATUS)
│   │   └── date_utils.py # Generic date helpers
│   └── simulation.py     # High-level simulation orchestrator (ties everything together)
├── tests/                # <--- NEW: Unit/integration tests
│   ├── __init__.py
│   └── ...               # Mirror structure of cost_model/
├── requirements.txt      # Main dependencies
├── requirements-dev.txt  # Development/testing dependencies
└── README.md             # Project overview
