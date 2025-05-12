# Project Structure

cost-model/
├── config/                # Scenario/config YAML files
├── data/                 # Input census & reference data
│   └── dev_tiny/         # Example data subdirectory
├── docs/                 # Documentation & design notes
├── cost_model/           # Core package
│   ├── __init__.py       # Makes 'cost_model' a package
│   ├── abm/              # Agent-Based Model implementation
│   │   ├── __init__.py
│   │   ├── agent.py      # EmployeeAgent implementation
│   │   ├── model.py      # RetirementPlanModel implementation
│   │   └── run_abm_simulation.py # ABM simulation entry point
│   ├── config/           # Configuration loading, validation, access
│   │   ├── __init__.py
│   │   ├── loaders.py    # Functions to load/parse YAML
│   │   ├── models.py     # Pydantic models for config structure validation
│   │   └── accessors.py  # Helper functions to get specific config values
│   ├── dynamics/         # Population dynamics (hires, terms)
│   │   ├── __init__.py
│   │   ├── compensation.py # Comp increase logic
│   │   ├── engine.py       # Orchestrates dynamics steps
│   │   ├── hiring.py       # New hire generation logic
│   │   └── termination.py  # Termination logic
│   ├── engines/          # Core business logic engines
│   │   ├── __init__.py
│   │   ├── comp.py       # Compensation engine
│   │   ├── run_one_year.py # Single year simulation
│   │   └── term.py       # Termination engine
│   ├── plan_rules/       # Plan rule implementations
│   │   ├── __init__.py
│   │   ├── auto_enrollment.py
│   │   ├── auto_increase.py
│   │   ├── contribution_increase.py
│   │   └── eligibility.py
│   ├── projections/      # Multi-year projection framework
│   │   ├── __init__.py
│   │   └── cli.py        # CLI entry point
│   ├── rules/            # Business rule implementations
│   │   ├── __init__.py
│   │   ├── contributions.py
│   │   ├── eligibility.py
│   │   └── response.py   # Behavioral response rules
│   ├── state/            # State management
│   │   ├── __init__.py
│   │   ├── event_log.py  # Event logging system
│   │   └── snapshot.py   # Point-in-time snapshots
│   └── utils/            # Utilities and helpers
│       ├── __init__.py
│       ├── columns.py    # Constants for column names
│       ├── constants.py  # Other constants
│       └── date_utils.py # Date helper functions
├── scripts/              # Helper scripts
├── tests/                # Test suite
│   ├── __init__.py
│   └── ...               # Test files
├── output/               # Generated output files & logs
│   ├── output_dev/       # Development output
│   └── projection_logs/  # Projection log files
├── requirements.txt      # Main dependencies
├── requirements-dev.txt  # Development/testing dependencies
└── README.md             # Project overview
