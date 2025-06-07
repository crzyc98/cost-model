# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
```bash
# Run all tests
pytest

# Run quick tests only (< 30 seconds)
pytest -q tests/quick

# Run specific test markers
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m slow          # Slow tests

# Run with coverage
pytest --cov=cost_model
```

### Linting and Code Quality
```bash
# Type checking
mypy cost_model/

# Code style checking
flake8 cost_model/

# Code formatting (if black is configured)
black cost_model/
```

### Main Simulation Commands
```bash
# Run multi-year projection with CLI
python -m cost_model.projections.cli \
  --config config/default.yaml \
  --census data/census_preprocessed.parquet

# Run simulation with specific scenario
python -m cost_model.projections.cli \
  --config config/default.yaml \
  --census data/census_preprocessed.parquet \
  --scenario-name baseline

# Run legacy simulation engine
python scripts/run_simulation.py --config config/default.yaml

# Run Monte Carlo experiments
python scripts/run_monte_carlo.py --config config/default.yaml

# Run projection with specific output directory
python scripts/run_projection.py --config config/default.yaml --output output_dev/
```

### Development Tools
```bash
# Generate census data for testing
python scripts/generate_census.py

# Preprocess census data
python scripts/preprocess_census.py

# Inspect output data
python scripts/inspect_parquet.py output_dev/*/snapshot.parquet
python scripts/view_parquet_contents.py path/to/file.parquet

# Debug configuration loading
python debug_config_loading.py

# Quick sanity check
python scripts/smoke_run.py
```

## Architecture Overview

This is an event-driven workforce cost simulation framework with the following key architectural patterns:

### Core Engine Structure
- **Orchestrator Pattern**: `cost_model/engines/run_one_year/orchestrator/` contains the main simulation orchestrator that coordinates hiring, termination, promotion, and compensation events for each year
- **Modular Engines**: Each workforce dynamic (hiring, termination, promotion, compensation) is implemented as a separate engine in `cost_model/engines/`
- **Event-Driven**: All changes are recorded as events in an audit log (`cost_model/state/event_log.py`)
- **Snapshot-Based State**: Workforce state is captured in periodic snapshots (`cost_model/state/snapshot.py`)

### Configuration System
- **Hierarchical YAML Configuration**: All simulation parameters are defined in YAML files under `config/`
- **Pydantic Validation**: Configuration is validated using Pydantic models in `cost_model/config/models.py`
- **Scenario Support**: Multiple scenarios can be defined in a single config file
- **Parameter Inheritance**: Scenarios inherit from global parameters with selective overrides

### Data Flow
1. **Input**: Census data (Parquet format) + Configuration (YAML)
2. **Initialization**: Initial snapshot created from census (`cost_model/projections/snapshot.py`)
3. **Year Processing**: Each year runs through orchestrator pipeline:
   - Terminations → Hiring → Promotions → Compensation → Plan Rules
4. **Output**: Yearly snapshots + cumulative event log + summary metrics

### Key Modules
- **`cost_model/projections/cli.py`**: Main CLI entry point for multi-year projections
- **`cost_model/dynamics/engine.py`**: Legacy dynamics engine (being replaced by orchestrator)
- **`cost_model/engines/run_one_year/orchestrator/`**: New modular orchestrator system
- **`cost_model/state/`**: State management (snapshots, event logs, schema definitions)
- **`cost_model/plan_rules/`**: Retirement plan business rules (contributions, eligibility, etc.)
- **`cost_model/config/`**: Configuration loading and validation
- **`cost_model/projections/hazard.py`**: Stochastic modeling for terminations and other probabilistic events

### Testing Structure
- Tests are organized by type using pytest markers: `@pytest.mark.unit`, `@pytest.mark.integration`, etc.
- Quick tests (`@pytest.mark.quick`) should run in under 30 seconds for CI
- Test configuration is in `pytest.ini` with comprehensive marker definitions

### Configuration Patterns
- Global parameters in YAML control simulation behavior (start_year, projection_years, random_seed, etc.)
- Workforce dynamics configured via nested dictionaries (compensation rates, turnover models, hiring targets)
- Plan rules defined declaratively (auto-enrollment, employer match tiers, IRS limits)
- Hazard tables for stochastic modeling loaded from CSV/Parquet

### Output Structure
- Results saved to `output_dev/` or user-specified directory
- Parquet format for efficient storage and fast loading
- Yearly snapshots + consolidated final snapshot
- Cumulative event log with full audit trail
- Summary statistics and employment status summaries
- Configuration file copied to output for reproducibility

### Key Constants and Conventions
- Employee ID column: `EMP_ID` (defined in `cost_model/utils/columns.py`)
- Date handling uses pandas Timestamps
- All monetary values stored as floats (no Decimal for performance)
- Random number generation uses numpy Generator with configurable seeds
- Logging configured via `logging_config.py` with structured logging

When modifying this codebase:
- Follow existing patterns for orchestrator-based engines
- Use Pydantic models for new configuration parameters
- Add appropriate pytest markers to new tests
- Update YAML configuration files when adding new parameters
- Maintain event log compatibility when changing state schema
- Use existing column name constants rather than hardcoded strings