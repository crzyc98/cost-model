# CLAUDE.md

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with the Workforce Cost Model Project codebase.

## Project Overview

This is an event-driven workforce cost simulation framework designed for deterministic and stochastic workforce projections. The system models hiring, termination, promotion, and compensation events over multiple years, producing snapshots and comprehensive audit logs.

## Critical Rules and Guidelines

### Mandatory Requirements
- **Read Everything**: Process all provided documents, instructions, and code line by line without summarizing or skipping details
- **No Guessing**: Never assume, fabricate, or provide placeholder/TODO code. Ask for clarification when uncertain
- **Accuracy First**: All solutions must be complete, correct, tested, and immediately implementable
- **Preserve Intent**: Maintain original logic, structure, and functionality of existing code
- **Zero-Bug Philosophy**: Never introduce errors, bugs, or unresolved issues

### Code Quality Standards
- Use Python type hints consistently
- Implement structured logging (never use print statements)
- Follow proper error handling with meaningful messages
- Maintain clear docstrings and comments
- Write corresponding tests for new functionality

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
# Run multi-year projection with CLI (primary interface)
python -m cost_model.projections.cli \
  --config config/default.yaml \
  --census data/census_preprocessed.parquet

# Run simulation with specific scenario
python -m cost_model.projections.cli \
  --config config/default.yaml \
  --census data/census_preprocessed.parquet \
  --scenario-name baseline

# Run projection with CLI (primary interface) use this example to test always
python scripts/run_multi_year_projection.py --config config/dev_tiny.yaml --census data/census_preprocessed.parquet --debug
```

## Architecture Overview

### Core Engine Structure
- **Orchestrator Pattern**: `cost_model/engines/run_one_year/orchestrator/` coordinates all simulation events for each year
- **Modular Engines**: Separate engines for hiring, termination, promotion, and compensation in `cost_model/engines/`
- **Event-Driven**: All changes recorded as events in audit log (`cost_model/state/event_log.py`)
- **Snapshot-Based State**: Workforce state captured in periodic snapshots (`cost_model/state/snapshot.py`)

### Key Project Structure
```
cost_model/
├── simulation.py              # Core simulation logic
├── state/                     # State management
│   ├── snapshot.py           # Snapshot creation and management
│   ├── snapshot_update.py    # Snapshot update logic
│   ├── event_log.py          # Event logging functionality
│   └── schema.py             # Data schema definitions
├── dynamics/                  # Workforce dynamics
│   ├── engine.py             # Core dynamics orchestration
│   ├── hiring.py             # Hiring logic
│   ├── termination.py        # Termination logic
│   └── compensation.py       # Compensation adjustments
├── engines/run_one_year/     # New orchestrator system
│   └── orchestrator/         # Main simulation orchestrator
├── plan_rules/               # Business rules for benefits
├── projections/              # Projection logic
│   ├── cli.py               # Main CLI entry point
│   └── hazard.py            # Stochastic modeling
├── data/                     # Data loading and preprocessing
├── ml/                       # Machine learning models
└── reporting/                # Reporting utilities
```

### Configuration System
- **Hierarchical YAML Configuration**: All parameters in `config/` directory
- **Pydantic Validation**: Configuration validated using models in `cost_model/config/models.py`
- **Scenario Support**: Multiple scenarios per config file
- **Parameter Inheritance**: Scenarios inherit from global parameters with overrides

### Data Flow
1. **Input**: Census data (Parquet/CSV) + Configuration (YAML)
2. **Initialization**: Initial snapshot from census (`cost_model/projections/snapshot.py`)
3. **Year Processing**: Orchestrator pipeline per year:
   - Terminations → Hiring → Promotions → Compensation → Plan Rules
4. **Output**: Yearly snapshots + event log + summary metrics

## Critical Development Patterns

### Snapshot Management
- Use `snapshot.py` for snapshot creation
- Update snapshots using `snapshot_update.py` functions
- Maintain consistency with `schema.py` definitions
- **Compensation Validation**: All employees must have valid compensation values
- Log compensation issues with appropriate severity levels

### Data Handling Best Practices
- **File Format Support**: Handle both Parquet and CSV census files
- **Column Standardization**: Auto-map common variations (e.g., `employee_ssn` to `employee_id`)
- **DataFrame Merging**: Handle duplicate columns, ensure unique names
- **NA/NaN Handling**: Proper handling of pandas NA/NaN in all operations
- **Error Handling**: Comprehensive logging with graceful fallbacks

### Event Logging
- Log all significant state changes using structured logging
- Include relevant context, especially for compensation events
- Use appropriate log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Log warnings when default values are used for missing data

### Compensation Handling
- All employees must have valid compensation values
- Default to role-based compensation when data is missing
- Handle pandas NA/NaN appropriately in calculations
- Document assumptions and fallbacks

## Testing Structure
- Tests organized by pytest markers: `@pytest.mark.unit`, `@pytest.mark.integration`
- Quick tests (`@pytest.mark.quick`) run in under 30 seconds
- Test configuration in `pytest.ini`
- Maintain comprehensive test coverage

## Output Structure
- Results in `output_dev/` or specified directory
- Parquet format for efficiency
- Yearly snapshots + consolidated final snapshot
- Cumulative event log with full audit trail
- Summary statistics and employment status summaries
- Configuration copied to output for reproducibility

## Key Constants and Conventions
- Employee ID column: `EMP_ID` (from `cost_model/utils/columns.py`)
- Date handling uses pandas Timestamps
- Monetary values as floats (no Decimal for performance)
- Random generation with numpy Generator and configurable seeds
- Logging via `logging_config.py` with structured output

## Current Development Focus

### Immediate Priorities
1. **Termination Logic**: Ensuring deterministic and stochastic termination works for both experienced employees and new hires
2. **Data Validation**: Robust handling of different file formats and column variations
3. **Compensation Validation**: Ensuring all employees have valid compensation throughout simulation
4. **Event Log Consistency**: Maintaining accurate workforce summaries across simulation years

### Common Debugging Areas
- Check logs in `output_dev/projection_logs/`
- Validate termination rates aren't artificially decreasing
- Ensure new hire terminations are properly modeled
- Verify snapshot and event log orchestration accuracy

## When Modifying This Codebase

### Required Practices
- Follow existing orchestrator-based engine patterns
- Use Pydantic models for new configuration parameters
- Add appropriate pytest markers to new tests
- Update YAML configuration files for new parameters
- Maintain event log compatibility with schema changes
- Use existing column constants, not hardcoded strings
- Ensure comprehensive logging for all major operations

### Before Making Changes
1. Read and understand all relevant documentation
2. Review existing code patterns for similar functionality
3. Validate proposed changes maintain data consistency
4. Verify changes don't break existing functionality
5. Add appropriate tests and documentation

### Verification Process
1. Run full test suite with `pytest`
2. Execute smoke tests with `python scripts/smoke_run.py`
3. Validate output structure and data integrity
4. Check logs for warnings or errors
5. Verify configuration loading works correctly

## Performance Considerations
- Optimize for large datasets using vectorized operations
- Be mindful of memory usage with large snapshots
- Use chunking for large operations when needed
- Profile code to identify bottlenecks
- Clear unused variables to manage memory

Remember: This system requires high accuracy and zero tolerance for bugs. Always verify solutions thoroughly before implementation and maintain the integrity of the simulation's deterministic and stochastic modeling capabilities.