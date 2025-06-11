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
- **Projection Framework**: `cost_model/projections/` orchestrates multi-year simulations
- **Modular Engines**: Specialized engines in `cost_model/engines/` for each workforce event type
  - **Enhanced Termination Engine**: Intelligent fallback logic for missing hazard combinations
  - **Hiring Engine**: New employee generation with configurable parameters
  - **Compensation Engine**: Merit increases, COLA adjustments, and promotion raises
  - **Promotion Engine**: Career progression with age sensitivity
- **Dynamic Hazard Tables**: Runtime generation based on configuration parameters
- **Event-Driven Architecture**: All changes recorded as structured events
- **Snapshot-Based State**: Point-in-time workforce state with comprehensive tracking

### Key Project Structure
```
cost_model/
├── simulation.py              # Core simulation orchestration
├── config/                    # Configuration management
│   ├── loaders.py            # YAML configuration loading
│   ├── models.py             # Pydantic validation models
│   ├── accessors.py          # Configuration access helpers
│   ├── params.py             # Parameter management
│   └── plan_rules.py         # Plan rule configuration
├── data/                      # Data I/O operations
│   ├── readers.py            # Data loading (Parquet/CSV)
│   └── writers.py            # Data saving with format detection
├── engines/                   # Core simulation engines
│   ├── term.py               # Enhanced termination engine
│   ├── hire.py               # Hiring engine
│   ├── comp.py               # Compensation engine
│   ├── cola.py               # Cost-of-living adjustments
│   ├── promotion.py          # Promotion engine
│   ├── markov_promotion.py   # Advanced promotion modeling
│   ├── nh_termination.py     # New hire termination
│   ├── compensation.py       # Compensation orchestration
│   └── run_one_year/         # Year-level orchestration
├── projections/              # Multi-year projection framework
│   ├── cli.py               # Command-line interface
│   ├── runner.py            # Simulation orchestration
│   ├── dynamic_hazard.py    # Runtime hazard table generation
│   ├── hazard.py            # Static hazard table loading
│   ├── snapshot.py          # Snapshot processing
│   ├── event_log.py         # Event tracking
│   ├── reporting.py         # Output generation
│   ├── config.py            # Projection configuration
│   └── utils.py             # Projection utilities
├── state/                     # State management
│   ├── schema.py            # Column names and constants
│   ├── age.py               # Age calculations and bands
│   ├── tenure.py            # Tenure calculations
│   ├── job_levels.py        # Organizational hierarchy
│   ├── builder.py           # Snapshot construction
│   ├── snapshot.py          # Snapshot utilities
│   ├── snapshot_update.py   # State transitions
│   └── event_log.py         # Event logging system
├── plan_rules/               # Retirement plan rules
│   ├── auto_enrollment.py   # Auto-enrollment logic
│   ├── auto_increase.py     # Auto-increase rules
│   ├── contribution_increase.py # Contribution adjustments
│   ├── eligibility.py       # Eligibility rules
│   ├── eligibility_events.py # Eligibility tracking
│   ├── enrollment.py        # Enrollment processing
│   ├── contributions.py     # Contribution calculations
│   └── proactive_decrease.py # Contribution reductions
├── rules/                     # Business rule implementations
│   ├── engine.py            # Rule processing orchestration
│   ├── auto_enrollment.py   # Core auto-enrollment
│   ├── auto_increase.py     # Contribution escalation
│   ├── contributions.py     # General contribution logic
│   ├── eligibility.py       # Participation eligibility
│   ├── formula_parsers.py   # Configuration parsing
│   ├── response.py          # Behavioral response modeling
│   └── validators.py        # Rule validation
├── dynamics/                  # Population dynamics
│   ├── engine.py            # Dynamics orchestration
│   ├── compensation.py      # Salary progression
│   ├── hiring.py            # New hire generation
│   ├── termination.py       # Employee exit modeling
│   └── sampling/            # Statistical sampling tools
├── ml/                        # Machine learning
│   ├── turnover.py          # Turnover prediction models
│   └── ml_utils.py          # ML utilities
├── schema/                    # Data structure definitions
│   ├── columns.py           # Column definitions
│   ├── dtypes.py            # Data type validation
│   ├── events.py            # Event structure
│   ├── migration.py         # Schema migration
│   └── validation.py        # Data validation
├── reporting/                 # Analysis and reporting
│   └── metrics.py           # KPI calculations
└── utils/                     # Shared utilities
    ├── columns.py           # Column constants
    ├── constants.py         # System constants
    ├── date_utils.py        # Date/time helpers
    ├── data_processing.py   # Data manipulation
    ├── dataframe_validator.py # DataFrame validation
    ├── id_generation.py     # ID creation
    ├── tenure_utils.py      # Tenure calculations
    ├── simulation_utils.py  # Simulation helpers
    ├── census_generation_helpers.py # Test data
    └── compensation/        # Compensation utilities
```

### Configuration System
- **Hierarchical YAML Configuration**: All parameters in `config/` directory
- **Pydantic Validation**: Configuration validated using models in `cost_model/config/models.py`
- **Scenario Support**: Multiple scenarios per config file
- **Parameter Inheritance**: Scenarios inherit from global parameters with overrides

### Data Flow
1. **Input**: Census data (Parquet/CSV) + Configuration (YAML)
2. **Configuration Loading**: Pydantic validation and parameter extraction
3. **Hazard Table Generation**: Dynamic hazard tables created from configuration
4. **Initialization**: Initial snapshot from census with age/tenure calculations
5. **Year Processing**: Multi-engine pipeline per simulation year:
   - **Terminations**: Enhanced engine with intelligent fallback logic
   - **Hiring**: New employee generation with demographic targeting
   - **Promotions**: Career progression with age-based multipliers
   - **Compensation**: Merit raises, COLA adjustments, promotion increases
   - **Plan Rules**: Retirement plan eligibility and contribution processing
6. **State Updates**: Snapshot mutations and event logging
7. **Output**: Yearly snapshots + consolidated data + comprehensive event logs + summary analytics

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

### Schema Constants
- **Employee Schema**: Defined in `cost_model/state/schema.py`
  - `EMP_ID`: Employee identifier
  - `EMP_LEVEL`: Employee level (0-4 standard range)
  - `EMP_TENURE_BAND`: Tenure categories ('<1', '1-3', '3-5', '5-10', '10-15', '15+')
  - `EMP_AGE`, `EMP_AGE_BAND`: Age and age band calculations
  - `EMP_GROSS_COMP`: Employee compensation
  - `EMP_ACTIVE`: Active status flag
- **Event Schema**: Event types and structure in `cost_model/schema/events.py`
- **Column Utilities**: Helper constants in `cost_model/utils/columns.py`

### Data Conventions
- **Date Handling**: Pandas Timestamps with consistent `as_of_date` logic
- **Monetary Values**: Float precision for performance (no Decimal)
- **Random Generation**: Numpy Generator with configurable seeds for reproducibility
- **Logging**: Structured logging via `logging_config.py` with appropriate levels
- **File Formats**: Automatic Parquet/CSV detection and optimization

## Recent System Enhancements

### Enhanced Termination Engine (Latest)
- **Intelligent Fallback Logic**: Multi-strategy approach for missing hazard table combinations
- **Comprehensive Diagnostics**: Detailed logging for troubleshooting missing combinations
- **Age Sensitivity Integration**: Age-based multipliers for realistic retirement modeling
- **New Hire Termination**: Specialized handling for new employee attrition patterns

### Auto-Tuning System
- **Production-Ready Calibration**: Automated parameter optimization for realistic simulations
- **Multi-Campaign Support**: Iterative refinement with comprehensive result tracking
- **Evidence-Based Parameters**: Uses BLS/SSA data for realistic parameter ranges
- **Multi-Objective Scoring**: Balances demographic preservation, headcount growth, and compensation targets

### Dynamic Hazard Tables
- **Runtime Generation**: Creates hazard tables based on configuration parameters
- **Multi-Year Support**: Handles time-varying parameters across simulation years
- **Flexible Parameter Structure**: Supports complex scenarios and parameter combinations

### Robust Data Processing
- **Format Detection**: Automatic Parquet/CSV handling with optimization
- **Schema Validation**: Comprehensive data quality checks and error recovery
- **Memory Optimization**: Efficient handling of large datasets
- **Column Standardization**: Automatic mapping of common column variations

## Current Development Focus

### System Stability
1. **Production Validation**: Ensuring all engines work reliably with real-world data
2. **Performance Optimization**: Efficient processing of large employee populations
3. **Error Recovery**: Graceful handling of edge cases and data quality issues
4. **Comprehensive Testing**: Full coverage of simulation scenarios and edge cases

### Common Debugging Areas
- Check logs in `output_dev/projection_logs/` for detailed diagnostics
- Validate hazard table coverage for all employee level-tenure combinations
- Ensure age calculations are consistent across simulation and projection modules
- Verify event log integrity and snapshot consistency
- Monitor memory usage with large datasets

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