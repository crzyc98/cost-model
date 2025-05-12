# Cost Model Test Suite

This directory contains the test suite for the retirement plan cost model. The tests are organized into categories to make it easier to run specific groups of tests.

## Test Organization

```
tests/
├── unit/                   # Unit tests for individual components
│   ├── config/             # Tests for config loading/validation
│   ├── plan_rules/         # Tests for individual plan rules
│   ├── engines/            # Tests for simulation engines
│   ├── dynamics/           # Tests for population dynamics
│   ├── state/              # Tests for state management
│   └── utils/              # Tests for utilities
├── integration/            # Tests that span multiple components
│   ├── test_full_simulation.py  # Full simulation tests
│   └── test_one_year.py    # Single-year simulation tests
├── abm/                    # Agent-based model tests
│   ├── test_agent_eligibility.py  # Tests for agent eligibility
│   └── test_employee_agent.py     # Tests for employee agent behavior
├── quick/                  # Fast-running tests for CI
│   └── test_smoke.py       # Basic smoke tests
└── conftest.py             # Shared test fixtures and configuration
```

## Running Tests

### Running All Tests

```bash
pytest
```

### Running Tests by Category

```bash
# Run all unit tests
pytest tests/unit/

# Run all integration tests
pytest -m integration

# Run all ABM tests
pytest -m abm

# Run all quick tests (for CI)
pytest -m quick
```

### Running Tests by Component

```bash
# Run all plan_rules tests
pytest -m plan_rules

# Run all engines tests
pytest -m engines

# Run all state tests
pytest -m state

# Run all dynamics tests
pytest -m dynamics

# Run all config tests
pytest -m config

# Run all utils tests
pytest -m utils
```

### Running Fast Tests for CI

```bash
# Run all quick tests
pytest -m quick
```

### Excluding Slow Tests

```bash
# Run all tests except slow tests
pytest -m "not slow"
```

## Adding New Tests

When adding new tests, please follow these guidelines:

1. Place the test in the appropriate directory based on what it's testing
2. Add the appropriate pytest markers to categorize the test
3. Use descriptive test names that indicate what's being tested
4. Include docstrings that explain the purpose of the test

Example:

```python
import pytest

@pytest.mark.unit
@pytest.mark.plan_rules
def test_eligibility_calculation():
    """Test that eligibility is correctly calculated based on age and service."""
    # Test implementation here
```

## Test Fixtures

Common test fixtures are defined in `conftest.py`. If you need to add a new fixture, please add it there so it can be shared across tests.
