# Agents Module Overview

This directory contains the implementation of the `EmployeeAgent` and a set of focused mixins that encapsulate individual aspects of agent behavior in the Mesa-based ABM.

## Files

- **employee_agent.py**  
  Defines the `EmployeeAgent` class, which wires together all mixins and serves as the core agent constructor. All behavior and state logic is inherited from specialized mixins.

- **behavior.py**  
  `BehaviorMixin` orchestrates the annual simulation step, invoking in order:
  1. `_update_compensation()`
  2. `_update_eligibility()`
  3. `_make_deferral_decision()`
  4. `_calculate_contributions()`
  5. `_determine_status_for_year()`

- **state.py**  
  `StateMixin` provides:
  - Shared constants: `ZERO_DECIMAL`, `ENROLL_METHOD_*`, `STATUS_*`
  - Status lifecycle methods: `_initialize_employment_status()`, `_determine_status_for_year()`
  - Date-based helpers: `_calculate_age()`, `_calculate_tenure_months()`

- **compensation.py**  
  `CompensationMixin` implements `_update_compensation()`, applying annual raise rules from the model configuration.

- **eligibility.py**  
  `EligibilityMixin` implements `_update_eligibility()`, checking plan rules (age and service requirements).

- **deferral.py**  
  `DeferralMixin` implements `_make_deferral_decision()`, handling auto-enrollment, auto-increase, and voluntary rate adjustments.

- **contributions.py**  
  `ContributionsMixin` implements `_calculate_contributions()`, computing employee pre-tax and catch-up contributions, and employer match/NEC based on formulas.

## How It Fits Together

The `EmployeeAgent` class inherits in order:

```python
class EmployeeAgent(
    BehaviorMixin,
    StateMixin,
    CompensationMixin,
    EligibilityMixin,
    DeferralMixin,
    ContributionsMixin,
    mesa.Agent
):
    def __init__(...):
        super().__init__(...)
        # set initial attributes
```

This modular design keeps each concern isolated, making the codebase easier to maintain, test, and extend.

---

_For details on configuration and running the simulation, see the top-level README.md._
