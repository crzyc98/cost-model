# File Analysis for ABM Retirement Plan Simulation

This document analyzes the three key files in your Agent-Based Modeling (ABM) simulation using Mesa:
- `agent.py`
- `model.py`
- `run_abm_simulation.py`

For each file, I identify core production components, code that can be archived, and streamlining opportunities.

---

## 1. agent.py

### Role
Defines the `EmployeeAgent` class, representing each employee in the simulation. Handles agent state (demographics, compensation, eligibility, participation, behavioral flags), decision logic (auto-enrollment, auto-increase, voluntary changes), and stepwise updates.

### Needed for Production
- `EmployeeAgent` class: All core attributes and methods related to agent state, eligibility, deferral decisions, compensation updates, and annual step logic are essential.
- Initialization logic for demographics, employment status, and plan participation.
- Methods for eligibility checks, compensation updates, and deferral decision logic.
- Any code using `Decimal` for financial precision.

### Can Be Archived
- Extensive debug print statements (e.g., "DEBUG Agent ...", "DEBUG NH ...") and debug counters. For production, replace with logging (at DEBUG level) or remove.
- Placeholder/legacy fields (e.g., `behavioral_profile` if not used in production logic).
- Comments and code blocks marked as optional or legacy (e.g., old/grandfathering logic, commented-out prints).

### Streamlining Suggestions
- Replace all print-based debugging with Python `logging` (already partially present).
- Remove or condense legacy/experimental attributes and debug flags.
- Ensure only attributes needed for simulation output and analysis are kept.

---

## 2. model.py

### Role
Defines the `RetirementPlanModel` class, orchestrating the simulation loop, agent population, hiring/termination, plan rules, and data collection. Handles scenario configuration, population dynamics, and output aggregation.

### Needed for Production
- `RetirementPlanModel` class: Initialization, agent population management, scenario config parsing, annual step logic.
- Methods for creating new hires, handling terminations, updating the schedule, and collecting results.
- DataCollector setup for model-level and agent-level variables.
- All logic using `Decimal` for financial calculations and plan rule parsing.

### Can Be Archived
- Extensive debug print statements ("DEBUG MODEL INIT", census checks, agent population summaries, etc.).
- Comments and code for experimental or legacy config handling (e.g., fallback/placeholder config keys no longer in use).
- Placeholder enums/classes (e.g., fallback `EmploymentStatus` definition if not actually used).

### Streamlining Suggestions
- Replace print-based debugging with logging (already partially present).
- Remove legacy/experimental config handling and fallback code.
- Ensure only scenario config keys actually used in production are parsed and validated.
- Consider moving utility functions (e.g., max match rate parsing) to a utilities module if reused.

---

## 3. run_abm_simulation.py

### Role
Main entry point for running the simulation. Loads configuration and census data, initializes the model, runs the simulation loop, and saves results.

### Needed for Production
- `run_simulation` function: Handles loading config/data, running the model, and saving results.
- Argument parsing for config, census, and output paths.
- Data cleaning and type conversion (especially for financial/ID fields).
- Calls to `RetirementPlanModel` and its methods.

### Can Be Archived
- Print statements for step-by-step progress and error reporting (replace with logging for production).
- Optional/legacy sample data prints and commented-out debug code.
- Warnings about missing statuses if not needed for production monitoring.

### Streamlining Suggestions
- Replace all print statements with logging (already partially present).
- Remove commented-out code and optional sample data prints.
- Ensure only essential error handling and reporting remains.

---

# General Streamlining Recommendations
- **Logging:** Use Python's `logging` throughout for debug/info/error reporting. Set log level via config/CLI for production vs. debug runs.
- **Remove Print Debugging:** Archive or remove all print-based debugging after validating production stability.
- **Modularization:** Move any reusable utility functions to a shared module.
- **Config Validation:** Keep only config keys and validation logic needed for your current plan rules and scenarios.
- **Documentation:** Update docstrings and remove outdated comments.

If you want, I can create a checklist or script to help automate this cleanup.
