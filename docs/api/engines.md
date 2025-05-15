# cost_model.engines

This package contains the simulation engines for core HR events: hiring, terminations, and running a simulation year.

## Compensation Engine
Handles logic for applying compensation changes and bumps to employees.

::: cost_model.engines.comp

## Termination Engine
Simulates terminations across the active population.

::: cost_model.engines.term

## One-Year Simulation Engine
Runs a single year of the simulation, applying comp, term, hire, and snapshot logic.

::: cost_model.engines.run_one_year_engine
