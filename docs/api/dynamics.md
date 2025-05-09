# cost_model.dynamics.engine

This module manages the simulation of workforce dynamics, including hiring, terminations, and state transitions for each year. It coordinates the application of business rules to the evolving employee population.

Example usage:

```python
from cost_model.dynamics.engine import run_dynamics_for_year
run_dynamics_for_year(employee_df, config, year)
```

::: cost_model.dynamics.engine
