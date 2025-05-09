# cost_model.simulation

This module contains your **multi-year orchestration** entrypoint: spinning up the census, running each year’s mini-engine (`comp`, `term`, `hire`), persisting snapshots, and producing summary metrics.

Example usage:

```python
from cost_model.simulation import run_simulation
# ... set up config, census, output_dir ...
run_simulation(config, 'Baseline', census_path, output_dir)
```

::: cost_model.simulation
