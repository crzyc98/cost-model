# cost_model.state.snapshot

This module defines the simulation state and snapshotting logic. It provides data structures and utilities for capturing the state of all employees and plan parameters at each simulation year, enabling downstream analysis and reporting.

Example usage:

```python
from cost_model.state.snapshot import take_snapshot
snapshot = take_snapshot(current_state)
```

::: cost_model.state.snapshot
