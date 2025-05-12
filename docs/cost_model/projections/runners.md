## QuickStart

To run a projection programmatically from a Python REPL or notebook:

```python
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
from cost_model.projections.config import load_config_to_namespace
from cost_model.projections.runner import run_projection_engine
from cost_model.projections.snapshot import create_initial_snapshot
from cost_model.projections.event_log import create_initial_event_log

# 1. Load configuration
config_path = 'config/my_scenario.yaml'
config_ns = load_config_to_namespace(config_path)

# 2. Create initial snapshot from census
census_path = 'data/census_2025.parquet'
start_year = config_ns.global_parameters.start_year
initial_snapshot = create_initial_snapshot(start_year, census_path)

# 3. Create initial event log
initial_event_log = create_initial_event_log(start_year)

# 4. Run the projection engine
(
    yearly_snapshots,  # Dict[int, pd.DataFrame] - Snapshots by year
    final_snapshot,    # pd.DataFrame - Final EOY snapshot
    event_log,         # pd.DataFrame - Cumulative event log
    core_summary,      # pd.DataFrame - Core metrics by year
    employment_summary # pd.DataFrame - Employment status by year
) = run_projection_engine(
    config_ns=config_ns,
    initial_snapshot_df=initial_snapshot,
    initial_event_log_df=initial_event_log
)

# 5. Analyze results
print(f"Projection complete: {len(yearly_snapshots)} years")
print(f"Final headcount: {final_snapshot[final_snapshot['active']].shape[0]}")
print(core_summary[['Year', 'Active Headcount', 'Participant Count']])
```

This runs a complete projection and returns all the key outputs for analysis.