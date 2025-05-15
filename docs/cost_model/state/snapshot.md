## QuickStart

To work with workforce snapshots programmatically:

```python
import pandas as pd
from pathlib import Path
from cost_model.state.snapshot import build_full, update
from cost_model.state.builder import SnapshotBuilder  # Higher-level API

# Option 1: Use the high-level SnapshotBuilder API (recommended)
snapshot_year = 2025
builder = SnapshotBuilder(snapshot_year)

# Build a snapshot from an event log
events_df = pd.read_parquet('data/events_2025.parquet')
snapshot_df = builder.build(events_df)
print(f"Built snapshot with {len(snapshot_df)} employees")

# Update a snapshot with new events
prev_snapshot = pd.read_parquet('data/snapshot_2024.parquet')
new_events = pd.read_parquet('data/new_events_2025.parquet')
updated_snapshot = builder.update(prev_snapshot, new_events)
print(f"Updated snapshot: {updated_snapshot['active'].sum()} active employees")

# Option 2: Use the direct functions (for more control)

# Build a snapshot from scratch
events_df = pd.read_parquet('data/all_events.parquet')
snapshot_df = build_full(events_df, snapshot_year)

# Update a snapshot with new events
prev_snapshot = pd.read_parquet('data/snapshot_2024.parquet')
new_events = pd.read_parquet('data/new_events_2025.parquet')
updated_snapshot = update(prev_snapshot, new_events, snapshot_year)

# Analyze the snapshot
active_employees = updated_snapshot[updated_snapshot['active']]
print(f"Active employees: {len(active_employees)}")

terminated_employees = updated_snapshot[~updated_snapshot['active']]
print(f"Terminated employees: {len(terminated_employees)}")

# Check for experienced terminated employees (employees hired before the current year
# but terminated during the current year)
hire_dates = pd.to_datetime(updated_snapshot['employee_hire_date'])
term_dates = pd.to_datetime(updated_snapshot['employee_termination_date'])

experienced_terminated = updated_snapshot[
    (~updated_snapshot['active']) &  # Terminated
    (hire_dates.dt.year < snapshot_year) &  # Hired before this year
    (term_dates.dt.year == snapshot_year)   # Terminated this year
]
print(f"Experienced terminated employees: {len(experienced_terminated)}")

# Save the snapshot
updated_snapshot.to_parquet(f'data/snapshot_{snapshot_year}.parquet')
```

This demonstrates both the high-level SnapshotBuilder API and the direct functions for building and updating snapshots, with a focus on analyzing employment status including experienced terminated employees.