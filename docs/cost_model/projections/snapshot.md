## QuickStart

To create and update snapshots programmatically during projections:

```python
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from cost_model.projections.snapshot import create_initial_snapshot, update_snapshot_with_events, consolidate_snapshots_to_parquet

# Create an initial snapshot from a census file
start_year = 2025
census_path = Path('data/census.parquet')
initial_snapshot = create_initial_snapshot(start_year, census_path)
print(f"Created initial snapshot with {len(initial_snapshot)} employees")
print(f"Active employees: {initial_snapshot['active'].sum()}")

# Save the initial snapshot
output_dir = Path('output/snapshots')
output_dir.mkdir(parents=True, exist_ok=True)
initial_snapshot.to_parquet(output_dir / f'snapshot_{start_year}_initial.parquet')

# Create sample events for the first projection year
events = pd.DataFrame([
    # Hire event
    {
        'event_id': '001',
        'event_time': pd.Timestamp(f'{start_year}-03-15'),
        'employee_id': 'EMP101',
        'event_type': 'EVT_HIRE',
        'value_num': 80000.0,
        'value_json': '{"role": "Engineer", "birth_date": "1990-05-12"}',
        'meta': None
    },
    # Termination event
    {
        'event_id': '002',
        'event_time': pd.Timestamp(f'{start_year}-06-30'),
        'employee_id': initial_snapshot.index[0],  # First employee in initial snapshot
        'event_type': 'EVT_TERM',
        'value_num': None,
        'value_json': None,
        'meta': None
    },
    # Compensation event
    {
        'event_id': '003',
        'event_time': pd.Timestamp(f'{start_year}-07-01'),
        'employee_id': initial_snapshot.index[1],  # Second employee in initial snapshot
        'event_type': 'EVT_COMP',
        'value_num': 85000.0,
        'value_json': None,
        'meta': None
    }
])

# Define event priority (lower number = higher priority)
event_priority = {
    'EVT_HIRE': 1,
    'EVT_TERM': 2,
    'EVT_COMP': 3,
    'EVT_COLA': 4,
    'EVT_PROMOTION': 5,
    'EVT_RAISE': 6,
    'EVT_CONTRIB': 7
}

# Update the snapshot with events up to mid-year
mid_year_date = pd.Timestamp(f'{start_year}-06-30')
updated_snapshot, active_employees = update_snapshot_with_events(
    initial_snapshot,
    events,
    mid_year_date,
    event_priority
)

print(f"\nUpdated snapshot as of {mid_year_date}:")
print(f"Total employees: {len(updated_snapshot)}")
print(f"Active employees: {len(active_employees)}")

# Check for changes in compensation
if 'employee_gross_compensation' in initial_snapshot.columns and 'employee_gross_compensation' in updated_snapshot.columns:
    initial_comp = initial_snapshot['employee_gross_compensation'].sum()
    updated_comp = updated_snapshot['employee_gross_compensation'].sum()
    print(f"Total compensation change: ${updated_comp - initial_comp:,.2f}")

# Update the snapshot with all events for the full year
year_end_date = pd.Timestamp(f'{start_year}-12-31')
final_snapshot, active_employees = update_snapshot_with_events(
    initial_snapshot,
    events,
    year_end_date,
    event_priority
)

print(f"\nFinal snapshot as of {year_end_date}:")
print(f"Total employees: {len(final_snapshot)}")
print(f"Active employees: {len(active_employees)}")

# Save the final snapshot
final_snapshot.to_parquet(output_dir / f'snapshot_{start_year}_final.parquet')
```

This demonstrates how to create an initial snapshot from census data and update it with events during a projection.