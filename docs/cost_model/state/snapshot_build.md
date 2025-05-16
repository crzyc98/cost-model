To build a complete workforce snapshot from an event log programmatically:

```python
import pandas as pd
import logging
from pathlib import Path
from cost_model.state.snapshot_build import build_full
from cost_model.state.event_log import EVT_HIRE, EVT_COMP, EVT_TERM

# Configure logging to see detailed information
logging.basicConfig(level=logging.INFO)

# Load or create an event log
events_df = pd.read_parquet('data/events_2025.parquet')

# Alternatively, create a synthetic event log for testing
synth_events = pd.DataFrame([
    # Hire events
    {
        'event_id': 'evt_h1',
        'event_time': pd.Timestamp('2025-01-15'),
        'employee_id': 'EMP001',
        'event_type': EVT_HIRE,
        'value_num': 75000.0,  # Starting compensation
        'value_json': '{"role": "Engineer", "birth_date": "1990-05-12"}',
        'meta': 'Initial hire'
    },
    {
        'event_id': 'evt_h2',
        'event_time': pd.Timestamp('2025-02-01'),
        'employee_id': 'EMP002',
        'event_type': EVT_HIRE,
        'value_num': 85000.0,
        'value_json': '{"role": "Manager", "birth_date": "1985-08-23"}',
        'meta': 'Initial hire'
    },
    # Compensation change event
    {
        'event_id': 'evt_c1',
        'event_time': pd.Timestamp('2025-06-01'),
        'employee_id': 'EMP001',
        'event_type': EVT_COMP,
        'value_num': 80000.0,  # New compensation
        'value_json': None,
        'meta': 'Mid-year adjustment'
    },
    # Termination event
    {
        'event_id': 'evt_t1',
        'event_time': pd.Timestamp('2025-09-30'),
        'employee_id': 'EMP002',
        'event_type': EVT_TERM,
        'value_num': None,
        'value_json': None,
        'meta': 'Voluntary termination'
    }
])

# Build a snapshot for the specified year
snapshot_year = 2025
snapshot_df = build_full(events_df, snapshot_year)

# Examine the snapshot
print(f"Snapshot contains {len(snapshot_df)} employees")
print(f"Active employees: {snapshot_df['active'].sum()}")
print(f"Terminated employees: {(~snapshot_df['active']).sum()}")

# Check tenure calculations
print("\nTenure distribution:")
if 'tenure_band' in snapshot_df.columns:
    print(snapshot_df['tenure_band'].value_counts())

# Save the snapshot
output_dir = Path('output/snapshots')
output_dir.mkdir(parents=True, exist_ok=True)
snapshot_df.to_parquet(output_dir / f'snapshot_{snapshot_year}.parquet')
```

This demonstrates how to build a complete workforce snapshot from an event log, including handling of hire, compensation, and termination events.