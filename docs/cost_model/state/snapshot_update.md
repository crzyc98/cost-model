# QuickStart

To update an existing workforce snapshot with new events programmatically:

```python
import pandas as pd
import logging
from cost_model.state.snapshot_update import update
from cost_model.state.event_log import EVT_HIRE, EVT_TERM, EVT_COMP
from cost_model.utils.columns import EMP_ID, EMP_LEVEL, EMP_LEVEL_SOURCE, EMP_HIRE_DATE, EMP_BIRTH_DATE, EMP_TERM_DATE, EMP_GROSS_COMP, EMP_DEFERRAL_RATE, EMP_ACTIVE, EMP_TENURE_BAND, EMP_LEVEL_SOURCE, EMP_EXITED, EMP_TENURE, SIMULATION_YEAR, TERM_RATE, COMP_RAISE_PCT, NEW_HIRE_TERM_RATE, COLA_PCT, CFG

# Configure logging to see detailed information about the update process
logging.basicConfig(level=logging.DEBUG)

# Load an existing snapshot
prev_snapshot = pd.read_parquet('data/snapshot_2024.parquet')

# Create or load new events that occurred since the previous snapshot
new_events = pd.DataFrame([
    # New hire event
    {
        'event_id': 'evt_001',
        'event_time': pd.Timestamp('2025-01-15'),
        'employee_id': 'EMP101',
        'event_type': EVT_HIRE,
        'value_num': 60000.0,  # Starting compensation
        'value_json': None,
        'meta': 'New hire - Software Engineer'
    },
    # Termination event for existing employee
    {
        'event_id': 'evt_002',
        'event_time': pd.Timestamp('2025-03-31'),
        'employee_id': 'EMP050',  # Existing employee ID
        'event_type': EVT_TERM,
        'value_num': None,
        'value_json': None,
        'meta': 'Voluntary resignation'
    },
    # Compensation change for existing employee
    {
        'event_id': 'evt_003',
        'event_time': pd.Timestamp('2025-04-01'),
        'employee_id': 'EMP025',  # Existing employee ID
        'event_type': EVT_COMP,
        'value_num': 85000.0,  # New compensation
        'value_json': None,
        'meta': 'Annual merit increase'
    }
])

# Update the snapshot for the new year
snapshot_year = 2025
updated_snapshot = update(prev_snapshot, new_events, snapshot_year)

# Examine the results
print(f"Updated snapshot has {len(updated_snapshot)} employees")
print(f"Active employees: {updated_snapshot['active'].sum()}")

# Check for the new hire
if 'EMP101' in updated_snapshot.index:
    print(f"New hire EMP101 added with compensation: {updated_snapshot.loc['EMP101', 'employee_gross_compensation']}")

# Check for the terminated employee
if 'EMP050' in updated_snapshot.index:
    is_active = updated_snapshot.loc['EMP050', 'active']
    term_date = updated_snapshot.loc['EMP050', 'employee_termination_date']
    print(f"EMP050 active status: {is_active}, termination date: {term_date}")

# Save the updated snapshot
updated_snapshot.to_parquet(f'data/snapshot_{snapshot_year}.parquet')
```
