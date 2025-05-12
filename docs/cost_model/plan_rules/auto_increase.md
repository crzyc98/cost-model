## QuickStart

To generate auto-increase events programmatically:

```python
import pandas as pd
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from cost_model.plan_rules.auto_increase import run
from cost_model.config.plan_rules import AutoIncreaseConfig

# Create a sample snapshot with employee data
snapshot = pd.DataFrame({
    'employee_id': ['EMP001', 'EMP002', 'EMP003', 'EMP004'],
    'employee_gross_compensation': [75000.0, 85000.0, 65000.0, 55000.0],
    'employee_deferral_rate': [3.0, 4.0, 5.0, 6.0],  # Current deferral rates
    'active': [True, True, True, True],
    'eligible': [True, True, True, True],
    'enrolled': [True, True, True, True]
}).set_index('employee_id')

# Create a sample events DataFrame with enrollment events
as_of = pd.Timestamp('2025-06-15')
enrollment_date = as_of - timedelta(days=180)  # Enrolled 6 months ago

events = pd.DataFrame([
    # Enrollment events
    {
        'event_id': str(uuid.uuid4()),
        'event_time': enrollment_date,
        'employee_id': 'EMP001',
        'event_type': 'EVT_ENROLL',
        'value_num': 3.0,
        'value_json': None,
        'meta': None
    },
    {
        'event_id': str(uuid.uuid4()),
        'event_time': enrollment_date,
        'employee_id': 'EMP002',
        'event_type': 'EVT_AUTO_ENROLL',
        'value_num': 4.0,
        'value_json': None,
        'meta': None
    },
    {
        'event_id': str(uuid.uuid4()),
        'event_time': enrollment_date,
        'employee_id': 'EMP003',
        'event_type': 'EVT_ENROLL',
        'value_num': 5.0,
        'value_json': None,
        'meta': None
    },
    # Opt-out event for EMP004
    {
        'event_id': str(uuid.uuid4()),
        'event_time': enrollment_date + timedelta(days=15),
        'employee_id': 'EMP004',
        'event_type': 'EVT_OPT_OUT',
        'value_num': None,
        'value_json': None,
        'meta': None
    }
])

# Configure auto-increase rules
auto_increase_config = AutoIncreaseConfig(
    enabled=True,
    increase_rate=1.0,  # Increase by 1% each year
    cap_rate=10.0       # Cap at 10%
)

# Generate auto-increase events
auto_increase_events = run(snapshot, events, as_of, auto_increase_config)

# Examine the generated events
if auto_increase_events:
    # Combine all event DataFrames
    all_events = pd.concat(auto_increase_events)
    print(f"Generated {len(all_events)} auto-increase events")
    
    # Show auto-increase details
    print("\nAuto-increase details:")
    for _, event in all_events.iterrows():
        old_rate = snapshot.loc[event['employee_id'], 'employee_deferral_rate']
        new_rate = event['value_num']
        print(f"Employee {event['employee_id']}: {old_rate}% → {new_rate}% (+{new_rate - old_rate}%)")
    
    # Save the events
    output_dir = Path('output/auto_increase')
    output_dir.mkdir(parents=True, exist_ok=True)
    all_events.to_parquet(output_dir / 'auto_increase_events_2025.parquet')
else:
    print("No auto-increase events generated")
```

This demonstrates how to generate automatic contribution rate increase events for enrolled employees.