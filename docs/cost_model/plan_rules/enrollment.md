## QuickStart

To generate enrollment events programmatically:

```python
import pandas as pd
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from cost_model.plan_rules.enrollment import run, EVT_ELIGIBLE, EVT_AUTO_ENROLL, EVT_ENROLL, EVT_OPT_OUT
from cost_model.config.plan_rules import EnrollmentConfig

# Create a sample snapshot with employee data
snapshot = pd.DataFrame({
    'employee_id': ['EMP001', 'EMP002', 'EMP003', 'EMP004'],
    'employee_gross_compensation': [75000.0, 85000.0, 65000.0, 55000.0],
    'active': [True, True, True, True],
    'eligible': [True, True, True, True],
    'enrolled': [False, False, False, False]
}).set_index('employee_id')

# Create a sample events DataFrame with eligibility events
as_of = pd.Timestamp('2025-06-15')
eligibility_date = as_of - timedelta(days=30)  # Eligible 30 days ago

events = pd.DataFrame([
    {
        'event_id': str(uuid.uuid4()),
        'event_time': eligibility_date,
        'employee_id': emp_id,
        'event_type': EVT_ELIGIBLE,
        'value_num': None,
        'value_json': None,
        'meta': None
    } for emp_id in snapshot.index
])

# Configure enrollment rules
enrollment_config = EnrollmentConfig(
    enabled=True,
    auto_enroll=True,                # Enable auto-enrollment
    auto_enroll_delay_days=30,       # Wait 30 days after eligibility
    auto_enroll_default_pct=3.0,     # Default contribution rate of 3%
    opt_out_pct=25.0,                # 25% of employees opt out
    opt_out_delay_days=15            # Opt out within 15 days of auto-enrollment
)

# Generate enrollment events
enrollment_events = run(snapshot, events, as_of, enrollment_config)

# Examine the generated events
if enrollment_events:
    # Combine all event DataFrames
    all_events = pd.concat(enrollment_events)
    print(f"Generated {len(all_events)} enrollment events")
    
    # Count events by type
    event_counts = all_events['event_type'].value_counts()
    print("\nEvent counts by type:")
    for event_type, count in event_counts.items():
        print(f"{event_type}: {count}")
    
    # Show auto-enrollment details
    auto_enroll_events = all_events[all_events['event_type'] == EVT_AUTO_ENROLL]
    if not auto_enroll_events.empty:
        print("\nAuto-enrollment details:")
        print(auto_enroll_events[['employee_id', 'event_time', 'value_num']].head())
    
    # Show opt-out details
    opt_out_events = all_events[all_events['event_type'] == EVT_OPT_OUT]
    if not opt_out_events.empty:
        print("\nOpt-out details:")
        print(opt_out_events[['employee_id', 'event_time']].head())
    
    # Save the events
    output_dir = Path('output/enrollment')
    output_dir.mkdir(parents=True, exist_ok=True)
    all_events.to_parquet(output_dir / 'enrollment_events_2025.parquet')
else:
    print("No enrollment events generated")
```

This demonstrates how to generate auto-enrollment and opt-out events based on eligibility and plan rules.