# Eligibility Events Engine

The eligibility events engine emits milestone eligibility events for employees who cross configured service milestones between two dates.

## QuickStart

To generate eligibility milestone events programmatically:

```python
import pandas as pd
from datetime import datetime
from uuid import uuid4
from cost_model.plan_rules.eligibility_events import run
from cost_model.config.plan_rules import EligibilityEventsConfig
from cost_model.utils.columns import EMP_ID, EMP_HIRE_DATE

# Create a sample snapshot with employee data
snapshot = pd.DataFrame({
    'employee_id': ['EMP001', 'EMP002', 'EMP003'],
    'employee_hire_date': [
        pd.Timestamp('2023-01-15'),  # 2+ years of service
        pd.Timestamp('2024-06-01'),  # 1 year of service
        pd.Timestamp('2025-01-01')   # 6 months of service
    ],
    'active': [True, True, True]
}).set_index('employee_id')

# Create a sample events DataFrame
events = pd.DataFrame(columns=[
    'event_id', 'event_time', 'employee_id', 'event_type',
    'value_num', 'value_json', 'meta'
])

# Set the reference dates
as_of = pd.Timestamp('2025-06-15')
prev_as_of = pd.Timestamp('2025-01-01')

# Configure eligibility milestones
eligibility_config = EligibilityEventsConfig(
    milestone_months=[6, 12, 24],  # 6 months, 12 months, 24 months
    milestone_years=[1, 2],        # 1 year, 2 years (automatically converted to months)
    event_type_map={
        6: 'EVT_ELIGIBLE_6M',      # 6 months milestone
        12: 'EVT_ELIGIBLE_1Y',     # 1 year milestone
        24: 'EVT_ELIGIBLE_2Y'      # 2 year milestone
    }
)

# Generate eligibility events
eligibility_events = run(
    snapshot=snapshot,
    events=events,
    as_of=as_of,
    prev_as_of=prev_as_of,
    cfg=eligibility_config
)

# Examine the generated events
if eligibility_events:
    events_df = eligibility_events[0]
    print(f"Generated {len(events_df)} eligibility events")
    
    # Show eligibility details
    print("\nEligibility milestones:")
    for _, event in events_df.iterrows():
        emp_id = event['employee_id']
        milestone = json.loads(event['value_json'])['milestone_months']
        print(f"  {emp_id}: Reached {milestone} months of service")
    
    # Save the events
    output_dir = Path('output/eligibility')
    output_dir.mkdir(parents=True, exist_ok=True)
    events_df.to_parquet(output_dir / 'eligibility_events_2025.parquet')
else:
    print("No eligibility events generated")
```

## Key Features

- Tracks service milestones in months and years
- Automatically converts years to months for milestone calculations
- Configurable milestone events and event types
- Handles missing previous date by defaulting to before everyone's hire date
- Ensures proper index handling for employee IDs
- Validates required columns in the snapshot
- Returns empty list if no milestones are configured

## Configuration Options

The `EligibilityEventsConfig` class allows you to configure:

- `milestone_months`: List of months to track (e.g., [6, 12, 24])
- `milestone_years`: List of years to track (automatically converted to months)
- `event_type_map`: Mapping of milestone months to event types

## Event Schema

The engine generates events with the following schema:

- `event_id`: Unique identifier for the event
- `event_time`: Timestamp when the milestone was reached
- `employee_id`: ID of the employee who reached the milestone
- `event_type`: Type of eligibility event (configured via event_type_map)
- `value_num`: None (not used for eligibility events)
- `value_json`: JSON string containing milestone information
- `meta`: Metadata (currently not used)

## Best Practices

1. Always validate that your snapshot contains the required columns:
   - `employee_id` (as index or column)
   - `employee_hire_date`

2. Use meaningful event types in your `event_type_map` to distinguish different milestones

3. Consider the impact of your milestone configuration on performance:
   - Too many milestones can generate large numbers of events
   - Milestones should be meaningful points in an employee's career

4. When in doubt about previous date, let the engine default to before everyone's hire date

## Common Use Cases

1. Retirement plan eligibility tracking
2. Benefits enrollment milestones
3. Performance review cycles
4. Career progression tracking
5. Training program eligibility

## Error Handling

The engine includes several error checks:

1. Raises `ValueError` if `employee_id` column is missing and not set as index
2. Raises `ValueError` if `employee_hire_date` column is missing
3. Returns empty list if no milestones are configured
4. Returns empty list if no employees cross any milestones between dates

## Performance Considerations

- The engine processes each employee individually, which is efficient for typical workforce sizes
- Uses vectorized operations where possible (e.g., for service month calculations)
- Returns early if no milestones are configured to avoid unnecessary processing

## Example Scenarios

### Scenario 1: Basic Eligibility Tracking
```python
# Track 6-month and 1-year eligibility milestones
eligibility_config = EligibilityEventsConfig(
    milestone_months=[6],
    milestone_years=[1],
    event_type_map={
        6: 'EVT_ELIGIBLE_6M',
        12: 'EVT_ELIGIBLE_1Y'
    }
)
```

### Scenario 2: Complex Multi-Year Tracking
```python
# Track multiple milestones across different time periods
eligibility_config = EligibilityEventsConfig(
    milestone_months=[6, 12, 18, 24],
    milestone_years=[1, 2, 3, 4],
    event_type_map={
        6: 'EVT_ELIGIBLE_6M',
        12: 'EVT_ELIGIBLE_1Y',
        24: 'EVT_ELIGIBLE_2Y',
        36: 'EVT_ELIGIBLE_3Y',
        48: 'EVT_ELIGIBLE_4Y'
    }
)
```

## Troubleshooting

If you're not seeing expected eligibility events:

1. Verify that your snapshot contains the required columns
2. Check that your milestone configuration matches your business requirements
3. Confirm that the `as_of` and `prev_as_of` dates span the time period you're interested in
4. Review the `event_type_map` to ensure you're using the correct event types
5. Use the debug logging to track service month calculations

## Output Format

The engine returns a list containing a single DataFrame with the generated events. Each event includes:

- The employee ID who reached the milestone
- The date when the milestone was reached
- The milestone in months (stored in value_json)
- The configured event type for that milestone
