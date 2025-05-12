# Proactive Decrease Engine

The proactive decrease engine monitors employee contribution rates and detects significant decreases relative to their peak contribution rates. When a decrease exceeds a configured threshold, it emits a proactive decrease event for plan adjustment.

## QuickStart

To detect and handle proactive decreases in contribution rates:

```python
import pandas as pd
from datetime import datetime
from cost_model.plan_rules.proactive_decrease import run
from cost_model.config.models import ProactiveDecreaseConfig
from cost_model.utils.columns import EMP_ID, CONTRIBUTION_RATE

# Create a sample snapshot with employee data
snapshot = pd.DataFrame({
    'employee_id': ['EMP001', 'EMP002', 'EMP003'],
    'contribution_rate': [0.05, 0.08, 0.10],  # Current contribution rates
    'peak_contribution_rate': [0.10, 0.12, 0.15],  # Historical peak rates
    'active': [True, True, True]
}).set_index('employee_id')

# Create a sample events DataFrame
events = pd.DataFrame(columns=[
    'event_id', 'event_time', 'employee_id', 'event_type',
    'value_num', 'value_json', 'meta'
])

# Set the reference date
as_of = pd.Timestamp('2025-06-15')

# Configure proactive decrease detection
proactive_config = ProactiveDecreaseConfig(
    lookback_window_months=12,  # Look back 12 months
    decrease_threshold=0.30,    # Detect decreases of 30% or more
    event_type='EVT_PROACTIVE_DECREASE',
    min_contribution_rate=0.01  # Only consider employees with >1% contribution
)

# Generate proactive decrease events
proactive_events = run(
    snapshot=snapshot,
    events=events,
    as_of=as_of,
    cfg=proactive_config
)

# Examine the generated events
if proactive_events:
    events_df = proactive_events[0]
    print(f"Generated {len(events_df)} proactive decrease events")
    
    # Show decrease details
    print("\nProactive decrease details:")
    for _, event in events_df.iterrows():
        emp_id = event['employee_id']
        decrease_info = json.loads(event['value_json'])
        print(f"  {emp_id}: Contribution decreased from {decrease_info['peak_rate']:.2%} to {decrease_info['current_rate']:.2%}")
    
    # Save the events
    output_dir = Path('output/proactive_decrease')
    output_dir.mkdir(parents=True, exist_ok=True)
    events_df.to_parquet(output_dir / 'proactive_decrease_events_2025.parquet')
else:
    print("No proactive decrease events generated")
```

## Key Features

- Monitors contribution rate changes over a configurable lookback window
- Detects significant decreases relative to peak contribution rates
- Configurable decrease threshold for triggering events
- Minimum contribution rate filter to avoid noise from low contributors
- Handles missing peak contribution rates gracefully
- Returns empty list if no decreases exceed the threshold

## Configuration Options

The `ProactiveDecreaseConfig` class allows you to configure:

- `lookback_window_months`: Number of months to look back for peak contribution rates
- `decrease_threshold`: Minimum percentage decrease to trigger an event (e.g., 0.30 for 30%)
- `event_type`: Event type to use for proactive decrease events
- `min_contribution_rate`: Minimum contribution rate to consider employees

## Event Schema

The engine generates events with the following schema:

- `event_id`: Unique identifier for the event
- `event_time`: Timestamp when the decrease was detected
- `employee_id`: ID of the employee with the significant decrease
- `event_type`: Configured event type for proactive decreases
- `value_num`: None (not used for proactive decrease events)
- `value_json`: JSON string containing decrease information
- `meta`: Metadata (currently not used)

The `value_json` contains:
- `peak_rate`: The highest contribution rate observed in the lookback window
- `current_rate`: The current contribution rate
- `decrease_percentage`: The percentage decrease from peak to current
- `lookback_window_months`: The configured lookback window

## Best Practices

1. Set appropriate lookback window based on your business cycle:
   - Shorter windows (3-6 months) for more frequent monitoring
   - Longer windows (12+ months) for more stable contribution tracking

2. Configure decrease threshold based on your tolerance for variance:
   - Lower thresholds (20-30%) for more sensitive detection
   - Higher thresholds (30-50%) to avoid false positives

3. Set minimum contribution rate to avoid noise from low contributors
4. Regularly review and adjust thresholds based on actual patterns

## Common Use Cases

1. Early detection of contribution rate reduction trends
2. Identifying potential engagement issues
3. Monitoring impact of economic changes on contribution behavior
4. Tracking plan utilization patterns
5. Supporting plan adjustment decisions

## Error Handling

The engine includes several error checks:

1. Validates required columns in the snapshot:
   - `employee_id` (as index or column)
   - `contribution_rate`
   - `peak_contribution_rate`

2. Returns empty list if no significant decreases are found
3. Handles missing peak contribution rates by skipping those employees
4. Validates that contribution rates are within expected range (0-1)

## Performance Considerations

- The engine processes each employee individually
- Uses vectorized operations where possible for rate calculations
- Returns early if no employees meet the minimum contribution rate
- Efficiently handles large snapshots by filtering first

## Example Scenarios

### Scenario 1: Basic Proactive Decrease Detection
```python
# Detect decreases of 30% or more over 6 months
proactive_config = ProactiveDecreaseConfig(
    lookback_window_months=6,
    decrease_threshold=0.30,
    event_type='EVT_PROACTIVE_DECREASE',
    min_contribution_rate=0.01
)
```

### Scenario 2: Sensitive Monitoring for High Contributors
```python
# Detect smaller decreases (20%) for high contributors (10%+)
proactive_config = ProactiveDecreaseConfig(
    lookback_window_months=3,
    decrease_threshold=0.20,
    event_type='EVT_PROACTIVE_DECREASE_HIGH',
    min_contribution_rate=0.10
)
```

## Troubleshooting

If you're not seeing expected proactive decrease events:

1. Verify that your snapshot contains the required columns
2. Check that your lookback window is appropriate for your data
3. Review the decrease threshold to ensure it's not too high
4. Confirm that employees have contribution rates above the minimum
5. Use the debug logging to track contribution rate calculations

## Output Format

The engine returns a list containing a single DataFrame with the generated events. Each event includes:

- The employee ID who experienced a significant decrease
- The peak contribution rate observed
- The current contribution rate
- The percentage decrease
- The lookback window used for calculation
