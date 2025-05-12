## QuickStart

To generate contribution events programmatically:

```python
import pandas as pd
import uuid
from pathlib import Path
from datetime import datetime
from cost_model.plan_rules.contributions import run
from cost_model.config.models import EmployerMatchRules, MatchTier

# Create a sample snapshot with employee data
snapshot = pd.DataFrame({
    'employee_id': ['EMP001', 'EMP002', 'EMP003'],
    'employee_gross_compensation': [75000.0, 85000.0, 65000.0],
    'employee_deferral_rate': [6.0, 4.0, 3.0],
    'active': [True, True, True],
    'eligible': [True, True, True],
    'enrolled': [True, True, True]
}).set_index('employee_id')

# Create an empty events DataFrame
events = pd.DataFrame(columns=[
    'event_id', 'event_time', 'employee_id', 'event_type',
    'value_num', 'value_json', 'meta'
])

# Set the reference date
as_of = pd.Timestamp('2025-12-31')

# Configure employer match rules
match_rules = EmployerMatchRules(
    tiers=[
        MatchTier(match_rate=1.0, cap_deferral_pct=3.0),  # 100% match on first 3%
        MatchTier(match_rate=0.5, cap_deferral_pct=2.0)   # 50% match on next 2%
    ],
    dollar_cap=5000  # Annual employer match cap
)

# Generate contribution events
contribution_events = run(snapshot, events, as_of, match_rules)

# Examine the generated events
print(f"Generated {len(contribution_events)} contribution events")
print(contribution_events[['employee_id', 'event_type', 'value_num']].head())

# Calculate total employee and employer contributions
if not contribution_events.empty:
    total_ee = contribution_events['value_num'].sum()
    
    # Parse employer match from value_json
    def extract_er_match(json_str):
        if pd.isna(json_str):
            return 0.0
        try:
            data = json.loads(json_str)
            return data.get('employer_match', 0.0)
        except:
            return 0.0
    
    contribution_events['er_match'] = contribution_events['value_json'].apply(extract_er_match)
    total_er = contribution_events['er_match'].sum()
    
    print(f"Total employee contributions: ${total_ee:,.2f}")
    print(f"Total employer match: ${total_er:,.2f}")
    print(f"Total plan contributions: ${total_ee + total_er:,.2f}")

# Save the events
output_dir = Path('output/contributions')
output_dir.mkdir(parents=True, exist_ok=True)
contribution_events.to_parquet(output_dir / 'contribution_events_2025.parquet')
```

This demonstrates how to generate contribution events based on employee data and employer match rules.