"""Auto-increase module for generating automatic contribution rate increase events.

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
"""

import pandas as pd
from cost_model.config.plan_rules import AutoIncreaseConfig
from cost_model.utils.columns import EMP_ID, EMP_DEFERRAL_RATE
from typing import List
import uuid


def run(
    snapshot: pd.DataFrame,
    events: pd.DataFrame,
    as_of: pd.Timestamp,
    cfg: AutoIncreaseConfig,
) -> List[pd.DataFrame]:
    """
    For employees who’ve been enrolled (or are auto-enrolled) and haven’t opted out,
    generate a single EVT_AUTO_INCREASE event at `as_of` that bumps their deferral rate
    by cfg.increase_rate, capped at cfg.cap_rate.
    """
    EVT_ENROLL = "EVT_ENROLL"
    EVT_AUTO_ENROLL = "EVT_AUTO_ENROLL"
    EVT_OPT_OUT = "EVT_OPT_OUT"
    EVT_AUTO_INCREASE = "EVT_AUTO_INCREASE"

    EVENT_COLS = [
        "event_id", "event_time", EMP_ID, "event_type",
        "value_num", "value_json", "meta"
    ]

    current_snapshot = snapshot.copy()
    if EMP_ID in current_snapshot.columns and current_snapshot.index.name != EMP_ID:
        current_snapshot = current_snapshot.set_index(EMP_ID)
    elif current_snapshot.index.name != EMP_ID:
        if EMP_ID not in current_snapshot.columns:
             raise ValueError(f"{EMP_ID} column not found in snapshot and not set as index for auto_increase.")

    if not {EMP_ID, "event_type", "event_time"}.issubset(events.columns):
        return []
        
    event_dates = events["event_time"].dt.date
    as_of_date = as_of.date()
    
    enrolled_ids = set(
        events.loc[
            (events["event_type"].isin([EVT_ENROLL, EVT_AUTO_ENROLL]))
            & (event_dates <= as_of_date),
            EMP_ID,
        ].astype(str)
    )
    opted_out_ids = set(
        events.loc[
            (events["event_type"] == EVT_OPT_OUT) & (event_dates <= as_of_date),
            EMP_ID,
        ].astype(str)
    )
    already_increased_ids = set(
        events.loc[
            (events["event_type"] == EVT_AUTO_INCREASE)
            & (event_dates == as_of_date),
            EMP_ID,
        ].astype(str)
    )
    candidates = enrolled_ids - opted_out_ids - already_increased_ids

    out_events = []
    for emp in candidates:
        if emp not in current_snapshot.index:
            continue
        
        if EMP_DEFERRAL_RATE not in current_snapshot.columns:
            continue 
            
        old_rate = float(current_snapshot.loc[emp, EMP_DEFERRAL_RATE])
        new_rate = min(old_rate + cfg.increase_pct, cfg.cap)
        if new_rate <= old_rate:
            continue
        
        event_data = {
            "event_id": str(uuid.uuid4()),
            "event_time": as_of,
            EMP_ID: emp,
            "event_type": EVT_AUTO_INCREASE,
            "value_num": new_rate,
            "value_json": pd.io.json.dumps({"old_rate": old_rate, "new_rate": new_rate}),
            "meta": None,
        }
        out_events.append(event_data)

    if not out_events:
        return []
        
    df_out = pd.DataFrame(out_events, columns=EVENT_COLS)
    return [df_out]
