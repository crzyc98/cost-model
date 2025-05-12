# cost_model/projections/event_log.py
"""
Event log module for creating and managing event logs during projections.

## QuickStart

To create and work with event logs programmatically during projections:

```python
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from cost_model.projections.event_log import create_initial_event_log
from cost_model.state.event_log import (
    EVENT_COLS, EVENT_PANDAS_DTYPES, 
    EVT_HIRE, EVT_TERM, EVT_COMP, EVT_CONTRIB,
    create_event, append_events, filter_events
)

# Create an initial event log for the start year
start_year = 2025
initial_event_log = create_initial_event_log(start_year)
print(f"Created initial event log with {len(initial_event_log)} events")

# Save the initial event log
output_dir = Path('output/events')
output_dir.mkdir(parents=True, exist_ok=True)
initial_event_log.to_parquet(output_dir / f'event_log_{start_year}_initial.parquet')

# Create new events for the projection
new_events = []

# Add a hire event
hire_event = create_event(
    employee_id='EMP101',
    event_type=EVT_HIRE,
    event_time=pd.Timestamp(f'{start_year}-03-15'),
    value_num=80000.0,  # Starting salary
    value_json='{"role": "Engineer", "birth_date": "1990-05-12"}'
)
new_events.append(hire_event)

# Add a termination event
term_event = create_event(
    employee_id='EMP001',  # Existing employee ID
    event_type=EVT_TERM,
    event_time=pd.Timestamp(f'{start_year}-06-30')
)
new_events.append(term_event)

# Add a compensation event
comp_event = create_event(
    employee_id='EMP002',  # Existing employee ID
    event_type=EVT_COMP,
    event_time=pd.Timestamp(f'{start_year}-07-01'),
    value_num=85000.0  # New compensation amount
)
new_events.append(comp_event)

# Add a contribution event
contrib_event = create_event(
    employee_id='EMP003',  # Existing employee ID
    event_type=EVT_CONTRIB,
    event_time=pd.Timestamp(f'{start_year}-09-15'),
    value_num=0.06,  # 6% contribution rate
    value_json='{"employer_match": 3000.0}'
)
new_events.append(contrib_event)

# Convert events to DataFrame
new_events_df = pd.DataFrame(new_events)

# Ensure proper column types
for col, dtype in EVENT_PANDAS_DTYPES.items():
    if col in new_events_df.columns:
        new_events_df[col] = new_events_df[col].astype(dtype)

# Combine with initial event log
combined_events = append_events(initial_event_log, new_events_df)
print(f"Combined event log has {len(combined_events)} events")

# Filter events by type and time range
hire_events = filter_events(
    combined_events,
    event_types=[EVT_HIRE],
    start_date=pd.Timestamp(f'{start_year}-01-01'),
    end_date=pd.Timestamp(f'{start_year}-12-31')
)
print(f"Found {len(hire_events)} hire events in {start_year}")

# Save the combined event log
combined_events.to_parquet(output_dir / f'event_log_{start_year}_final.parquet')
```

This demonstrates how to create an initial event log and add various types of events during a projection.
"""
# create_initial_event_log logic
import pandas as pd
import numpy as np
import logging

from cost_model.state.event_log import EVENT_COLS, EVENT_PANDAS_DTYPES, EMP_ID as EVT_EMP_ID # Alias if EMP_ID means something else here
# It's better if all event types are centrally defined. For now, defining used one here if not available.
# from cost_model.state.event_log import EVT_CONTRIB_INCR # Ideal: import from central location
EVT_CONTRIB_INCR = "EVT_CONTRIB_INCR" # Placeholder: define locally if not in cost_model.state.event_log

logger = logging.getLogger(__name__)

def create_initial_event_log(start_year: int) -> pd.DataFrame:
    logger.info(f"Creating initial event log for events up to start of year: {start_year}")
    # These are events that occurred *before* the first simulation year
    # E.g., deferral rates set in the prior year that are active on start_year-01-01
    base_event_data = [
        {"event_id": f"prior_evt_B_{start_year-1}", "event_time": pd.Timestamp(f"{start_year-1}-01-01"), EVT_EMP_ID: "B", "event_type": EVT_CONTRIB_INCR, "value_num": 0.05},
        {"event_id": f"prior_evt_C_{start_year-1}", "event_time": pd.Timestamp(f"{start_year-1}-01-01"), EVT_EMP_ID: "C", "event_type": EVT_CONTRIB_INCR, "value_num": 0.10},
    ]
    event_log_df = pd.DataFrame(base_event_data)

    # Ensure all EVENT_COLS are present and correctly typed
    for col in EVENT_COLS:
        if col not in event_log_df.columns:
            dtype = EVENT_PANDAS_DTYPES.get(col, 'object')
            logger.debug(f"Adding missing event log column: {col} with presumed dtype: {dtype}")
            if pd.api.types.is_datetime64_any_dtype(dtype) or str(dtype).lower().startswith('datetime64'): 
                event_log_df[col] = pd.NaT
            elif str(dtype).lower() == 'boolean' or str(dtype).lower() == 'bool': 
                event_log_df[col] = pd.NA
            elif pd.api.types.is_numeric_dtype(dtype): 
                event_log_df[col] = np.nan
            else: # Default to StringDtype for 'object' or 'string'
                event_log_df[col] = pd.NA
                event_log_df[col] = event_log_df[col].astype(pd.StringDtype())
            
    event_log_df = event_log_df[EVENT_COLS] # Ensure column order
    for col, dtype_str in EVENT_PANDAS_DTYPES.items():
        if col in event_log_df.columns:
            try:
                if str(dtype_str).lower() == 'boolean':
                     event_log_df[col] = event_log_df[col].astype(pd.BooleanDtype())
                elif str(dtype_str).lower() == 'string':
                     event_log_df[col] = event_log_df[col].astype(pd.StringDtype())
                else:
                     event_log_df[col] = event_log_df[col].astype(dtype_str)
            except Exception as e:
                logger.error(f"Failed to cast event log column '{col}' to '{dtype_str}': {e}. Current type: {event_log_df[col].dtype}")
    
    logger.info(f"Initial event log created with {len(event_log_df)} records. Dtypes:\n{event_log_df.dtypes}")
    return event_log_df
