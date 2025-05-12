"""Utility helpers for snapshot build/update flows.

## QuickStart

To use the snapshot utility functions programmatically:

```python
import pandas as pd
import json
from cost_model.state.snapshot_utils import (
    get_first_event, get_last_event, parse_hire_payload,
    extract_hire_details, ensure_columns_and_types
)
from cost_model.state.schema import EMP_ID, EVT_HIRE, EVT_COMP, EVT_TERM

# Load an event log
events_df = pd.read_parquet('data/events.parquet')

# Get the first hire event for each employee
first_hires = get_first_event(events_df, EVT_HIRE)
print(f"Found {len(first_hires)} unique employee hire events")

# Get the last compensation event for each employee
last_comp_events = get_last_event(events_df, EVT_COMP)
print(f"Found {len(last_comp_events)} unique employee compensation events")

# Extract employee details from hire events
hire_details = extract_hire_details(first_hires)
print(f"Extracted details for {len(hire_details)} employees")
print(hire_details.head())

# Parse JSON payload from a hire event
hire_event = first_hires.iloc[0]
json_payload = hire_event.get('value_json')
parsed_data = parse_hire_payload(json_payload)
print(f"Parsed hire payload: {parsed_data}")

# Create a snapshot and ensure it has the correct schema
snapshot_df = pd.DataFrame({
    EMP_ID: ['EMP001', 'EMP002', 'EMP003'],
    'employee_hire_date': ['2025-01-15', '2025-02-01', '2025-03-10'],
    'employee_gross_compensation': [75000.0, 85000.0, 65000.0],
    'active': [True, True, True]
})

# Ensure the snapshot has all required columns with correct types
snapshot_df = ensure_columns_and_types(snapshot_df)
print(f"Snapshot has {len(snapshot_df.columns)} columns with correct schema")

# Filter events by employee ID
emp_id = 'EMP001'
employee_events = events_df[events_df[EMP_ID] == emp_id]
print(f"Employee {emp_id} has {len(employee_events)} events")

# Get events in chronological order
chronological_events = employee_events.sort_values('event_time')
for _, event in chronological_events.iterrows():
    print(f"{event['event_time']}: {event['event_type']} - {event['value_num']}")
```

This demonstrates how to use the utility functions to work with events and ensure snapshots have the correct schema.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from .schema import (
    EMP_ID,
    EMP_ROLE,
    EMP_BIRTH_DATE,
    SNAPSHOT_COLS,
    SNAPSHOT_DTYPES,
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Event filtering helpers
# -----------------------------------------------------------------------------

def get_first_event(events: pd.DataFrame, event_type: str) -> pd.DataFrame:  # noqa: N802
    """Return first occurrence of *event_type* per employee."""
    return events[events["event_type"] == event_type].drop_duplicates(subset=EMP_ID, keep="first")

def get_last_event(events: pd.DataFrame, event_type: str) -> pd.DataFrame:  # noqa: N802
    """Return last occurrence of *event_type* per employee."""
    return events[events["event_type"] == event_type].drop_duplicates(subset=EMP_ID, keep="last")

# -----------------------------------------------------------------------------
# JSON parsing helper for hire events
# -----------------------------------------------------------------------------

def parse_hire_payload(value_json: str | None) -> Dict[str, Any]:  # noqa: D401
    """Return dict parsed from *value_json* or empty dict on error."""
    if value_json is None or pd.isna(value_json):
        return {}
    try:
        return json.loads(value_json)
    except (TypeError, json.JSONDecodeError):
        return {}

# -----------------------------------------------------------------------------
# Hire details extraction
# -----------------------------------------------------------------------------

def extract_hire_details(hire_events: pd.DataFrame) -> pd.DataFrame:  # noqa: N802
    """Return DF indexed by EMP_ID with EMP_ROLE & EMP_BIRTH_DATE columns."""
    if hire_events.empty:
        return pd.DataFrame(columns=[EMP_ROLE, EMP_BIRTH_DATE], index=pd.Index([], name=EMP_ID))

    records = []
    for _, evt in hire_events.iterrows():
        emp_id = evt[EMP_ID]
        payload = parse_hire_payload(evt.get("value_json"))
        role = payload.get("role")
        birth_raw = payload.get("birth_date")
        birth_dt = pd.to_datetime(birth_raw, errors="coerce") if birth_raw else pd.NaT
        records.append({EMP_ID: emp_id, EMP_ROLE: role, EMP_BIRTH_DATE: birth_dt})

    df = pd.DataFrame(records).set_index(EMP_ID)
    df = df.astype({EMP_ROLE: pd.StringDtype()})
    return df

# -----------------------------------------------------------------------------
# Column/type enforcement
# -----------------------------------------------------------------------------

def ensure_columns_and_types(df: pd.DataFrame) -> pd.DataFrame:  # noqa: N802
    """Ensure *df* conforms to SNAPSHOT_COLS/DTYPES order & dtypes."""
    for col, dtype in SNAPSHOT_DTYPES.items():
        if col not in df.columns:
            if pd.api.types.is_numeric_dtype(dtype):
                df[col] = np.nan
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                df[col] = pd.NaT
            else:
                df[col] = pd.NA
    df = df[SNAPSHOT_COLS]  # order
    # replace pd.NA in numeric cols
    for col, dtype in SNAPSHOT_DTYPES.items():
        if pd.api.types.is_numeric_dtype(dtype):
            df[col] = df[col].replace({pd.NA: np.nan})
    return df.astype(SNAPSHOT_DTYPES)
