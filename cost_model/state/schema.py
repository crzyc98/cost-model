"""Centralized schema constants for cost_model state and events.

This module defines event type constants, employee column constants,
 snapshot column ordering, and pandas dtypes. All other modules
 should import from here instead of redefining their own versions.

## QuickStart

To use the centralized schema constants programmatically:

```python
import pandas as pd
from cost_model.state.schema import (
    # Event type constants
    EVT_HIRE, EVT_TERM, EVT_COMP, EVT_CONTRIB,
    
    # Employee column constants
    EMP_ID, EMP_HIRE_DATE, EMP_BIRTH_DATE, EMP_ROLE,
    EMP_GROSS_COMP, EMP_TERM_DATE, EMP_DEFERRAL_RATE, EMP_TENURE,
    
    # Schema definitions
    SNAPSHOT_COLS, SNAPSHOT_DTYPES, EVENT_COLS
)

# Create a DataFrame with the correct schema
df = pd.DataFrame(columns=SNAPSHOT_COLS)
df = df.astype(SNAPSHOT_DTYPES)
df.index.name = EMP_ID
print(f"Created empty snapshot with {len(SNAPSHOT_COLS)} columns")

# Access column names consistently
print(f"Employee ID column: {EMP_ID}")
print(f"Hire date column: {EMP_HIRE_DATE}")
print(f"Termination date column: {EMP_TERM_DATE}")

# Use event type constants for filtering
events_df = pd.read_parquet('data/events.parquet')
hire_events = events_df[events_df['event_type'] == EVT_HIRE]
term_events = events_df[events_df['event_type'] == EVT_TERM]
comp_events = events_df[events_df['event_type'] == EVT_COMP]

print(f"Found {len(hire_events)} hire events")
print(f"Found {len(term_events)} termination events")
print(f"Found {len(comp_events)} compensation events")

# Create a new DataFrame with the correct schema
new_snapshot = pd.DataFrame({
    EMP_ID: ['EMP001', 'EMP002', 'EMP003'],
    EMP_HIRE_DATE: ['2025-01-15', '2025-02-01', '2025-03-10'],
    EMP_BIRTH_DATE: ['1990-05-12', '1985-08-23', '1992-11-30'],
    EMP_ROLE: ['Engineer', 'Manager', 'Analyst'],
    EMP_GROSS_COMP: [75000.0, 85000.0, 65000.0],
    'active': [True, True, True]
})

# Ensure all required columns exist with correct types
for col in SNAPSHOT_COLS:
    if col not in new_snapshot.columns:
        if col in SNAPSHOT_DTYPES:
            dtype = SNAPSHOT_DTYPES[col]
            if pd.api.types.is_float_dtype(dtype):
                new_snapshot[col] = pd.NA
            elif pd.api.types.is_integer_dtype(dtype):
                new_snapshot[col] = pd.NA
            elif pd.api.types.is_string_dtype(dtype):
                new_snapshot[col] = pd.NA
            elif pd.api.types.is_bool_dtype(dtype):
                new_snapshot[col] = False
            else:
                new_snapshot[col] = pd.NA

# Set the index and convert to proper types
new_snapshot = new_snapshot.set_index(EMP_ID)
new_snapshot = new_snapshot[SNAPSHOT_COLS].astype(SNAPSHOT_DTYPES)

print(f"Created snapshot with {len(new_snapshot)} employees and proper schema")
```

This demonstrates how to use the centralized schema constants to ensure consistency across your codebase when working with snapshots and event logs.
"""
from __future__ import annotations

import pandas as pd
from typing import List

# -----------------------------------------------------------------------------
# Event type constants (re-export from event_log so there is a single source)
# -----------------------------------------------------------------------------
try:
    from .event_log import EVT_HIRE, EVT_TERM, EVT_COMP, EVT_COLA, EVT_PROMOTION, EVT_RAISE, EVT_CONTRIB  # type: ignore
except ImportError:  # pragma: no cover  – stand-alone import safety
    EVT_HIRE = "EVT_HIRE"
    EVT_TERM = "EVT_TERM"
    EVT_COMP = "EVT_COMP"
    EVT_COLA = "EVT_COLA"
    EVT_PROMOTION = "EVT_PROMOTION"
    EVT_RAISE = "EVT_RAISE"
    EVT_CONTRIB = "EVT_CONTRIB"

EVENT_COLS: List[str] = [
    "event_id",
    "event_time",
    "employee_id",
    "event_type",
    "value_num",
    "value_json",
    "meta",
]

# -----------------------------------------------------------------------------
# Employee column constants (import from utils.columns where possible)
# -----------------------------------------------------------------------------
try:
    from ..utils.columns import (
        EMP_ID,
        EMP_HIRE_DATE,
        EMP_BIRTH_DATE,
        EMP_ROLE,
        EMP_GROSS_COMP,
        EMP_TERM_DATE,
        EMP_DEFERRAL_RATE,
        EMP_TENURE,
    )
except ImportError:  # pragma: no cover  – stand-alone import safety
    EMP_ID = "employee_id"
    EMP_HIRE_DATE = "hire_date"
    EMP_BIRTH_DATE = "birth_date"
    EMP_ROLE = "role"
    EMP_GROSS_COMP = "current_comp"
    EMP_TERM_DATE = "term_date"
    EMP_DEFERRAL_RATE = "employee_deferral_rate"
    EMP_TENURE = "tenure_years"

# -----------------------------------------------------------------------------
# Snapshot schema definition
# -----------------------------------------------------------------------------
SNAPSHOT_COLS = [
    EMP_ID,
    EMP_HIRE_DATE,
    EMP_BIRTH_DATE,
    EMP_ROLE,
    EMP_GROSS_COMP,
    EMP_TERM_DATE,
    "active",
    EMP_DEFERRAL_RATE,
    "tenure_band",
    EMP_TENURE,
]

SNAPSHOT_DTYPES = {
    EMP_ID: pd.StringDtype(),
    EMP_HIRE_DATE: "datetime64[ns]",
    EMP_BIRTH_DATE: "datetime64[ns]",
    EMP_ROLE: pd.StringDtype(),
    EMP_GROSS_COMP: pd.Float64Dtype(),
    EMP_TERM_DATE: "datetime64[ns]",
    "active": pd.BooleanDtype(),
    EMP_DEFERRAL_RATE: pd.Float64Dtype(),
    "tenure_band": pd.StringDtype(),
    EMP_TENURE: "float64",
}

__all__ = [
    # events
    "EVT_HIRE",
    "EVT_TERM",
    "EVT_COMP",
    "EVT_COLA",
    "EVT_PROMOTION",
    "EVT_RAISE",
    "EVT_CONTRIB",
    "EVENT_COLS",
    # employee cols
    "EMP_ID",
    "EMP_HIRE_DATE",
    "EMP_BIRTH_DATE",
    "EMP_ROLE",
    "EMP_GROSS_COMP",
    "EMP_TERM_DATE",
    "EMP_DEFERRAL_RATE",
    "EMP_TENURE",
    # snapshot
    "SNAPSHOT_COLS",
    "SNAPSHOT_DTYPES",
]
