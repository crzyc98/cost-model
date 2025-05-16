# cost_model/state/event_log.py
"""
Handles loading, saving, and appending events to the central event log.

The event log is stored as a Parquet file.

## QuickStart

To work with event logs programmatically:

```python
import pandas as pd
from pathlib import Path
from cost_model.state.event_log import (
    load_log, append_events, save_log, create_event,
    EVT_HIRE, EVT_COMP, EVT_TERM, EVENT_COLS
)

# Create a new event log or load an existing one
log_path = Path('data/event_log.parquet')
event_log = load_log(log_path)
print(f"Loaded event log with {len(event_log)} events")

# Create new events
new_events = []

# Add a hire event
hire_event = create_event(
    event_time=pd.Timestamp('2025-01-15'),
    employee_id='EMP001',
    event_type=EVT_HIRE,
    value_num=75000.0,  # Starting compensation
    value_json='{"role": "Engineer", "birth_date": "1990-05-12"}',
    meta='Initial hire'
)
new_events.append(hire_event)

# Add a compensation change event
comp_event = create_event(
    event_time=pd.Timestamp('2025-06-01'),
    employee_id='EMP001',
    event_type=EVT_COMP,
    value_num=80000.0,  # New compensation
    meta='Mid-year adjustment'
)
new_events.append(comp_event)

# Add a termination event
term_event = create_event(
    event_time=pd.Timestamp('2025-09-30'),
    employee_id='EMP001',
    event_type=EVT_TERM,
    meta='Voluntary termination'
)
new_events.append(term_event)

# Convert events to DataFrame
new_events_df = pd.DataFrame(new_events, columns=EVENT_COLS)

# Append to the existing log
updated_log = append_events(event_log, new_events_df)
print(f"Updated log now has {len(updated_log)} events")

# Filter events by type or employee
hire_events = updated_log[updated_log['event_type'] == EVT_HIRE]
print(f"Found {len(hire_events)} hire events")

employee_events = updated_log[updated_log['employee_id'] == 'EMP001']
print(f"Employee EMP001 has {len(employee_events)} events")

# Sort events chronologically
sorted_events = updated_log.sort_values('event_time')

# Save the updated log
save_log(updated_log, log_path)
print(f"Saved event log to {log_path}")
```

This demonstrates how to create, load, filter, and save event logs, which form the foundation of the event-driven simulation system.
"""

import logging
import pandas as pd
import pyarrow as pa  # Recommended for explicit Parquet schema handling
import pyarrow.parquet as pq
import uuid
from pathlib import Path
from typing import Dict, Optional, Any

# Attempt to import EMP_ID, define fallback if needed (e.g., for standalone testing)
try:
    from ..utils.columns import EMP_ID, SIMULATION_YEAR
except ImportError:
    print(
        "Warning: Could not import EMP_ID from utils.columns. Defaulting to 'employee_id'."
    )
    EMP_ID = "employee_id"

logger = logging.getLogger(__name__)

# --- Event Schema Definition ---

# Define standard event types as constants
EVT_HIRE      = "EVT_HIRE"
EVT_TERM      = "EVT_TERM"
EVT_COMP      = "EVT_COMP"
EVT_CONTRIB   = "EVT_CONTRIB"  # Contribution event type for plan rules/engines
EVT_COLA      = "EVT_COLA"      # cost-of-living adjustment
EVT_PROMOTION = "EVT_PROMOTION" # title or grade change
EVT_RAISE     = "EVT_RAISE"     # ad-hoc merit increase
# Add future retirement event types here, e.g.:
# EVT_ELIG_ACHIEVED = 'elig_achieved'
# EVT_ENROLL_AUTO = 'enroll_auto'
# EVT_DEFERRAL_CHANGE = 'deferral_change'

# Define column names and order
# Using constants where applicable is good practice
EVENT_COLS = [
    "event_id",  # uuid4 as string, pk
    "event_time",  # ns timestamp
    EMP_ID,  # str
    "event_type",  # str (controlled vocabulary like EVT_HIRE)
    "value_num",  # nullable<float64>
    "value_json",  # nullable<string> (JSON blob)
    "meta",  # nullable<string> (free-text, JSON ok)
]

# Define explicit Schema using PyArrow for Parquet consistency
# This helps ensure dtypes are handled correctly, especially nullables.
EVENT_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("event_time", pa.timestamp("ns"), nullable=False),
        pa.field(EMP_ID, pa.string(), nullable=False),
        pa.field("event_type", pa.string(), nullable=False),
        pa.field("value_num", pa.float64(), nullable=True),
        pa.field("value_json", pa.string(), nullable=True),
        pa.field("meta", pa.string(), nullable=True),
    ]
)

# Corresponding Pandas dtypes using nullable types where appropriate
EVENT_PANDAS_DTYPES = {
    "event_id": pd.StringDtype(),
    "event_time": "datetime64[ns]",
    EMP_ID: pd.StringDtype(),
    "event_type": pd.StringDtype(),
    "value_num": pd.Float64Dtype(),  # Pandas nullable Float
    "value_json": pd.StringDtype(),  # Pandas nullable String
    "meta": pd.StringDtype(),  # Pandas nullable String
}

# --- Core Functions ---


def load_log(path: Path) -> pd.DataFrame:
    """
    Loads the event log from a Parquet file.

    Args:
        path: Path object pointing to the Parquet file.

    Returns:
        A pandas DataFrame containing the event log. Returns an empty
        DataFrame with the correct schema if the file does not exist.
    """
    if path.exists() and path.stat().st_size > 0:  # Check size > 0 for robustness
        try:
            logger.debug(f"Loading event log from: {path}")
            # Read with specified Arrow schema to enforce types
            table = pq.read_table(path, schema=EVENT_SCHEMA)
            df = table.to_pandas()
            # Ensure correct Pandas dtypes after loading (sometimes needed)
            df = df.astype(EVENT_PANDAS_DTYPES)
            logger.debug(f"Loaded {len(df)} events.")
            return df
        except Exception as e:
            logger.error(f"Error loading event log from {path}: {e}", exc_info=True)
            # Depending on desired behavior, could raise error or return empty
            # Returning empty allows simulation to potentially start fresh
    else:
        logger.debug(
            f"Event log file not found or empty at {path}. Returning empty DataFrame."
        )

    # Return empty DataFrame with correct columns and types if file doesn't exist or load fails
    empty_df = pd.DataFrame(columns=EVENT_COLS)
    empty_df = empty_df.astype(EVENT_PANDAS_DTYPES)
    return empty_df


def append_events(log: pd.DataFrame, new_events: pd.DataFrame) -> pd.DataFrame:
    """
    Appends new events to an existing event log DataFrame.

    Args:
        log: The existing event log DataFrame.
        new_events: A DataFrame containing new events to append.
                    It MUST have columns compatible with EVENT_COLS.
                    It SHOULD have unique event_ids generated beforehand.

    Returns:
        A new DataFrame containing the combined events.
    """
    if new_events is None or new_events.empty:
        logger.debug("append_events called with no new events.")
        return log  # Return original log if nothing to append

    # Basic validation: Check if essential columns exist in new_events
    missing_cols = [col for col in EVENT_COLS if col not in new_events.columns]
    if missing_cols:
        # This indicates an error in the upstream process generating new_events
        logger.error(
            f"New events DataFrame is missing required columns: {missing_cols}. Cannot append."
        )
        # Depending on desired behavior, could raise error or return original log
        raise ValueError(
            f"New events DataFrame is missing required columns: {missing_cols}"
        )

    # Ensure dtypes of new_events match the standard schema before concat
    try:
        new_events = new_events[EVENT_COLS].astype(EVENT_PANDAS_DTYPES)
    except Exception as e:
        logger.error(f"Error conforming new events to schema: {e}", exc_info=True)
        raise TypeError(
            f"Could not conform new events DataFrame to required schema/dtypes: {e}"
        )

    # Check for event_id collisions (optional but recommended)
    if not log.empty and "event_id" in log.columns and "event_id" in new_events.columns:
        existing_ids = set(log["event_id"])
        colliding_ids = [eid for eid in new_events["event_id"] if eid in existing_ids]
        if colliding_ids:
            logger.error(
                f"Detected {len(colliding_ids)} duplicate event_ids being appended: {colliding_ids[:5]}..."
            )
            # Decide handling: raise error, filter duplicates, allow duplicates?
            raise ValueError(
                f"Duplicate event_ids detected during append: {colliding_ids[:5]}..."
            )

    logger.debug(f"Appending {len(new_events)} new events to log.")
    # Using copy=True is safer default, False might be slightly faster but shares underlying data
    combined_log = pd.concat([log, new_events], ignore_index=True, copy=True)

    # Optionally re-verify dtypes after concat, though usually okay if inputs conform
    # combined_log = combined_log.astype(EVENT_PANDAS_DTYPES)

    return combined_log


def save_log(log: pd.DataFrame, path: Path) -> None:
    """
    Saves the event log DataFrame to a Parquet file.

    Args:
        log: The event log DataFrame to save.
        path: The Path object for the output Parquet file.
    """
    try:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure correct order and types before saving
        log_to_save = log[EVENT_COLS].astype(EVENT_PANDAS_DTYPES)

        # Convert to Arrow Table using the explicit schema before writing
        table = pa.Table.from_pandas(
            log_to_save, schema=EVENT_SCHEMA, preserve_index=False
        )

        logger.debug(f"Saving {len(log_to_save)} events to: {path}")
        pq.write_table(
            table,
            path,
            compression="snappy",  # Common, good balance
            use_dictionary=True,  # Good for low-cardinality string columns like event_type
            write_statistics=True,  # Helps query planners later if needed
        )
        logger.debug("Save complete.")
    except Exception as e:
        logger.error(f"Error saving event log to {path}: {e}", exc_info=True)
        # Re-raise the exception so the calling process knows saving failed
        raise


# --- Helper Functions (Example) ---


def create_event(
    event_time: pd.Timestamp,
    employee_id: str,
    event_type: str,
    value_num: Optional[float] = None,
    value_json: Optional[str] = None,
    meta: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Helper to create a single event dictionary with a new UUID.
    Ensures variant strategy (only one value_* is non-null).
    """
    if value_num is not None and value_json is not None:
        raise ValueError("Only one of value_num or value_json should be provided.")

    return {
        "event_id": str(uuid.uuid4()),
        "event_time": event_time,
        EMP_ID: str(employee_id),  # Ensure EMP_ID is string
        "event_type": event_type,
        "value_num": value_num,
        "value_json": value_json,
        "meta": meta,
        SIMULATION_YEAR: int(event_time.year) if not pd.isna(event_time) else None
    }


# Example usage within another module:
# from .event_log import load_log, append_events, save_log, create_event
#
# log = load_log(path_to_log)
# new_event_list = []
# new_event_list.append(create_event(pd.Timestamp('2025-01-15'), 'E123', EVT_COMP, value_num=60000.0))
# # ... create more events
# if new_event_list:
#   new_events_df = pd.DataFrame(new_event_list)
#   log = append_events(log, new_events_df)
# save_log(log, path_to_log)
