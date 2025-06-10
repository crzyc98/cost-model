# cost_model/state/event_log.py
"""
Handles loading, saving, and appending events to the central event log.
This demonstrates how to create, load, filter, and save event logs, which form the foundation of the event-driven simulation system.
"""

import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import pyarrow as pa  # Recommended for explicit Parquet schema handling
import pyarrow.parquet as pq
from logging_config import get_diagnostic_logger, get_logger

# Import schema constants directly - these are required
from cost_model.state.schema import EMP_ID, SIMULATION_YEAR

logger = get_logger(__name__)
diag_logger = get_diagnostic_logger(__name__)

# --- Event Schema Definition ---

# Define standard event types as constants
EVT_HIRE = "EVT_HIRE"
EVT_TERM = "EVT_TERM"
EVT_NEW_HIRE_TERM = "EVT_NEW_HIRE_TERM"  # New Hire Termination event
EVT_COMP = "EVT_COMP"
EVT_CONTRIB = "EVT_CONTRIB"  # Contribution event type for plan rules/engines
EVT_COLA = "EVT_COLA"  # cost-of-living adjustment
EVT_PROMOTION = "EVT_PROMOTION"  # title or grade change
EVT_RAISE = "EVT_RAISE"  # ad-hoc merit increase
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
    SIMULATION_YEAR,  # int (year the event occurs in)
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
        pa.field(SIMULATION_YEAR, pa.int32(), nullable=True),
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
    SIMULATION_YEAR: pd.Int32Dtype(),  # Pandas nullable Int32
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
            diag_logger.debug(f"Loading event log from: {path}")
            # Read with specified Arrow schema to enforce types
            table = pq.read_table(path, schema=EVENT_SCHEMA)
            df = table.to_pandas()
            # Ensure correct Pandas dtypes after loading (sometimes needed)
            df = df.astype(EVENT_PANDAS_DTYPES)
            diag_logger.debug(f"Loaded {len(df)} events.")
            return df
        except Exception as e:
            logger.error(f"Error loading event log from {path}: {e}", exc_info=True)
            # Depending on desired behavior, could raise error or return empty
            # Returning empty allows simulation to potentially start fresh
    else:
        diag_logger.debug(
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
        diag_logger.debug("append_events called with no new events.")
        return log  # Return original log if nothing to append

    # Basic validation: Check if essential columns exist in new_events
    missing_cols = [col for col in EVENT_COLS if col not in new_events.columns]
    if missing_cols:
        # This indicates an error in the upstream process generating new_events
        logger.error(
            f"New events DataFrame is missing required columns: {missing_cols}. Cannot append."
        )
        # Depending on desired behavior, could raise error or return original log
        raise ValueError(f"New events DataFrame is missing required columns: {missing_cols}")

    # Ensure dtypes of new_events match the standard schema before concat
    try:
        new_events = new_events[EVENT_COLS].astype(EVENT_PANDAS_DTYPES)
    except Exception as e:
        logger.error(f"Error conforming new events to schema: {e}", exc_info=True)
        raise TypeError(f"Could not conform new events DataFrame to required schema/dtypes: {e}")

    # Check for event_id collisions (optional but recommended)
    if not log.empty and "event_id" in log.columns and "event_id" in new_events.columns:
        existing_ids = set(log["event_id"])
        colliding_ids = [eid for eid in new_events["event_id"] if eid in existing_ids]
        if colliding_ids:
            logger.error(
                f"Detected {len(colliding_ids)} duplicate event_ids being appended: {colliding_ids[:5]}..."
            )
            # Decide handling: raise error, filter duplicates, allow duplicates?
            raise ValueError(f"Duplicate event_ids detected during append: {colliding_ids[:5]}...")

    diag_logger.debug(f"Appending {len(new_events)} new events to log.")
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
        table = pa.Table.from_pandas(log_to_save, schema=EVENT_SCHEMA, preserve_index=False)

        diag_logger.debug(f"Saving {len(log_to_save)} events to: {path}")
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
    employee_id: Union[str, None],
    event_type: str,
    value_num: Optional[float] = None,
    value_json: Optional[Union[Dict, str]] = None,
    meta: Optional[Union[Dict, str]] = None,
) -> Dict:
    """
    Helper to create a single event dictionary with a new UUID.
    Ensures variant strategy (only one value_* is non-null).

    Args:
        event_time: Timestamp when the event occurred
        employee_id: String identifier for the employee (must not be None, empty, or NA)
        event_type: Type of the event (e.g., EVT_HIRE, EVT_TERM)
        value_num: Optional numeric value for the event
        value_json: Optional JSON-serializable value for the event
        meta: Optional metadata for the event

    Returns:
        A dictionary representing the event with all required fields

    Raises:
        ValueError: If employee_id is None, empty string, NA, or otherwise invalid
    """
    # Safely check for None, pd.NA, or empty string after conversion
    try:
        # First check for pd.NA or None using pandas' isna() which handles pd.NA properly
        if pd.isna(employee_id):
            error_msg = f"Invalid employee_id: cannot be NA or None. Event type: {event_type}, Time: {event_time}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Convert to string and check if it's empty or just whitespace
        try:
            emp_id_str = str(employee_id).strip()
            if not emp_id_str:
                error_msg = f"Invalid employee_id: cannot be empty or whitespace. Value: '{employee_id}', Event type: {event_type}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Failed to process employee_id: {str(e)}. Value: {employee_id}, Type: {type(employee_id).__name__}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
    except Exception as e:
        # Catch any other unexpected errors during validation
        error_msg = f"Unexpected error validating event data: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg) from e

    # Validate event_type is not empty
    if not event_type or not isinstance(event_type, str):
        error_msg = f"Event type must be a non-empty string, got {event_type}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Ensure only one of value_num or value_json is provided
    if value_num is not None and value_json is not None:
        error_msg = f"Only one of value_num or value_json can be provided. Event type: {event_type}, Employee ID: {employee_id}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Convert value_json to string if it's a dictionary
    if isinstance(value_json, dict):
        try:
            value_json = json.dumps(value_json)
        except (TypeError, ValueError) as e:
            logger.warning(f"Could not serialize value_json to JSON: {e}")
            value_json = str(value_json)

    # Convert meta to string if it's a dictionary
    if isinstance(meta, dict):
        try:
            meta = json.dumps(meta)
        except (TypeError, ValueError) as e:
            logger.warning(f"Could not serialize meta to JSON: {e}")
            meta = str(meta)

    # Convert meta to string if it's not None
    meta_str = str(meta) if meta is not None else None

    # Ensure only one value field is populated
    if value_num is not None and value_json is not None:
        logger.warning(
            f"Both value_num and value_json provided for employee_id={employee_id}, "
            f"event_type={event_type}. Using value_num."
        )
        value_json = None

    # Generate event with simulation year if event_time is valid
    event = {
        "event_id": str(uuid.uuid4()),
        "event_time": event_time,
        EMP_ID: str(employee_id).strip(),  # Ensure string type and trim whitespace
        "event_type": event_type,
        "value_num": float(value_num) if value_num is not None and pd.notna(value_num) else None,
        "value_json": value_json,
        "meta": meta,
    }

    # Add simulation year if event_time is valid
    if not pd.isna(event_time):
        event[SIMULATION_YEAR] = int(event_time.year)

    return event


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
