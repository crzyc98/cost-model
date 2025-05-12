# cost_model/projections/event_log.py
"""
Event log module for creating and managing event logs during projections.
QuickStart: see docs/cost_model/projections/event_log.md
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
