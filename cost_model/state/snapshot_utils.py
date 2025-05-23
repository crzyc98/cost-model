# cost_model/state/snapshot_utils.py
"""
Utility helpers for snapshot build/update flows.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Union, List

import numpy as np
import pandas as pd

from cost_model.state.schema import (
    EMP_ID,
    EMP_LEVEL,
    EMP_BIRTH_DATE,
    SNAPSHOT_COLS,
    SNAPSHOT_DTYPES,
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Event filtering helpers
# -----------------------------------------------------------------------------

def get_first_event(events: pd.DataFrame, event_type: Union[str, List[str]]) -> pd.DataFrame:  # noqa: N802
    """Return first occurrence of *event_type* per employee."""
    if isinstance(event_type, str):
        events_of_type = events[events["event_type"] == event_type]
    else:  # it's a list
        events_of_type = events[events["event_type"].isin(event_type)]
    return events_of_type.drop_duplicates(subset=EMP_ID, keep="first")

def get_last_event(events: pd.DataFrame, event_type: Union[str, List[str]]) -> pd.DataFrame:  # noqa: N802
    """Return last occurrence of *event_type* per employee."""
    if isinstance(event_type, str):
        events_of_type = events[events["event_type"] == event_type]
    else:  # it's a list
        events_of_type = events[events["event_type"].isin(event_type)]
    return events_of_type.drop_duplicates(subset=EMP_ID, keep="last")

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
    """Return DF indexed by EMP_ID with EMP_LEVEL & EMP_BIRTH_DATE columns."""
    if hire_events.empty:
        return pd.DataFrame(columns=[EMP_ID, EMP_LEVEL, EMP_BIRTH_DATE]).set_index(EMP_ID)
    
    records = []
    for _, evt in hire_events.iterrows():
        emp_id = evt[EMP_ID]
        payload = parse_hire_payload(evt.get("value_json"))
        
        # Get level from payload or default to 1
        level = payload.get("level", 1)
        
        # Parse birth date
        birth_raw = payload.get("birth_date")
        birth_dt = pd.to_datetime(birth_raw, errors="coerce") if birth_raw else pd.NaT
        
        records.append({
            EMP_ID: emp_id, 
            EMP_LEVEL: level, 
            EMP_BIRTH_DATE: birth_dt
        })
    
    # Create DataFrame with proper types
    df = pd.DataFrame(records)
    df[EMP_LEVEL] = df[EMP_LEVEL].fillna(1).astype('int64')  # Ensure level is integer
    df = df.set_index(EMP_ID)
    
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
