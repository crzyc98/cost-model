"""
Functions to build a complete workforce snapshot from event logs.
Provides full rebuild capability for bootstrapping or testing.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

from .constants import (
    EMP_ID, EMP_HIRE_DATE, EMP_BIRTH_DATE, EMP_ROLE, EMP_GROSS_COMP, 
    EMP_TERM_DATE, EMP_ACTIVE, EMP_DEFERRAL_RATE, EMP_TENURE, 
    EMP_TENURE_BAND, SNAPSHOT_COLS, SNAPSHOT_DTYPES, 
    EVT_HIRE, EVT_TERM, EVT_COMP
)
from .helpers import get_first_event, get_last_event, ensure_columns_and_types, validate_snapshot
from .details import extract_hire_details
from .tenure import compute_tenure

logger = logging.getLogger(__name__)

def _empty_snapshot() -> pd.DataFrame:
    """
    Creates an empty snapshot DataFrame with the correct schema.
    
    Returns:
        Empty DataFrame with proper columns and dtypes
    """
    empty_df = pd.DataFrame(columns=SNAPSHOT_COLS)
    empty_df = empty_df.astype(SNAPSHOT_DTYPES)
    empty_df.set_index(EMP_ID, inplace=True)
    return empty_df

def build_full(events: pd.DataFrame, snapshot_year: int) -> pd.DataFrame:
    """
    Builds a complete employee snapshot from the beginning based on all events provided.
    Recommended only for bootstrapping or testing on smaller datasets.

    Args:
        events: DataFrame containing all relevant events, expected to conform
                to EVENT_COLS and schema.
        snapshot_year: Calendar year whose 31-Dec is the *as-of* date for tenure calcs.

    Returns:
        A pandas DataFrame representing the snapshot state, indexed by EMP_ID.
    """
    # Return empty, correctly typed DF if no events
    if events.empty:
        logger.warning("build_full called with empty events DataFrame.")
        return _empty_snapshot()

    logger.info("Building full snapshot from %d events…", len(events))

    # 1. Sort events chronologically
    events = events.sort_values(["event_time", "event_type"], ascending=[True, True])

    # 2. Get the first hire event for each employee (contains original/static details)
    first_hires = get_first_event(events, EVT_HIRE)
    if first_hires.empty:
        logger.warning("No hire events found. Returning empty snapshot.")
        return _empty_snapshot()

    # 3. Extract hire dates from first hire events
    hire_details = pd.DataFrame({
        EMP_HIRE_DATE: first_hires.set_index(EMP_ID)["event_time"]
    })

    # 4. Extract employee demographic details (role, birth_date) from hire events
    employee_details = extract_hire_details(first_hires)
    
    # 5. Get most recent compensation values
    last_comp = get_last_event(events, EVT_COMP)
    if not last_comp.empty:
        comp_df = pd.DataFrame({
            EMP_GROSS_COMP: last_comp.set_index(EMP_ID)["value_num"]
        })
    else:
        comp_df = pd.DataFrame(columns=[EMP_GROSS_COMP])
        logger.warning("No compensation events found.")

    # 6. Get termination dates from termination events
    last_term = get_last_event(events, EVT_TERM)
    if not last_term.empty:
        term_df = pd.DataFrame({
            EMP_TERM_DATE: last_term.set_index(EMP_ID)["event_time"]
        })
    else:
        term_df = pd.DataFrame(columns=[EMP_TERM_DATE])
        logger.debug("No termination events found.")

    # 7. Combine all the data, using hire_details as base
    dfs = [hire_details, employee_details, comp_df, term_df]
    snapshot_df = pd.concat(dfs, axis=1, join="outer")
    
    # 8. Set the active flag based on termination dates
    snapshot_df[EMP_ACTIVE] = snapshot_df[EMP_TERM_DATE].isna()
    
    # 9. Compute tenure (years) and tenure_band
    as_of = pd.Timestamp(f"{snapshot_year}-12-31")
    snapshot_df = compute_tenure(
        df=snapshot_df, 
        as_of=as_of,
        hire_date_col=EMP_HIRE_DATE,
        out_tenure_col=EMP_TENURE,
        out_band_col=EMP_TENURE_BAND
    )
    
    # 10. Add EMP_ID as a column (in addition to being the index)
    snapshot_df[EMP_ID] = snapshot_df.index.astype(str)
    
    # 11. Ensure all required columns and proper dtypes
    final_cols = SNAPSHOT_COLS.copy()
    output_dtypes = SNAPSHOT_DTYPES.copy()
    
    # Select and order columns
    snapshot_df = ensure_columns_and_types(snapshot_df)
    
    # Validate snapshot
    snapshot_df = validate_snapshot(snapshot_df)
    
    # Ensure index name is set
    snapshot_df.index.name = EMP_ID
    
    logger.info(f"Full snapshot built. Shape: {snapshot_df.shape}")
    return snapshot_df
