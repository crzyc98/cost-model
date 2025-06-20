"""
Functions to incrementally update a workforce snapshot based on new events.
Provides efficient update capability for year-over-year processing.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .constants import (
    EMP_ACTIVE,
    EMP_BIRTH_DATE,
    EMP_DEFERRAL_RATE,
    EMP_EXITED,
    EMP_GROSS_COMP,
    EMP_HIRE_DATE,
    EMP_ID,
    EMP_LEVEL,
    EMP_LEVEL_SOURCE,
    EMP_TENURE,
    EMP_TENURE_BAND,
    EMP_TERM_DATE,
    EVT_COMP,
    EVT_HIRE,
    EVT_TERM,
    SNAPSHOT_COLS,
    SNAPSHOT_DTYPES,
)
from .details import extract_hire_details
from .helpers import ensure_columns_and_types, get_first_event, get_last_event, validate_snapshot
from .tenure import apply_tenure, compute_tenure

logger = logging.getLogger(__name__)


def _apply_new_hires(current: pd.DataFrame, new_events: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Process new hire events and add them to the current snapshot.

    Args:
        current: Current snapshot DataFrame
        new_events: New events DataFrame
        year: Snapshot year

    Returns:
        Updated snapshot DataFrame with new hires added
    """
    # Find hire events for employees not in current snapshot
    hires = get_first_event(new_events, EVT_HIRE)
    new_ids = hires[~hires[EMP_ID].isin(current.index)][EMP_ID].unique()

    # Filter out null/empty employee IDs to prevent duplicate <NA> entries
    original_count = len(new_ids)
    new_ids = new_ids[pd.notna(new_ids)]  # Remove null values
    new_ids = new_ids[new_ids != ""]      # Remove empty strings
    new_ids = new_ids[new_ids != "<NA>"]  # Remove explicit <NA> strings

    if original_count > len(new_ids):
        filtered_count = original_count - len(new_ids)
        logger.warning(f"Filtered out {filtered_count} hire events with null/empty employee IDs")

    if len(new_ids) == 0:
        return current

    logger.debug("%d new hires to append", len(new_ids))

    # Get events only for the new employees
    batch = new_events[new_events[EMP_ID].isin(new_ids)]

    # Process hire events
    first_hire = get_first_event(batch, EVT_HIRE)
    details = extract_hire_details(first_hire)

    # Get last compensation for new hires
    last_comp = (
        get_last_event(batch, EVT_COMP).set_index(EMP_ID)["value_num"].rename(EMP_GROSS_COMP)
    )

    # Get termination events for new hires (if any were terminated immediately)
    last_term = get_last_event(batch, EVT_TERM)
    term_dates = pd.Series(pd.NaT, index=new_ids)
    if not last_term.empty:
        term_dates.update(last_term.set_index(EMP_ID)["event_time"])

    # Create a base dataframe for new hires
    new_hire_base = pd.DataFrame(index=pd.Index(new_ids, name=EMP_ID))
    new_hire_base[EMP_HIRE_DATE] = first_hire.set_index(EMP_ID)["event_time"]

    # Add employee details from hire events (birth date)
    new_hire_base = new_hire_base.join(details, how="left")

    # Add compensation
    new_hire_base[EMP_GROSS_COMP] = last_comp

    # Add termination dates and active status
    new_hire_base[EMP_TERM_DATE] = term_dates
    new_hire_base[EMP_ACTIVE] = new_hire_base[EMP_TERM_DATE].isna()

    # Create employee ID column, ensuring it's a string
    new_hire_base[EMP_ID] = new_hire_base.index.astype(str)

    # Ensure all required columns exist with correct dtypes
    for col, dtype in SNAPSHOT_DTYPES.items():
        if col not in new_hire_base.columns:
            if pd.api.types.is_datetime64_any_dtype(dtype):
                new_hire_base[col] = pd.NaT
            elif dtype == pd.BooleanDtype():
                new_hire_base[col] = pd.NA
            elif pd.api.types.is_numeric_dtype(dtype):
                new_hire_base[col] = np.nan  # Use np.nan for Float64Dtype etc.
            else:
                new_hire_base[col] = pd.NA  # Use pd.NA for StringDtype

    # Calculate tenure and tenure_band for new hires as of snapshot_year
    if not new_hire_base.empty:
        # Use end-of-year of the snapshot, not the max hire date!
        as_of = pd.Timestamp(f"{year}-12-31")
        new_hire_base = apply_tenure(
            df=new_hire_base,
            hire_date_col=EMP_HIRE_DATE,
            as_of=as_of,
            out_tenure_col=EMP_TENURE,
            out_band_col=EMP_TENURE_BAND,
        )

    # Ensure new hires have the correct columns and types
    new_hire_base = ensure_columns_and_types(new_hire_base)

    # Perform pre-concat diagnostics to prevent issues
    logger.info("--- Pre-concat diagnostics for _apply_new_hires ---")
    logger.info(f"current.index.is_unique: {current.index.is_unique}")
    logger.info(f"current.index.hasnans: {current.index.hasnans}")

    logger.info(f"new_df.index.is_unique: {new_hire_base.index.is_unique}")
    logger.info(f"new_df.index.hasnans: {new_hire_base.index.hasnans}")

    # Double-check that there are no overlap between indices
    overlap = new_hire_base.index.intersection(current.index).tolist()
    logger.info(f"Final overlap check (new_df.index.intersection(current.index)): {overlap}")

    # If overlap exists (shouldn't happen), filter out those IDs from new_hire_base
    if overlap:
        logger.warning(f"Unexpected overlap between new hires and existing employees: {overlap}")
        new_hire_base = new_hire_base.loc[~new_hire_base.index.isin(overlap)]

    logger.info("--- End Pre-concat diagnostics ---")

    # Concatenate new hires with current snapshot
    parts = [current, new_hire_base]
    result = pd.concat(parts, verify_integrity=True, copy=False)

    # Post-concat check (should be redundant if pre-concat checks are thorough)
    if not result.index.is_unique:
        # This case should ideally not be reached if pre-concat checks are correct
        post_concat_dups = result.index[result.index.duplicated()].unique().tolist()
        logger.error(
            f"CRITICAL: Duplicate EMP_IDs persisted after concat: {post_concat_dups}. Deduplicating (keep='last'). This indicates a deeper issue."
        )
        result = result[~result.index.duplicated(keep="last")]

    return result


def _apply_existing_updates(
    current: pd.DataFrame, new_events: pd.DataFrame, year: int
) -> pd.DataFrame:
    """
    Process updates for existing employees (comp changes, terminations).

    Args:
        current: Current snapshot DataFrame
        new_events: New events DataFrame
        year: Snapshot year

    Returns:
        Updated snapshot DataFrame
    """
    # Compensation updates
    comp_upd = new_events[new_events["event_type"] == EVT_COMP]
    if not comp_upd.empty:
        last_comp = comp_upd.sort_values("event_time").groupby(EMP_ID).tail(1)
        current.loc[last_comp[EMP_ID], EMP_GROSS_COMP] = last_comp.set_index(EMP_ID)["value_num"]

    # Termination updates
    term_upd = new_events[new_events["event_type"] == EVT_TERM]
    if not term_upd.empty:
        last_term = term_upd.sort_values("event_time").groupby(EMP_ID).tail(1)
        current.loc[last_term[EMP_ID], EMP_TERM_DATE] = last_term.set_index(EMP_ID)["event_time"]
        current[EMP_ACTIVE] = current[EMP_TERM_DATE].isna()

    # Recompute tenure and bands based on the snapshot_year
    as_of = pd.Timestamp(f"{year}-12-31")
    current = apply_tenure(
        df=current,
        hire_date_col=EMP_HIRE_DATE,
        as_of=as_of,
        out_tenure_col=EMP_TENURE,
        out_band_col=EMP_TENURE_BAND,
    )

    return current


def update(
    prev_snapshot: pd.DataFrame, new_events: pd.DataFrame, snapshot_year: int
) -> pd.DataFrame:
    """
    Updates an existing snapshot based on new events that occurred *since*
    the previous snapshot was generated. Designed for efficiency.

    Args:
        prev_snapshot: The snapshot DataFrame from the previous state, indexed by EMP_ID.
                       Expected to conform to SNAPSHOT_COLS/DTYPES.
        new_events: DataFrame containing only the new events since the last snapshot.
                    Expected to conform to EVENT_COLS schema.
        snapshot_year: The year (int) for the new snapshot (used for tenure calculation).

    Returns:
        The updated snapshot DataFrame, indexed by EMP_ID.
    """
    # If no new events, just return a copy of the previous snapshot with correct types
    if new_events.empty:
        return ensure_columns_and_types(prev_snapshot)

    # Start with a copy of the previous snapshot
    current = prev_snapshot.copy()

    # Sort events chronologically
    new_events = new_events.sort_values(["event_time", "event_type"], ascending=[True, True])

    # Process new hires first
    current = _apply_new_hires(current, new_events, snapshot_year)

    # Then apply updates to existing employees
    current = _apply_existing_updates(current, new_events, snapshot_year)

    # Ensure employee ID column exists and has the right type
    current[EMP_ID] = current.index.astype(str)

    # Ensure all columns and proper dtypes
    current = ensure_columns_and_types(current)

    # After all updates are applied, ensure active flag is in sync with term dates
    if EMP_TERM_DATE in current.columns:
        # Set active=False for anyone with a termination date in the past
        current[EMP_ACTIVE] = current[EMP_TERM_DATE].isna()

    # Final validation
    if EMP_ACTIVE in current.columns and EMP_TERM_DATE in current.columns:
        # Check for inconsistencies
        active_with_term = current[current[EMP_ACTIVE] & ~current[EMP_TERM_DATE].isna()]
        if not active_with_term.empty:
            logger.warning(
                f"Found {len(active_with_term)} employees with active=True but non-null termination date. "
                "Fixing inconsistency by setting active=False for these employees."
            )
            current.loc[active_with_term.index, EMP_ACTIVE] = False

        # Also check for terminated employees without a termination date
        term_without_date = current[~current[EMP_ACTIVE] & current[EMP_TERM_DATE].isna()]
        if not term_without_date.empty:
            logger.warning(
                f"Found {len(term_without_date)} employees with active=False but no termination date. "
                "Fixing inconsistency by setting active=True for these employees."
            )
            current.loc[term_without_date.index, EMP_ACTIVE] = True

    # Set index name
    current.index.name = EMP_ID

    # Handle categorical types
    from pandas import CategoricalDtype

    if isinstance(current[EMP_LEVEL_SOURCE].dtype, CategoricalDtype):
        current[EMP_LEVEL_SOURCE] = current[EMP_LEVEL_SOURCE].astype("category")

    return current
