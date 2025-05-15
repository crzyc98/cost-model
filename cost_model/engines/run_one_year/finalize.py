"""
Finalization module for run_one_year package.

Handles final event collection, new hire terminations, and snapshot finalization.
"""
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np

from cost_model.state.schema import EMP_ID, EMP_TERM_DATE, EMP_ACTIVE as ACTIVE
from cost_model.engines.run_one_year_engine.utils import dbg


def apply_new_hire_terminations(
    snap_with_hires: pd.DataFrame,
    hazard_slice: pd.DataFrame,
    year_rng: np.random.Generator,
    year: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies terminations to newly hired employees.
    
    Args:
        snap_with_hires: Snapshot with new hires
        hazard_slice: Hazard table slice for the current year
        year_rng: Random number generator
        year: Current simulation year
        
    Returns:
        Tuple containing (termination_events_df, updated_snapshot)
    """
    logger = logging.getLogger(__name__)
    
    # Get the snapshot with hires for processing
    snapshot = snap_with_hires.copy()
    
    # Determine new hire termination rate
    nh_term_rate = hazard_slice['new_hire_termination_rate'].mean()
    
    # Get new hire mask (use job_level_source if available)
    new_hire_mask = snapshot['job_level_source'] == 'hire' if 'job_level_source' in snapshot.columns else pd.Series(False, index=snapshot.index)
    new_hire_count = new_hire_mask.sum()
    
    if new_hire_count == 0:
        return pd.DataFrame(), snapshot
    
    # Determine number of new hires to terminate
    nh_term_count = int(new_hire_count * nh_term_rate)
    
    if nh_term_count == 0:
        return pd.DataFrame(), snapshot
    
    # Get new hire subset
    new_hires = snapshot.loc[new_hire_mask]
    
    # Randomly select new hires for termination
    term_indices = year_rng.choice(
        new_hires.index, 
        size=nh_term_count, 
        replace=False
    )
    
    # Create termination events
    term_events = []
    for idx in term_indices:
        emp_id = snapshot.loc[idx, EMP_ID]
        
        # Calculate termination date (randomly during the year)
        days_into_year = year_rng.integers(1, 365)
        term_date = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=days_into_year)
        
        # Update snapshot with termination
        snapshot.loc[idx, EMP_TERM_DATE] = term_date
        snapshot.loc[idx, ACTIVE] = False
        
        # Create termination event
        event = {
            "event_id": f"TERM_{year}_{emp_id}",
            "event_time": term_date,
            "employee_id": emp_id,
            "event_type": "EVT_TERM",
            "value_num": np.nan,
            "value_json": None,
            "meta": "new_hire_term"
        }
        term_events.append(event)
    
    # Create events dataframe
    term_events_df = pd.DataFrame(term_events)
    logger.info(f"[RUN_ONE_YEAR YR={year}] New-hire terminations: {len(term_events_df)}")
    
    return term_events_df, snapshot


def build_full_event_log(
    plan_rule_events: pd.DataFrame,
    comp_term_events: pd.DataFrame,
    hires_events: pd.DataFrame,
    nh_term_events: pd.DataFrame,
    year: int
) -> pd.DataFrame:
    """
    Combines all event types into a complete event log for the year.
    
    Args:
        plan_rule_events: Plan rule events
        comp_term_events: Compensation and termination events
        hires_events: Hiring events
        nh_term_events: New hire termination events
        year: Current simulation year
        
    Returns:
        DataFrame containing all events for the year
    """
    logger = logging.getLogger(__name__)
    
    # Collect all event dataframes
    all_events = []
    
    for events_df in [plan_rule_events, comp_term_events, hires_events, nh_term_events]:
        if events_df is not None and not events_df.empty:
            all_events.append(events_df)
    
    if not all_events:
        logger.warning(f"[YR={year}] No events generated for the year")
        return pd.DataFrame()
    
    # Combine all events
    combined_df = pd.concat(all_events, ignore_index=True)
    
    # Sort by event time
    if not combined_df.empty and "event_time" in combined_df.columns:
        combined_df = combined_df.sort_values("event_time")
        
    logger.info(f"[YR={year}] Final event log contains {len(combined_df)} events")
    return combined_df


def finalize_snapshot(
    snapshot: pd.DataFrame,
    year: int
) -> pd.DataFrame:
    """
    Performs final updates and checks on the snapshot.
    
    Args:
        snapshot: Current snapshot
        year: Current simulation year
        
    Returns:
        Finalized snapshot DataFrame
    """
    logger = logging.getLogger(__name__)
    
    # Make a copy for processing
    final_snapshot = snapshot.copy()
    
    # Ensure all required columns exist
    required_cols = [EMP_ID, EMP_TERM_DATE, ACTIVE]
    for col in required_cols:
        if col not in final_snapshot.columns:
            if col == ACTIVE:
                final_snapshot[col] = True
            elif col == EMP_TERM_DATE:
                final_snapshot[col] = pd.NaT
            else:
                logger.error(f"[YR={year}] Required column {col} missing from snapshot")
    
    # Log final snapshot statistics
    active_count = final_snapshot[ACTIVE].sum() if ACTIVE in final_snapshot.columns else 0
    logger.info(f"[YR={year}] Final snapshot: {len(final_snapshot[EMP_ID].unique())} unique EMP_IDs, {final_snapshot.shape[0]} rows")
    logger.info(f"[YR={year}] Post-NH-term    = {active_count}")
    
    return final_snapshot
