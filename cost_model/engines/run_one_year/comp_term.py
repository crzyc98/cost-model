"""
Compensation and termination module for run_one_year package.

Handles compensation bumps for experienced employees and termination processing.
"""
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np

from cost_model.engines import term
from cost_model.state.event_log import create_event, EVENT_COLS
from cost_model.dynamics.compensation import apply_comp_bump
from cost_model.state.schema import (
    EMP_ID, EMP_GROSS_COMP, EMP_TERM_DATE,
    EMP_ACTIVE as ACTIVE,
    EVT_COMP, EVT_TERM
)
from .utils import dbg


def apply_compensation_and_terminations(
    prev_snapshot: pd.DataFrame,
    hazard_slice: pd.DataFrame,
    global_params: Any,
    year_rng: np.random.Generator,
    deterministic_term: bool,
    year: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies compensation updates and processes terminations for experienced employees.
    
    Args:
        prev_snapshot: Previous year's snapshot DataFrame
        hazard_slice: Hazard table slice for the current year
        global_params: Global configuration parameters
        year_rng: Random number generator
        deterministic_term: Whether to use deterministic terminations
        year: Current simulation year
        
    Returns:
        Tuple containing (events_df, updated_snapshot)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[COMP_TERM YR={year}] Processing compensation and terminations")
    
    # Create a copy of the snapshot for processing
    snapshot = prev_snapshot.copy()
    
    # Get experienced active employee mask
    experienced_mask = snapshot[ACTIVE] == True
    experienced_count = experienced_mask.sum()
    logger.info(f"[YR={year}] SOY experienced mask: {experienced_count} | new hire mask: 0 | total: {len(snapshot)}")
    logger.info(f"[YR={year}] SOY Experienced Active = {experienced_count}")
    
    # 1. Process terminations
    term_events = []
    if experienced_count > 0:
        # Generate termination events for active employees
        term_events_list = term.run(
            snapshot=snapshot.loc[experienced_mask],
            hazard_slice=hazard_slice,
            rng=year_rng,
            deterministic=deterministic_term
        )
        
        if term_events_list:
            # Extend the term_events list with all DataFrames from the returned list
            for df in term_events_list:
                if df is not None and not df.empty:
                    term_events.append(df)
            
            # Calculate total number of terminations
            term_count = sum(len(df) for df in term_events_list if df is not None)
            logger.info(f"[YR={year}] Term events generated: {term_count}")
            
            # Log the terminated employee IDs
            term_ids = []
            for df in term_events_list:
                if df is not None and not df.empty and EMP_ID in df.columns:
                    term_ids.extend(df[EMP_ID].tolist())
            
            if term_ids:
                logger.info(f"[YR={year}] Terminated employee IDs: {term_ids}")
                # Update snapshot with terminations
                snapshot.loc[snapshot[EMP_ID].isin(term_ids), [ACTIVE, EMP_TERM_DATE]] = [False, pd.Timestamp(f"{year}-12-31")]
            
            # Log post-termination stats
            logger.info(f"[YR={year}] Post-term snapshot: {len(snapshot)} unique EMP_IDs, {snapshot.shape[0]} rows")
            dbg(year, "post-term snapshot", snapshot)
            
            active_count = snapshot[ACTIVE].sum()
            logger.info(f"[YR={year}] Post-term Active= {active_count}")
            logger.info(f"[RUN_ONE_YEAR YR={year}] Terminations: {term_count}, survivors active: {active_count}")
    
    # 2. Process compensation bumps
    comp_events_df = pd.DataFrame()
    if experienced_count > 0:
        # Get experienced employees for comp bumps
        experienced_df = prev_snapshot.loc[experienced_mask].copy()
        logger.debug(f"[YR={year}] Experienced slice: {len(experienced_df)} rows; sample IDs: {experienced_df[EMP_ID].head().tolist()}")
        
        # Extract rates from hazard slice
        comp_rates = hazard_slice["comp_raise_pct"].unique()
        mean_comp_rate = comp_rates.mean() if len(comp_rates) > 0 else 0.03
        
        # Log summary stats
        term_rate = hazard_slice['term_rate'].mean() if 'term_rate' in hazard_slice else 0.0
        nh_term_rate = hazard_slice['new_hire_termination_rate'].mean() if 'new_hire_termination_rate' in hazard_slice else 0.0
        logger.info(f"[RUN_ONE_YEAR YR={year}] Rates → term={term_rate:.3f}, comp={mean_comp_rate:.3f}, nh_term={nh_term_rate:.3f}")
        
        # Apply compensation bump
        comp_cols_before = experienced_df[[EMP_ID, EMP_GROSS_COMP]].copy()
        
        # Get pre-bump compensation for logging
        logger.debug(f"[YR={year}] Applying comp bump to {len(experienced_df)} rows")
        
        # Apply the compensation bump
        # First, ensure we don't have NA values in the compensation column
        if experienced_df[EMP_GROSS_COMP].isna().any():
            logger.warning(f"[YR={year}] Found {experienced_df[EMP_GROSS_COMP].isna().sum()} NA values in {EMP_GROSS_COMP}, filling with 0")
            experienced_df[EMP_GROSS_COMP] = experienced_df[EMP_GROSS_COMP].fillna(0.0)
            
        experienced_df = apply_comp_bump(
            df=experienced_df,
            comp_col=EMP_GROSS_COMP,
            rate=mean_comp_rate,
            rng=year_rng,
            log=logger
        )
        
        # Update snapshot with new compensation values
        for idx, row in experienced_df.iterrows():
            emp_id = row[EMP_ID]
            new_comp = row[EMP_GROSS_COMP]
            
            # Find the employee in the snapshot
            emp_idx = snapshot[snapshot[EMP_ID] == emp_id].index
            if not emp_idx.empty:
                snapshot.loc[emp_idx, EMP_GROSS_COMP] = new_comp
        
        # 3. Create compensation events
        comp_events = []
        for _, row in experienced_df.iterrows():
            emp_id = row[EMP_ID]
            after_comp = row[EMP_GROSS_COMP]
            
            # Get before compensation with proper NA handling
            before_comp = 0.0
            if emp_id in comp_cols_before[EMP_ID].values:
                before_comp = comp_cols_before.loc[
                    comp_cols_before[EMP_ID] == emp_id, 
                    EMP_GROSS_COMP
                ].iloc[0]  # Use iloc[0] to get the scalar value
                
                # Convert to float, handling NA/None values
                if pd.isna(before_comp):
                    before_comp = 0.0
                else:
                    before_comp = float(before_comp)
            
            # Ensure after_comp is a float
            after_comp = 0.0 if pd.isna(after_comp) else float(after_comp)
                
            # Create event (always log events, regardless of size)
            event = create_event(
                event_time=pd.Timestamp(f"{year}-01-01"),
                employee_id=emp_id,
                event_type=EVT_COMP,
                value_num=after_comp,
                meta=f"comp_bump;before={before_comp:.2f};after={after_comp:.2f}"
            )
            comp_events.append(event)
        
        if comp_events:
            comp_events_df = pd.DataFrame(comp_events)
            logger.info(f"[YR={year}] Adding {len(comp_events)} compensation bump events to log")
    
    # Combine all events
    all_events = []
    if term_events:
        all_events.extend(term_events)
    if not comp_events_df.empty:
        all_events.append(comp_events_df)
    
    if all_events:
        combined_events = pd.concat(all_events, ignore_index=True)
        return combined_events, snapshot
    
    return pd.DataFrame(), snapshot
