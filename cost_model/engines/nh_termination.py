# cost_model/engines/nh_termination.py
"""
Deterministic New-Hire Termination Logic

This module provides deterministic and stochastic termination logic for new hires, migrated from term.py as part of the hiring/termination flow migration.
"""
import pandas as pd
import numpy as np
import json
from typing import List
from cost_model.state.event_log import EVENT_COLS, EVT_TERM, EVT_COMP, create_event
from cost_model.utils.columns import EMP_ID, EMP_GROSS_COMP, EMP_HIRE_DATE, EMP_TERM_DATE, EMP_TENURE_BAND
from cost_model.state.schema import NEW_HIRE_TERM_RATE
from cost_model.utils.tenure_utils import standardize_tenure_band
import logging

logger = logging.getLogger(__name__)

def random_dates_between(start_dates, end_date, rng: np.random.Generator):
    result = []
    for start in start_dates:
        start = pd.Timestamp(start)
        days = max(0, (end_date - start).days)
        if days == 0:
            result.append(start)
        else:
            offset = rng.integers(0, days + 1)
            result.append(start + pd.Timedelta(days=int(offset)))
    return result

def run_new_hires(
    snapshot: pd.DataFrame,
    hazard_slice: pd.DataFrame,
    rng: np.random.Generator,
    year: int,
    deterministic: bool
) -> List[pd.DataFrame]:
    """
    Terminate only those employees whose hire date is in `year`, using the `new_hire_termination_rate` from hazard_slice.
    """
    as_of = pd.Timestamp(f"{year}-01-01")
    end_of_year = pd.Timestamp(f"{year}-12-31")
    df_nh = snapshot[
        (snapshot[EMP_HIRE_DATE] >= as_of) &
        ((snapshot[EMP_TERM_DATE].isna()) | (snapshot[EMP_TERM_DATE] > as_of))
    ].copy()
    if df_nh.empty:
        return [pd.DataFrame(columns=EVENT_COLS), pd.DataFrame(columns=EVENT_COLS)]
        
    # CRITICAL FIX: Standardize tenure bands to ensure consistent matching with hazard table
    if EMP_TENURE_BAND in df_nh.columns:
        # Log the original tenure bands for debugging
        logger.info(f"[NH-TERM] Original new hire tenure bands: {df_nh[EMP_TENURE_BAND].unique().tolist()}")
        
        # Apply standardization
        df_nh[EMP_TENURE_BAND] = df_nh[EMP_TENURE_BAND].apply(standardize_tenure_band)
        logger.info(f"[NH-TERM] Standardized new hire tenure bands: {df_nh[EMP_TENURE_BAND].unique().tolist()}")
    
    # Standardize hazard slice tenure bands as well to ensure consistency
    hz_slice_copy = hazard_slice.copy()
    if EMP_TENURE_BAND in hz_slice_copy.columns:
        hz_slice_copy[EMP_TENURE_BAND] = hz_slice_copy[EMP_TENURE_BAND].apply(standardize_tenure_band)
        logger.info(f"[NH-TERM] Standardized hazard slice tenure bands: {hz_slice_copy[EMP_TENURE_BAND].unique().tolist()}")
    
    # Get termination rate
    nh_term_rate = hz_slice_copy[NEW_HIRE_TERM_RATE].iloc[0] if NEW_HIRE_TERM_RATE in hz_slice_copy.columns else 0.0
    n = len(df_nh)
    k = min(int(round(n * nh_term_rate)), n)  # Ensure k is not larger than n
    
    # Create a mapping of employee IDs to their row indices for safer lookups
    if k <= 0:
        return [pd.DataFrame(columns=EVENT_COLS), pd.DataFrame(columns=EVENT_COLS)]
    
    # Select k employees to terminate
    if deterministic:
        # For deterministic, select the first k rows
        selected_indices = df_nh.index[:k]
    else:
        # For stochastic, randomly select k indices
        selected_indices = rng.choice(df_nh.index, size=k, replace=False)
    
    # Get the employee IDs and hire dates for the selected employees
    selected_employees = df_nh.loc[selected_indices]
    exit_ids = selected_employees[EMP_ID].values
    loser_hire_dates = selected_employees[EMP_HIRE_DATE].values
    dates = random_dates_between(loser_hire_dates, end_of_year, rng)
    term_events = []
    comp_events = []
    # Create a DataFrame for selected employees with all the data we need
    selected_df = selected_employees.copy()
    selected_df['term_date'] = dates
    
    # Filter out any rows with NA employee IDs before processing
    original_count = len(selected_df)
    
    # First, ensure EMP_ID is a column and not part of the index
    if EMP_ID in selected_df.index.names and EMP_ID not in selected_df.columns:
        selected_df = selected_df.reset_index(level=EMP_ID)
    
    # Filter out NA and invalid employee IDs
    selected_df = selected_df[~selected_df[EMP_ID].isna()].copy()
    
    # Additional check for valid string representation
    def is_valid_employee_id(emp_id):
        try:
            return emp_id is not None and str(emp_id).strip() != ''
        except Exception:
            return False
            
    valid_mask = selected_df[EMP_ID].apply(is_valid_employee_id)
    invalid_count = len(selected_df) - sum(valid_mask)
    selected_df = selected_df[valid_mask].copy()
    
    if invalid_count > 0 or len(selected_df) < original_count:
        logger.warning(
            f"Filtered out {original_count - len(selected_df) + invalid_count} records with invalid employee IDs "
            f"({original_count - len(selected_df)} NA, {invalid_count} invalid format)"
        )
    
    if selected_df.empty:
        logger.warning("No valid employee records to process after filtering NA employee IDs")
        return [pd.DataFrame(columns=EVENT_COLS), pd.DataFrame(columns=EVENT_COLS)]
    
    for i, row in selected_df.iterrows():
        try:
            emp = row[EMP_ID]
            hire_date = row[EMP_HIRE_DATE]
            term_date = row['term_date']
            
            # Skip if we have any NA values in required fields
            if pd.isna(emp) or pd.isna(hire_date) or pd.isna(term_date):
                logger.warning(f"Skipping record with missing required fields: emp_id={emp}")
                continue
                
            tenure_days = int((term_date - hire_date).days)
            comp = row.get(EMP_GROSS_COMP)  # Use .get() to avoid KeyError
            days_worked = (term_date - hire_date).days + 1
            prorated = comp * (days_worked / 365.25) if comp is not None and not pd.isna(comp) else None
            
            # Create termination event
            term_events.append(create_event(
                event_time=term_date,
                employee_id=emp,
                event_type=EVT_TERM,
                value_num=None,
                value_json=json.dumps({
                    "reason": "new_hire_termination",
                    "tenure_days": tenure_days
                }),
                meta=f"New-hire termination for {emp} in {year}"
            ))
            
            # Create compensation event if prorated comp is available
            if prorated is not None:
                comp_events.append(create_event(
                    event_time=term_date,
                    employee_id=emp,
                    event_type=EVT_COMP,
                    value_num=prorated,
                    value_json=None,
                    meta=f"Prorated comp for {emp} ({days_worked} days from hire to term)"
                ))
                
        except Exception as e:
            logger.error(f"Error processing termination for employee {emp if 'emp' in locals() else 'unknown'}: {str(e)}")
            continue
            
    # Create DataFrames for the events
    df_term = pd.DataFrame(term_events, columns=EVENT_COLS) if term_events else pd.DataFrame(columns=EVENT_COLS)
    df_comp = pd.DataFrame(comp_events, columns=EVENT_COLS) if comp_events else pd.DataFrame(columns=EVENT_COLS)
    
    # Sort by event time if there are events
    if not df_term.empty:
        df_term = df_term.sort_values("event_time", ignore_index=True)
    if not df_comp.empty:
        df_comp = df_comp.sort_values("event_time", ignore_index=True)
        
    return [df_term, df_comp]
