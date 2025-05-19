"""
Handles computation of various metrics and summaries for the projection.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
from datetime import datetime

from cost_model.utils.columns import (
    EMP_ACTIVE, EMP_CONTR, EMP_DEFERRAL_RATE, EMP_LEVEL, EMP_TENURE_BAND,
    EMPLOYER_CORE_CONTRIB, EMPLOYER_MATCH_CONTRIB, EVENT_TYPE,
    EVT_HIRE, EVT_TERM, EVT_CONTRIB, SIMULATION_YEAR, EMP_GROSS_COMP
)


def make_yearly_summaries(snapshot: pd.DataFrame, 
                         year_events: pd.DataFrame, 
                         year: int,
                         start_headcount: int = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compute yearly summaries for core metrics and employment status.
    
    Args:
        snapshot: Current year's snapshot
        year_events: Events that occurred during the year
        year: Current year
        
    Returns:
        Tuple containing:
        - core_summary: Dictionary of core metrics
        - employment_summary: Dictionary of employment status metrics
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Debug log the input types
    logger.debug(f"make_yearly_summaries called with snapshot type: {type(snapshot)}")
    if hasattr(snapshot, 'shape'):
        logger.debug(f"Snapshot shape: {snapshot.shape}")
    if hasattr(snapshot, 'columns'):
        logger.debug(f"Snapshot columns: {list(snapshot.columns)}")
    
    # Handle case where snapshot is a tuple instead of DataFrame
    if isinstance(snapshot, tuple):
        logger.warning(f"Snapshot is a tuple, converting to DataFrame. Length: {len(snapshot)}")
        
        # Debug log the tuple contents
        for i, item in enumerate(snapshot):
            logger.debug(f"Tuple item {i} type: {type(item)}")
            if hasattr(item, 'shape'):
                logger.debug(f"Item {i} shape: {item.shape}")
            if hasattr(item, 'columns'):
                logger.debug(f"Item {i} columns: {item.columns.tolist()}")
        
        # If the first element is a DataFrame, use it directly
        if len(snapshot) > 0 and isinstance(snapshot[0], pd.DataFrame):
            snapshot = snapshot[0]
        # If the first element is a Series, convert it to a DataFrame
        elif len(snapshot) > 0 and hasattr(snapshot[0], 'to_frame'):
            snapshot = snapshot[0].to_frame().T
        # If we have two elements and the second is a dict of DataFrames
        elif len(snapshot) == 2 and isinstance(snapshot[1], dict):
            # Try to find the main DataFrame in the second element
            for key, value in snapshot[1].items():
                if isinstance(value, pd.DataFrame):
                    snapshot = value
                    logger.debug(f"Using DataFrame from dict key: {key}")
                    break
            else:
                # If no DataFrame found, use the first element
                snapshot = pd.DataFrame([snapshot[0]])
        else:
            # Fallback: convert the first element to a DataFrame
            snapshot = pd.DataFrame([snapshot[0]] if len(snapshot) > 0 else [{}])
    
    # Ensure required columns exist with default values if missing
    required_columns = {
        EMP_ACTIVE: False,
        EMP_CONTR: 0.0,
        EMP_GROSS_COMP: 0.0,
        EMPLOYER_CORE_CONTRIB: 0.0,
        EMPLOYER_MATCH_CONTRIB: 0.0,
        EMP_DEFERRAL_RATE: 0.0,
        EMP_LEVEL: '',
        EMP_TENURE_BAND: ''
    }
    
    for col, default in required_columns.items():
        if col not in snapshot.columns:
            logger.warning(f"Adding missing column '{col}' with default value: {default}")
            snapshot[col] = default
    # Core metrics with robust DataFrame handling
    try:
        # Debug log the snapshot structure
        logger.debug(f"Snapshot columns: {snapshot.columns.tolist()}")
        logger.debug(f"Snapshot dtypes: {snapshot.dtypes}")
        logger.debug(f"Snapshot head (2):\n{snapshot.head(2).to_string()}")
        
        # Safely calculate active headcount
        if EMP_ACTIVE in snapshot.columns:
            active_mask = snapshot[EMP_ACTIVE].astype(bool)
            active_headcount = int(active_mask.sum())
        else:
            logger.warning(f"Column '{EMP_ACTIVE}' not found in snapshot")
            active_headcount = 0
        
        # Calculate participation rate safely
        if EMP_DEFERRAL_RATE in snapshot.columns:
            participation_rate = (len(snapshot[snapshot[EMP_DEFERRAL_RATE] > 0]) / 
                               max(1, active_headcount)) * 100  # Avoid division by zero
        else:
            logger.warning(f"Column '{EMP_DEFERRAL_RATE}' not found in snapshot")
            participation_rate = 0.0
        
        # Calculate employees with compensation > 0
        employees_with_comp = snapshot[snapshot[EMP_GROSS_COMP] > 0]
        comp_headcount = len(employees_with_comp)
        
        # Calculate average compensation only for those with compensation > 0
        avg_comp = float(employees_with_comp[EMP_GROSS_COMP].mean() if comp_headcount > 0 else 0.0)
        
        core_summary = {
            'Projection Year': year,  # Changed from SIMULATION_YEAR to match reporting.py
            SIMULATION_YEAR: year,    # Keep both for backward compatibility
            "headcount": comp_headcount,  # Only count employees with compensation
            "active_headcount": active_headcount,
            "total_contributions": float(snapshot[EMP_CONTR].sum() if EMP_CONTR in snapshot.columns else 0.0),
            "employer_contributions": float((snapshot[EMPLOYER_CORE_CONTRIB].sum() if EMPLOYER_CORE_CONTRIB in snapshot.columns else 0.0) + 
                                        (snapshot[EMPLOYER_MATCH_CONTRIB].sum() if EMPLOYER_MATCH_CONTRIB in snapshot.columns else 0.0)),
            "total_employee_gross_compensation": float(snapshot[EMP_GROSS_COMP].sum()),
            "avg_employee_gross_compensation": avg_comp,  # Average of only employees with compensation
            "avg_deferral_rate": float(snapshot[EMP_DEFERRAL_RATE].mean() if EMP_DEFERRAL_RATE in snapshot.columns else 0.0),
            "participation_rate": participation_rate
        }
        
    except Exception as e:
        logger.error(f"Error calculating core metrics: {str(e)}")
        logger.error(f"Snapshot type: {type(snapshot)}")
        logger.error(f"Snapshot shape: {getattr(snapshot, 'shape', 'N/A')}")
        logger.error(f"Snapshot columns: {getattr(snapshot, 'columns', 'N/A')}")
        logger.error(f"Snapshot head (2): {getattr(snapshot, 'head', lambda _: 'N/A')(2) if hasattr(snapshot, 'head') else 'N/A'}")
        raise
    
    # Employment status metrics
    # Active headcount at year end - count active employees in the filtered snapshot
    # snapshot here is the filtered year_eoy_rows (active + current-year terms)
    actives_at_year_end = int(snapshot[EMP_ACTIVE].sum())
    
    # Use the provided start headcount if available, otherwise calculate it
    if start_headcount is not None:
        actives_at_year_start = start_headcount
    else:
        # Fall back to the original calculation method if start_headcount not provided
        hires_count = len(year_events[year_events[EVENT_TYPE] == EVT_HIRE]) if not year_events.empty else 0
        terms_count = len(year_events[year_events[EVENT_TYPE] == EVT_TERM]) if not year_events.empty else 0
        actives_at_year_start = max(0, actives_at_year_end + terms_count - hires_count)
        logger.warning(f"start_headcount not provided, calculated as {actives_at_year_start} based on end + terms - hires")
    
    # Gross hires = every EVT_HIRE event
    hires = int((year_events[EVENT_TYPE] == EVT_HIRE).sum() if not year_events.empty else 0)
    
    # All terminations
    terminations = int((year_events[EVENT_TYPE] == EVT_TERM).sum() if not year_events.empty else 0)
    
    # New hire terminations: check meta field for new-hire mentions
    mask_new_hire_term = None
    if not year_events.empty:
        if 'meta' in year_events.columns:
            # Look for "new-hire" or "new hire" in the meta field
            mask_new_hire_term = ((year_events[EVENT_TYPE] == EVT_TERM) & 
                                  year_events['meta'].str.contains('new[- ]hire', case=False, na=False))
        elif 'value_json' in year_events.columns:
            # Fall back to the original method if meta isn't available
            mask_new_hire_term = ((year_events[EVENT_TYPE] == EVT_TERM) &
                                 year_events['value_json'].str.contains('new_hire_termination', na=False))
        
    new_hire_terminations = int(mask_new_hire_term.sum()) if mask_new_hire_term is not None else 0
    
    # How many of our hires survived
    new_hire_actives = hires - new_hire_terminations
    
    # Calculate experienced terminations (total terminations - new hire terminations)
    experienced_terminations = max(0, terminations - new_hire_terminations)
    
    employment_summary = {
        'Projection Year': year,
        SIMULATION_YEAR: year,
        "actives_at_year_start": actives_at_year_start,
        "actives_at_year_end": actives_at_year_end,
        "terminations": terminations,
        "new_hires": hires,
        "new_hire_terminations": new_hire_terminations,
        "new_hire_actives": new_hire_actives,
        "experienced_terminations": experienced_terminations,
        "by_level": snapshot[EMP_LEVEL].value_counts().to_dict() if not snapshot.empty and EMP_LEVEL in snapshot.columns else {},
        "by_tenure_band": snapshot[EMP_TENURE_BAND].value_counts().to_dict() if not snapshot.empty and EMP_TENURE_BAND in snapshot.columns else {}
    }
    
    return core_summary, employment_summary
