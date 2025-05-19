# /cost_model/projections/runner/year_processor.py
"""
Handles the processing of each year in the projection.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
from numpy.random import Generator
from datetime import datetime

logger = logging.getLogger(__name__)

from cost_model.state.schema import (
    EVT_HIRE, EVT_TERM, EVT_COMP, EVT_COLA, EVT_PROMOTION, EVT_RAISE, EVT_CONTRIB
)
from cost_model.state.snapshot_update import update
from cost_model.projections.hazard import build_hazard_table
from cost_model.engines.run_one_year_engine import run_one_year
from .constants import EVENT_PRIORITY
from .summaries import make_yearly_summaries


def process_year(
    year: int,
    current_snapshot: pd.DataFrame,
    cumulative_log: pd.DataFrame,
    global_params: Dict[str, Any],
    plan_rules: Dict[str, Any],
    rng: np.random.Generator,
    years: List[int],
    census_path: str,
    ee_contrib_event_types: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame], Any, pd.DataFrame]:
    """
    Process a single year in the projection.
    
    Args:
        year: Current year to process
        current_snapshot: Current employee snapshot
        cumulative_log: Cumulative event log
        global_params: Global simulation parameters
        plan_rules: Plan rules configuration
        rng: Random number generator
        years: List of all simulation years
        census_path: Path to census template
        ee_contrib_event_types: Employee contribution event types
        
    Returns:
        Tuple containing:
        - new_snapshot: Updated snapshot after processing
        - updated_cumulative_log: Updated cumulative event log
        - core_summary: Core metrics summary for the year
        - employment_summary: Employment status summary for the year
        - year_eoy_rows: End-of-year snapshot rows
    """
    # 1. Build hazard table
    hazard_table = build_hazard_table(
        [year],  # Pass years as list
        current_snapshot,
        global_params,
        plan_rules
    )
    
    # 2. Run one year simulation
    year_events, _ = run_one_year(
        event_log=cumulative_log,
        prev_snapshot=current_snapshot,
        year=year,
        global_params=global_params,
        plan_rules=plan_rules,
        hazard_table=hazard_table,
        rng=rng,
        census_template_path=census_path,
        rng_seed_offset=0,
        deterministic_term=False
    )
    
    # Define event priority order
    EVENT_PRIORITY = {
        EVT_HIRE: 10,      # Highest priority - must be processed first
        EVT_TERM: 9,       # Next highest - terminations should be processed before other events
        EVT_COMP: 8,       # Compensation changes
        EVT_COLA: 7,       # Cost of Living Adjustments
        EVT_PROMOTION: 6,  # Promotions
        EVT_RAISE: 5,      # Raises
        EVT_CONTRIB: 4     # Contribution events
    }
    
    # 3. Update snapshot with events
    new_snapshot = update(
        current_snapshot,
        year_events,
        year
    )
    
    # 4. Update cumulative log
    updated_cumulative_log = pd.concat([cumulative_log, year_events])
    
    # 5. Get end-of-year snapshot rows and filter out prior year terminations
    year_eoy_rows = new_snapshot.copy()
    if 'employee_termination_date' in year_eoy_rows.columns:
        # Keep active employees and those terminated in the current year
        year_eoy_rows = year_eoy_rows[
            (year_eoy_rows['employee_termination_date'].isna()) | 
            (pd.to_datetime(year_eoy_rows['employee_termination_date']).dt.year == year)
        ]
        logger.info(f"Filtered out {len(new_snapshot) - len(year_eoy_rows)} employees terminated in prior years from {year} snapshot")
    
    # 6. Compute summaries using the filtered snapshot
    # Calculate start_headcount as the number of active employees at the start of the year
    from cost_model.utils.columns import EMP_ACTIVE
    # current_snapshot *includes* prior-year terminations, so filter to only the still-active folks:
    start_headcount = int(current_snapshot[EMP_ACTIVE].sum())
    
    core_summary, employment_summary = make_yearly_summaries(
        year_eoy_rows,  # Use the filtered snapshot here (already filtered to active + current-year terms)
        year_events,
        year,
        start_headcount=start_headcount  # Pass the actual start headcount of active employees
    )
    
    # Convert to tuple at the very end if needed by the calling code
    return_tuple = (new_snapshot, None)
    
    return (
        return_tuple,  # Return as tuple here to match expected signature
        updated_cumulative_log, 
        core_summary, 
        employment_summary, 
        year_eoy_rows
    )
