"""
Plan rules orchestration module for run_one_year package.

Handles all business logic related to plan rules, including:
- Eligibility determination
- Enrollment processing
- Contribution rate changes
"""
import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

from cost_model.plan_rules import eligibility, enrollment, contributions
from cost_model.state.schema import EMP_ID
from .utils import dbg


def run_all_plan_rules(
    prev_snapshot: pd.DataFrame,
    all_events: List[pd.DataFrame],
    hazard_cfg: Dict[str, Any],
    as_of: pd.Timestamp,
    prev_as_of: pd.Timestamp,
    year: int,
) -> pd.DataFrame:
    """
    Orchestrates the execution of all plan rules and generates related events.
    
    Args:
        prev_snapshot: Previous year's snapshot DataFrame
        all_events: List of event DataFrames from previous years
        hazard_cfg: Configuration parameters for the simulation
        as_of: Current timestamp
        prev_as_of: Previous timestamp
        year: Current simulation year
        
    Returns:
        DataFrame containing all plan rule events
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[PLAN_RULES YR={year}] Running all plan rules")
    
    # Run eligibility checks
    eligibility_df = eligibility.run(
        snapshot=prev_snapshot, 
        as_of=as_of,
        log_level="debug"
    )
    
    # Create eligibility events if needed
    eligibility_events_df = eligibility.run_events(
        eligibility_df=eligibility_df,
        prev_snapshot=prev_snapshot,
        as_of=as_of
    )
    
    # Run enrollment
    enrollment_df = enrollment.run(
        snapshot=prev_snapshot,
        eligibility_df=eligibility_df,
        as_of=as_of
    )
    
    # Process contribution increases
    contrib_increase_df = contributions.run_increases(
        snapshot=prev_snapshot,
        eligibility_df=eligibility_df,
        enrollment_df=enrollment_df,
        as_of=as_of,
        prev_as_of=prev_as_of
    )
    
    # Process proactive decreases
    proactive_decrease_df = contributions.run_proactive_decreases(
        snapshot=prev_snapshot,
        eligibility_df=eligibility_df,
        enrollment_df=enrollment_df,
        as_of=as_of
    )
    
    # Collect all plan rule events
    plan_rule_events = []
    for evt_df in [eligibility_events_df, contrib_increase_df, proactive_decrease_df]:
        if evt_df is not None and not evt_df.empty:
            plan_rule_events.append(evt_df)
    
    # Combine all plan rule events
    if plan_rule_events:
        combined_events = pd.concat(plan_rule_events, ignore_index=True)
        logger.info(f"[PLAN_RULES YR={year}] Generated {len(combined_events)} plan rule events")
        return combined_events
    
    # Return empty DataFrame if no events
    return pd.DataFrame()
