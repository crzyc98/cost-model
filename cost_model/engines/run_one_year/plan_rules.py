"""
Plan rules orchestration module for run_one_year package.

Handles all business logic related to plan rules, including:
- Eligibility determination
- Enrollment processing
- Contribution rate changes
"""
import logging
import uuid
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
    
    # Get eligibility config from hazard_cfg or use defaults
    min_age = hazard_cfg.get('min_age', 21)
    min_service_months = hazard_cfg.get('min_service_months', 0)
    eligibility_cfg = type('EligibilityConfig', (), {
        'min_age': min_age,
        'min_service_months': min_service_months
    })()
    
    # Run eligibility checks - this returns a list of DataFrames with eligibility events
    eligibility_events = eligibility.run(
        snapshot=prev_snapshot, 
        as_of=as_of,
        cfg=eligibility_cfg
    )
    
    # Convert the list of DataFrames to a single DataFrame
    eligibility_events_df = pd.concat(eligibility_events) if eligibility_events else pd.DataFrame()
    
    # Run enrollment
    enrollment_df = pd.DataFrame()  # Default to empty DataFrame
    if not eligibility_events_df.empty:
        # Convert eligibility_events_df to the format expected by enrollment.run()
        # The enrollment module expects events with specific columns
        enrollment_events = []
        for _, row in eligibility_events_df.iterrows():
            enrollment_events.append({
                'event_id': row.get('event_id', str(uuid.uuid4())),
                'event_time': as_of,
                'employee_id': row[EMP_ID],
                'event_type': 'ELIGIBILITY',
                'value_num': 1.0,
                'value_json': '{}',
                'meta': '{}'
            })
        
        enrollment_events_df = pd.DataFrame(enrollment_events) if enrollment_events else pd.DataFrame()
        
        if not enrollment_events_df.empty:
            # Create a default enrollment config if not provided
            enrollment_cfg = type('EnrollmentConfig', (), {
                'auto_enroll_enabled': hazard_cfg.get('auto_enroll_enabled', False),
                'auto_enroll_rate': hazard_cfg.get('auto_enroll_rate', 0.0),
                'auto_increase_enabled': hazard_cfg.get('auto_increase_enabled', False),
                'auto_increase_rate': hazard_cfg.get('auto_increase_rate', 0.0)
            })()
            
            enrollment_df = enrollment.run(
                snapshot=prev_snapshot,
                events=enrollment_events_df,
                as_of=as_of,
                cfg=enrollment_cfg
            )
    
    # Process contributions
    # Convert events to DataFrame if it's a list
    if isinstance(enrollment_df, list):
        if not enrollment_df:  # If list is empty
            events_df = pd.DataFrame(columns=[EMP_ID, 'event_type', 'event_time'])
        else:
            events_df = pd.concat(enrollment_df, ignore_index=True)
    else:
        events_df = enrollment_df if enrollment_df is not None else pd.DataFrame(columns=[EMP_ID, 'event_type', 'event_time'])
    
    contrib_events = contributions.run(
        snapshot=prev_snapshot,
        events=events_df,
        as_of=as_of,
        cfg=hazard_cfg.get('plan_rules', {}).get('contributions', {})
    )
    
    # For now, we'll use the same events for both increases and proactive decreases
    # since the contributions.run() function handles both cases
    contrib_increase_df = contrib_events
    proactive_decrease_df = pd.DataFrame(columns=contrib_events.columns) if not contrib_events.empty else contrib_events
    
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
