"""
Employment status module: builds per-year snapshots and status summaries.
"""

import pandas as pd
from cost_model.projections.utils import assign_employment_status, filter_prior_terminated
from cost_model.state.event_log import EVT_HIRE, EVT_TERM

def build_employment_status_snapshot(snapshot_df: pd.DataFrame, event_log_df: pd.DataFrame, sim_year: int) -> pd.DataFrame:
    """
    Adds 'employment_status' to the snapshot based on sim_year.
    """
    df = snapshot_df.copy()
    df['employment_status'] = df.apply(lambda row: assign_employment_status(row, sim_year), axis=1)
    return df


def make_yearly_status(prev_snap, eoy_snap, event_log, year):
    """Return the 5-column dict with employment status metrics.
    
    Args:
        prev_snap: The snapshot DataFrame at the start of the year
        eoy_snap: The snapshot DataFrame at the end of the year
        event_log: The cumulative event log DataFrame
        year: The simulation year
        
    Returns:
        A dictionary with employment status metrics
    """
    active_start = int(prev_snap['active'].sum())
    active_end = int(eoy_snap['active'].sum())

    # Slice the log to this plan year
    yr = event_log[pd.to_datetime(event_log['event_time']).dt.year == year]

    hires = set(yr.loc[yr['event_type'] == EVT_HIRE, 'employee_id'])
    term = yr.loc[yr['event_type'] == EVT_TERM]
    nh_terms = term['employee_id'].isin(hires).sum()
    experienced_terms = (~term['employee_id'].isin(hires)).sum()
    nh_actives = len(hires) - nh_terms

    return {
        'Projection Year': year,
        'active_at_year_start': active_start,
        'active_at_year_end': active_end,
        'new_hire_actives': nh_actives,
        'new_hire_terms': nh_terms,
        'experienced_terms': experienced_terms,
    }

def build_employment_status_summary(snapshot_df: pd.DataFrame, event_log_df: pd.DataFrame, sim_year: int) -> dict:
    """Build a summary of employment status metrics for a given simulation year.
    
    Args:
        snapshot_df: The snapshot DataFrame at the end of the simulation year
        event_log_df: The event log DataFrame for the simulation year
        sim_year: The simulation year
        
    Returns:
        A dictionary with employment status metrics
    """
    # This function is maintained for backward compatibility
    # but now delegates to make_yearly_status for more accurate metrics
    
    # For now, we'll use the snapshot_df as both prev_snap and eoy_snap
    # This is not ideal but maintains the function signature
    return make_yearly_status(snapshot_df, snapshot_df, event_log_df, sim_year)