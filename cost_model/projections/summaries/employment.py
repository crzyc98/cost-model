"""
Employment status module: builds per-year snapshots and status summaries.
"""

import pandas as pd

from cost_model.projections.utils import assign_employment_status, filter_prior_terminated
from cost_model.state.event_log import EVT_HIRE, EVT_TERM


def build_employment_status_snapshot(
    snapshot_df: pd.DataFrame, event_log_df: pd.DataFrame, sim_year: int
) -> pd.DataFrame:
    """
    Adds 'employment_status' to the snapshot based on sim_year.
    """
    df = snapshot_df.copy()
    df["employment_status"] = df.apply(lambda row: assign_employment_status(row, sim_year), axis=1)
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
    active_start = int(prev_snap["active"].sum())
    active_end = int(eoy_snap["active"].sum())

    # Use robust snapshot-based counting (same logic as make_yearly_summaries fix)
    # Add hire year to end-of-year snapshot for cohort analysis
    eoy_snap = eoy_snap.copy()
    eoy_snap["hire_year"] = pd.to_datetime(eoy_snap["employee_hire_date"]).dt.year

    # Identify current year hires (both active and terminated are in snapshot!)
    is_curr_year_hire = eoy_snap["hire_year"] == year
    curr_year_active = eoy_snap["active"] & is_curr_year_hire
    curr_year_terminated = (~eoy_snap["active"]) & is_curr_year_hire

    # Count from snapshot (ground truth)
    nh_actives = int(curr_year_active.sum())
    nh_terms = int(curr_year_terminated.sum())

    # Calculate experienced terminations from events
    if not event_log.empty:
        yr = event_log[pd.to_datetime(event_log["event_time"]).dt.year == year]
        total_terms = int((yr["event_type"] == EVT_TERM).sum())
        experienced_terms = max(0, total_terms - nh_terms)
    else:
        experienced_terms = 0

    # Import canonical column names from schema
    from cost_model.state.schema import SUMMARY_YEAR

    return {
        SUMMARY_YEAR: year,  # Use canonical year column name from schema
        "active_at_year_start": active_start,
        "active_at_year_end": active_end,
        "new_hire_actives": nh_actives,
        "new_hire_terms": nh_terms,
        "experienced_terms": experienced_terms,
    }


def build_employment_status_summary(
    snapshot_df: pd.DataFrame, event_log_df: pd.DataFrame, sim_year: int
) -> dict:
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
