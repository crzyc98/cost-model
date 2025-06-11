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

    # --- New-hire metrics --------------------------------------------------
    eoy_snap = eoy_snap.copy()
    eoy_snap["hire_year"] = pd.to_datetime(eoy_snap["employee_hire_date"], errors="coerce").dt.year

    is_curr_year_hire = eoy_snap["hire_year"] == year
    nh_actives = int((eoy_snap["active"] & is_curr_year_hire).sum())
    nh_terms = int((~eoy_snap["active"] & is_curr_year_hire).sum())

    # --- Experienced termination metrics ----------------------------------
    # Derive terminations directly from snapshots to avoid dependency on
    # event log coverage discrepancies. An experienced termination is an
    # employee who was active at SOY but is inactive at EOY and was *NOT*
    # hired in the current year.
    EMP_ID = "employee_id"

    soy_active_ids = set(prev_snap.loc[prev_snap["active"], EMP_ID].astype(str))
    eoy_active_ids = set(eoy_snap.loc[eoy_snap["active"], EMP_ID].astype(str))

    # Employees present in SOY active set but absent from EOY active set
    terminated_ids = soy_active_ids - eoy_active_ids

    # Remove new-hire terminations (those hired in current year)
    nh_terminated_ids = set(eoy_snap.loc[~eoy_snap["active"] & is_curr_year_hire, EMP_ID].astype(str))
    experienced_terminated_ids = terminated_ids - nh_terminated_ids

    experienced_terms = len(experienced_terminated_ids)

    # Safety: fall back to event-log count if snapshot-based method yields 0
    if experienced_terms == 0 and not event_log.empty:
        yr_log = event_log[pd.to_datetime(event_log["event_time"]).dt.year == year]
        experienced_terms = int((yr_log["event_type"] == EVT_TERM).sum()) - nh_terms
        experienced_terms = max(0, experienced_terms)

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
