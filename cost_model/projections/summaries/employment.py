"""
Employment status module: builds per-year snapshots and status summaries.
"""

import pandas as pd
from cost_model.projections.utils import assign_employment_status

def build_employment_status_snapshot(snapshot_df: pd.DataFrame, event_log_df: pd.DataFrame, sim_year: int) -> pd.DataFrame:
    """
    Adds 'employment_status' to the snapshot based on sim_year.
    """
    df = snapshot_df.copy()
    df['employment_status'] = df.apply(lambda row: assign_employment_status(row, sim_year), axis=1)
    return df


def build_employment_status_summary(snapshot_df: pd.DataFrame, event_log_df: pd.DataFrame, sim_year: int) -> dict:
    """
    Returns a summary dict with counts for each employment status.
    """
    snap = build_employment_status_snapshot(snapshot_df, event_log_df, sim_year)
    counts = snap['employment_status'].value_counts().to_dict()
    return {
        'Year': sim_year,
        'Continuous Active': counts.get('Continuous Active', 0),
        'New Hire Active': counts.get('New Hire Active', 0),
        'Experienced Terminated': counts.get('Experienced Terminated', 0),
        'New Hire Terminated': counts.get('New Hire Terminated', 0),
        'Total Terminated': counts.get('Experienced Terminated', 0) + counts.get('New Hire Terminated', 0),
        'Active': counts.get('Continuous Active', 0) + counts.get('New Hire Active', 0)
    }