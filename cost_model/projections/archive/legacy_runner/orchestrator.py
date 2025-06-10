"""
[ARCHIVED] Legacy orchestrator for the projection engine.
Superseded by: cost_model.engines.run_one_year.orchestrator

This file is kept for historical reference only. Do not use in new code.
"""

import warnings

warnings.warn(
    "This module is deprecated. Use cost_model.engines.run_one_year.orchestrator instead.",
    DeprecationWarning,
    stacklevel=2,
)

from typing import Any, Dict, List, Tuple

import pandas as pd

from .init import initialize
from .summaries import make_yearly_summaries
from .year_processor import process_year


def run_projection_engine(
    config_ns: Dict[str, Any], initial_snapshot: pd.DataFrame, initial_log: pd.DataFrame
) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Runs the complete projection engine.

    Args:
        config_ns: Configuration namespace
        initial_snapshot: Initial employee snapshot
        initial_log: Initial event log

    Returns:
        Tuple containing:
        - snapshots: Dictionary of yearly snapshots
        - final_snapshot: Final snapshot DataFrame
        - cumulative_log: Final cumulative event log
        - core_summary: Core metrics summary
        - employment_summary: Employment status summary
    """
    # Initialize the projection engine
    params = initialize(config_ns, initial_snapshot, initial_log)

    # Initialize output structures
    snapshots = {}
    cumulative_log = initial_log.copy()
    core_summaries = []
    employment_summaries = []

    # Process each year
    for year in params[3]:  # params[3] contains the years list
        # Process the year
        (new_snap, cum_log, core_sum, emp_sum, eoy_rows) = process_year(
            year,
            snapshots.get(year - 1, initial_snapshot),
            cumulative_log,
            *params,  # Unpack all initialization parameters
        )

        # Update output structures
        snapshots[year] = pd.concat([snapshots.get(year, pd.DataFrame()), eoy_rows])
        cumulative_log = cum_log
        core_summaries.append(core_sum)
        employment_summaries.append(emp_sum)

    # Create final summary DataFrames
    final_snapshot = snapshots[params[3][-1]]  # Last year's snapshot
    core_summary_df = pd.DataFrame(core_summaries)
    employment_summary_df = pd.DataFrame(employment_summaries)

    return (snapshots, final_snapshot, cumulative_log, core_summary_df, employment_summary_df)
