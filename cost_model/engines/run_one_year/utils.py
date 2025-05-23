"""
Utility functions for run_one_year package.

Contains debugging, logging, and other helper functions.
"""
import logging
from typing import Dict, Optional, Any, List, Union
import pandas as pd


def dbg(year: int, label: str, df: pd.DataFrame) -> None:
    """
    Debug helper for logging DataFrame stats during simulation.
    
    Args:
        year: Current simulation year
        label: Label for the debug message
        df: DataFrame to analyze
    """
    active_ct = 0
    if "active" in df.columns:
        active_ct = df["active"].sum()
    
    unique_ids = len(df["employee_id"].unique()) if "employee_id" in df.columns else 0
    
    logging.debug(
        f"[DBG YR={year}] {label:25s} rows={df.shape[0]:5d} "
        f"uniq_ids={unique_ids:5d} act={active_ct}"
    )


def compute_headcount_targets(start_count: int, soy_count: int, target_growth: float, nh_term_rate: float):
    """
    Compute headcount targets for the year given growth and new-hire termination rates.

    Args:
        start_count: Number of employees at start of year
        soy_count: Same as start_count (for compatibility)
        target_growth: Target growth rate (e.g., 0.05 for 5%)
        nh_term_rate: New hire termination rate (e.g., 0.2 for 20%)
    Returns:
        (target_eoy, net_hires, gross_hires)
    """
    # End-of-year headcount target
    target_eoy = int(round(start_count * (1 + target_growth)))
    net_hires = target_eoy - start_count
    gross_hires = int(round(net_hires / (1 - nh_term_rate))) if nh_term_rate < 1.0 else 0
    return target_eoy, net_hires, gross_hires
