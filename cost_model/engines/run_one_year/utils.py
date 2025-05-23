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


import math

def compute_headcount_targets(start_count: int, survivor_count: int, target_growth: float, nh_term_rate: float):
    """
    Compute headcount targets for the year given growth and new-hire termination rates.

    Args:
        start_count: Number of employees at start of year
        survivor_count: Employees remaining after attrition
        target_growth: Target growth rate (e.g., 0.05 for 5%)
        nh_term_rate: New hire termination rate (e.g., 0.2 for 20%)
    Returns:
        (target_eoy, net_needed, gross_needed)
    """
    target_eoy = int(round(start_count * (1 + target_growth)))
    net_needed = max(target_eoy - survivor_count, 0)
    gross_needed = math.ceil(net_needed / max(1 - nh_term_rate, 1e-9)) if net_needed > 0 else 0
    return target_eoy, net_needed, gross_needed


def test_compute_headcount_targets():
    # Test growing from 100 to 103 employees with 15% new hire termination rate
    start = 100
    target_eoy = 103  # 3% growth
    target_growth = 0.03
    nh_term_rate = 0.15  # 15% new hire termination rate
    
    # Simulate some attrition - let's say we lose 10 people to attrition
    survivors = 90  # 100 - 10
    
    # Calculate hiring needs
    target_eoy, net_needed, gross_needed = compute_headcount_targets(
        start, survivors, target_growth, nh_term_rate
    )
    
    # Verify calculations
    assert target_eoy == 103, f"Target EOY should be 103, got {target_eoy}"
    assert net_needed == 13, f"Net needed should be 13 (103-90), got {net_needed}"
    # Expected gross: 13 / (1 - 0.15) = 15.29 → 16
    assert gross_needed == 16, f"Expected gross_needed=16, got {gross_needed}"
    print("test_compute_headcount_targets passed!")
