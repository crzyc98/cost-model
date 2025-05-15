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
