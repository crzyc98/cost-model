"""
Functions for tenure calculations and tenure band assignments.
"""

import pandas as pd
import numpy as np
import logging
from typing import Union, Optional, Dict, Tuple

from .constants import EMP_HIRE_DATE, EMP_TENURE, EMP_TENURE_BAND

logger = logging.getLogger(__name__)

def assign_tenure_band(tenure: Optional[Union[float, int]]) -> Optional[str]:
    """
    Map numeric tenure (years) to a categorical band string.
    
    Args:
        tenure: Employee tenure in years
        
    Returns:
        Tenure band as string, or pd.NA if input is NA
    """
    if pd.isna(tenure):
        return pd.NA
    if tenure < 1:
        return "<1"
    if tenure < 3:
        return "1-3"
    if tenure < 5:
        return "3-5"
    return "5+"

def compute_tenure(
    df: pd.DataFrame, 
    as_of: pd.Timestamp, 
    hire_date_col: str = EMP_HIRE_DATE,
    out_tenure_col: str = EMP_TENURE,
    out_band_col: str = EMP_TENURE_BAND
) -> pd.DataFrame:
    """
    Compute tenure in years and assign tenure bands for a DataFrame.
    
    Args:
        df: DataFrame with hire_date_col
        as_of: Reference date to calculate tenure
        hire_date_col: Column containing hire dates
        out_tenure_col: Column to store computed tenure
        out_band_col: Column to store computed tenure band
        
    Returns:
        DataFrame with added/updated tenure and tenure band columns
    """
    result = df.copy()
    
    # Calculate tenure in years
    hire_dates = pd.to_datetime(result[hire_date_col], errors='coerce')
    tenure_years = (as_of - hire_dates).dt.days / 365.25
    result[out_tenure_col] = tenure_years.round(3)
    
    # Map to tenure bands
    result[out_band_col] = result[out_tenure_col].map(assign_tenure_band).astype(pd.StringDtype())
    
    # Log any rows with missing tenure
    missing_tenure = result[result[out_tenure_col].isna()].index.tolist()
    if missing_tenure:
        logger.warning(
            "Missing tenure for %d employees: %s", 
            len(missing_tenure),
            missing_tenure[:10]  # Only show first 10 to avoid log spam
        )
    
    return result

def apply_tenure(
    df: pd.DataFrame, 
    hire_date_col: str, 
    as_of: pd.Timestamp,
    out_tenure_col: str = EMP_TENURE,
    out_band_col: str = EMP_TENURE_BAND
) -> pd.DataFrame:
    """
    Convenience wrapper for compute_tenure that preserves index structure.
    
    Args:
        df: DataFrame with hire dates
        hire_date_col: Column containing hire dates
        as_of: Reference date for tenure calculation
        out_tenure_col: Column to store computed tenure
        out_band_col: Column to store computed tenure band
        
    Returns:
        DataFrame with added/updated tenure and tenure band columns
    """
    # Compute tenure while preserving the index
    result = compute_tenure(
        df=df,
        as_of=as_of,
        hire_date_col=hire_date_col,
        out_tenure_col=out_tenure_col,
        out_band_col=out_band_col
    )
    
    return result
