"""
Validation module for run_one_year package.

Contains functions for validating and ensuring consistency of inputs.
"""
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np

from cost_model.state.schema import (
    EMP_ID, EMP_HIRE_DATE, EMP_BIRTH_DATE, EMP_ROLE, 
    EMP_GROSS_COMP, EMP_TERM_DATE, EMP_LEVEL,
    EMP_TENURE_BAND as TENURE_BAND,
    EMP_ACTIVE as ACTIVE
)

# Required columns with default values for snapshot DataFrame
REQUIRED_SNAPSHOT_DEFAULTS = {
    EMP_ID: None,         # must exist, no default
    EMP_HIRE_DATE: None,  # must exist, no default
    EMP_BIRTH_DATE: None, # must exist, no default
    EMP_ROLE: None,       # must exist, no default
    EMP_GROSS_COMP: None, # must exist, no default
    EMP_TERM_DATE: pd.NaT,
    ACTIVE: True,
    "employee_deferral_rate": 0.0,
    TENURE_BAND: "<1yr",
    "employee_tenure": np.nan,
}


def ensure_snapshot_cols(snapshot: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures the snapshot DataFrame has all required columns with valid defaults.
    
    Args:
        snapshot: Input snapshot DataFrame
        
    Returns:
        DataFrame with all required columns
        
    Raises:
        ValueError: If a required column is missing and has no default
    """
    snap_copy = snapshot.copy()
    
    # Verify or add required columns
    for col, default in REQUIRED_SNAPSHOT_DEFAULTS.items():
        if col not in snap_copy.columns:
            if default is None:
                raise ValueError(f"Required column {col} missing from snapshot")
            snap_copy[col] = default
            logging.info(f"Added missing required column {col} with default {default}")
            
    return snap_copy


def validate_and_extract_hazard_slice(
    hazard_table: pd.DataFrame, 
    year: int
) -> pd.DataFrame:
    """
    Validates hazard table and extracts the slice for the current year.
    
    Args:
        hazard_table: Full hazard table with multiple years
        year: Current simulation year
        
    Returns:
        Hazard table slice for the current year with standardized column names
        
    Raises:
        ValueError: If hazard table is missing required columns or year
    """
    if hazard_table is None or hazard_table.empty:
        raise ValueError("Hazard table is empty or None")
        
    # Check if simulation_year column exists
    if "simulation_year" not in hazard_table.columns:
        raise ValueError("Hazard table missing 'simulation_year' column")
        
    # Extract slice for current year
    hazard_slice = hazard_table[hazard_table["simulation_year"] == year].copy()
    
    if hazard_slice.empty:
        raise ValueError(f"No hazard data found for year {year}")
        
    # Rename employee_role to role if needed
    if "employee_role" in hazard_slice.columns and "role" not in hazard_slice.columns:
        hazard_slice = hazard_slice.rename(columns={"employee_role": "role"})
        logging.debug(f"Renamed hazard_table.employee_role → hazard_table.role")
    
    # Log available roles and tenure bands
    if "role" in hazard_slice.columns:
        roles = list(hazard_slice["role"].unique())
        logging.info(f"[RUN_ONE_YEAR YR={year}] hazard_slice roles: {roles}")
    
    if TENURE_BAND in hazard_slice.columns:
        tenure_bands = list(hazard_slice[TENURE_BAND].unique())
        logging.info(f"[RUN_ONE_YEAR YR={year}] hazard_slice tenure_bands: {tenure_bands}")
        
    return hazard_slice


def nonempty_frames(*frames: Union[pd.DataFrame, List[pd.DataFrame]]) -> List[pd.DataFrame]:
    """
    Filters out empty DataFrames from a list of DataFrames.
    
    Args:
        *frames: Variable number of DataFrames or lists of DataFrames
        
    Returns:
        List of non-empty DataFrames
    """
    result = []
    for frame in frames:
        if isinstance(frame, list):
            # If it's a list, flatten it and filter
            for subframe in frame:
                if subframe is not None and not subframe.empty:
                    result.append(subframe)
        elif frame is not None and not frame.empty:
            result.append(frame)
    return result
