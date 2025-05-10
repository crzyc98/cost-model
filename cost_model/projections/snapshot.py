# cost_model/projections/snapshot.py
# create_initial_snapshot logic
import pandas as pd
import numpy as np
import logging
from pathlib import Path # Though not used directly in func, good for path handling if extended
from typing import Union

from cost_model.state.snapshot import SNAPSHOT_COLS as SNAPSHOT_COL_NAMES, SNAPSHOT_DTYPES
from cost_model.utils.columns import (
    EMP_ID, EMP_HIRE_DATE, EMP_BIRTH_DATE, EMP_ROLE, 
    EMP_GROSS_COMP, EMP_DEFERRAL_RATE # EMP_TERM_DATE is not in census for initial snapshot
) 

logger = logging.getLogger(__name__)

def create_initial_snapshot(start_year: int, census_path: Union[str, Path]) -> pd.DataFrame:
    logger.info(f"Creating initial snapshot for start year: {start_year} from {census_path}")
    # Load census data
    try:
        if isinstance(census_path, str):
            census_path = Path(census_path)
        census_df = pd.read_parquet(census_path)
    except Exception as e:
        logger.error(f"Error reading census Parquet file {census_path}: {e}")
        raise
    
    logger.info(f"Loaded census data with {len(census_df)} records. Columns: {census_df.columns.tolist()}")

    # Data for the snapshot at the beginning of the start_year
    # First, check if the census has termination dates
    has_term_dates = 'employee_termination_date' in census_df.columns
    
    # If termination dates exist, set active status based on them
    if has_term_dates:
        # An employee is active if they have no termination date
        active_status = census_df['employee_termination_date'].isna()
        logger.info(f"Setting active status based on termination dates: {active_status.sum()} active out of {len(census_df)}")
    else:
        # If no termination dates, assume all are active
        active_status = True
        logger.info("No termination dates found in census, assuming all employees are active")
    
    initial_data = {
        EMP_ID: census_df[EMP_ID],
        EMP_HIRE_DATE: pd.to_datetime(census_df[EMP_HIRE_DATE]),
        EMP_BIRTH_DATE: pd.to_datetime(census_df[EMP_BIRTH_DATE]),
        EMP_ROLE: census_df[EMP_ROLE],
        EMP_GROSS_COMP: census_df[EMP_GROSS_COMP],
        'active': active_status,  # Set active status based on termination dates
        EMP_DEFERRAL_RATE: census_df.get(EMP_DEFERRAL_RATE, 0.0), # Default if not present
    }
    # Add EMP_TERM_DATE if it's in SNAPSHOT_COL_NAMES, otherwise it's handled by the loop below
    if 'employee_termination_date' in SNAPSHOT_COL_NAMES: # Assuming EMP_TERM_DATE is 'employee_termination_date'
        initial_data['employee_termination_date'] = pd.NaT

    snapshot_df = pd.DataFrame(initial_data)
    
    # Ensure 'active' column is broadcasted if it was a scalar
    if 'active' in snapshot_df.columns and len(snapshot_df) > 1 and isinstance(snapshot_df['active'].iloc[0], bool):
        snapshot_df['active'] = [snapshot_df['active'].iloc[0]] * len(snapshot_df)

    # Add tenure_band (example calculation)
    current_date_for_tenure = pd.Timestamp(f"{start_year}-01-01")
    snapshot_df['tenure_years'] = (current_date_for_tenure - snapshot_df[EMP_HIRE_DATE]).dt.days / 365.25
    bins = [0, 1, 3, 5, 10, float('inf')] # Example tenure bins (in years)
    labels = ['0-1yr', '1-3yrs', '3-5yrs', '5-10yrs', '10+yrs']
    snapshot_df['tenure_band'] = pd.cut(snapshot_df['tenure_years'], bins=bins, labels=labels, right=False).astype(pd.StringDtype())
    snapshot_df = snapshot_df.drop(columns=['tenure_years'])

    # Ensure all columns in SNAPSHOT_COL_NAMES exist, fill missing ones
    for col in SNAPSHOT_COL_NAMES:
        if col not in snapshot_df.columns:
            logger.warning(f"Snapshot column '{col}' defined in SNAPSHOT_COL_NAMES is missing. Will be added as NA.")
            # Determine appropriate fill value based on SNAPSHOT_DTYPES
            dtype = SNAPSHOT_DTYPES.get(col)
            if pd.api.types.is_datetime64_any_dtype(dtype) or str(dtype).lower().startswith('datetime64'):
                snapshot_df[col] = pd.NaT
            elif pd.api.types.is_bool_dtype(dtype) or str(dtype).lower() in ['bool', 'boolean']:
                snapshot_df[col] = pd.NA
            elif pd.api.types.is_numeric_dtype(dtype):
                snapshot_df[col] = np.nan
            else: # Default to StringDtype for others like 'object' or 'string'
                snapshot_df[col] = pd.NA
                snapshot_df[col] = snapshot_df[col].astype(pd.StringDtype()) # Ensure it's nullable string
    
    # Select and order columns according to SNAPSHOT_COL_NAMES
    # Ensure EMP_ID is present before trying to set it as index
    if EMP_ID not in snapshot_df.columns and EMP_ID in SNAPSHOT_COL_NAMES:
        # This case should ideally be handled by the loop above, but as a safeguard:
        logger.error(f"{EMP_ID} column is missing but listed in SNAPSHOT_COL_NAMES. Adding as NA.")
        snapshot_df[EMP_ID] = pd.NA 
        snapshot_df[EMP_ID] = snapshot_df[EMP_ID].astype(SNAPSHOT_DTYPES.get(EMP_ID, pd.StringDtype()))
    
    snapshot_df = snapshot_df[[col for col in SNAPSHOT_COL_NAMES if col in snapshot_df.columns]]
    
    if EMP_ID in snapshot_df.columns:
        snapshot_df = snapshot_df.set_index(EMP_ID, drop=False)
    else:
        logger.error(f"{EMP_ID} column is missing, cannot set it as index.")
        # Depending on requirements, either raise an error or proceed without index

    # Apply final dtypes
    for col, dtype_str in SNAPSHOT_DTYPES.items():
        if col in snapshot_df.columns:
            try:
                current_col_data = snapshot_df[col]
                if str(dtype_str).lower() == 'boolean':
                    snapshot_df[col] = current_col_data.astype(pd.BooleanDtype())
                elif str(dtype_str).lower() == 'string':
                     snapshot_df[col] = current_col_data.astype(pd.StringDtype())
                # For Int64Dtype, Float64Dtype, etc., pandas handles them directly
                else:
                    snapshot_df[col] = current_col_data.astype(dtype_str)
            except Exception as e:
                logger.error(f"Failed to cast snapshot column '{col}' to '{dtype_str}': {e}. Current type: {snapshot_df[col].dtype}. Data sample: {snapshot_df[col].head().to_dict()}")
                # Potentially skip casting or re-raise depending on strictness

    logger.info(f"Initial snapshot created with {len(snapshot_df)} records. Columns: {snapshot_df.columns.tolist()}")
    logger.debug(f"Initial snapshot dtypes:\n{snapshot_df.dtypes}")
    return snapshot_df
