# cost_model/projections/snapshot.py
"""
Snapshot module for creating and updating workforce snapshots during projections.
QuickStart: see docs/cost_model/projections/snapshot.md
"""
# create_initial_snapshot logic
import pandas as pd
import numpy as np
import logging
from pathlib import Path # Though not used directly in func, good for path handling if extended
from typing import Union, Dict, Tuple, List

from cost_model.state.snapshot.constants import SNAPSHOT_COLS as SNAPSHOT_COL_NAMES, SNAPSHOT_DTYPES
from cost_model.state.job_levels.loader import ingest_with_imputation
from cost_model.state.schema import (
    EMP_ID, EMP_HIRE_DATE, EMP_BIRTH_DATE, EMP_ROLE, 
    EMP_GROSS_COMP, EMP_DEFERRAL_RATE, EMP_TENURE_BAND, EMP_TENURE,
    EMP_TERM_DATE, EMP_ACTIVE, EMP_LEVEL, EMP_LEVEL_SOURCE, EMP_EXITED, SIMULATION_YEAR
) # Import all required column constants

logger = logging.getLogger(__name__)

def create_initial_snapshot(start_year: int, census_path: Union[str, Path]) -> pd.DataFrame:
    """
    Create the initial employee snapshot from census data.
    
    Args:
        start_year: The starting year for the simulation
        census_path: Path to the census data file (Parquet format)
        
    Returns:
        DataFrame containing the initial employee snapshot
        
    Raises:
        FileNotFoundError: If the census file doesn't exist
        ValueError: If the census data is invalid or missing required columns
    """
    logger.info("Creating initial snapshot for start year: %d from %s", start_year, str(census_path))
    
    # Load census data
    try:
        census_path = Path(census_path) if isinstance(census_path, str) else census_path
        if not census_path.exists():
            raise FileNotFoundError(f"Census file not found: {census_path}")
            
        logger.debug("Reading census data from %s", census_path)
        
        # Try to read the file as Parquet first, fall back to CSV if that fails
        try:
            # Try reading as Parquet first
            census_df = pd.read_parquet(census_path)
            logger.info("Successfully loaded census data from Parquet file")
        except Exception as parquet_error:
            # If Parquet read fails, try CSV
            try:
                logger.info("Parquet read failed, attempting to read as CSV")
                file_extension = census_path.suffix.lower()
                
                if file_extension == '.csv':
                    census_df = pd.read_csv(census_path)
                    logger.info("Successfully loaded census data from CSV file")
                else:
                    # If not a recognized format, raise the original Parquet error
                    logger.error("Census file is neither a valid Parquet nor a CSV file.")
                    raise parquet_error
            except Exception as csv_error:
                logger.error("Failed to read census file as either Parquet or CSV: %s", str(csv_error), exc_info=True)
                raise csv_error
        
        if census_df.empty:
            logger.warning("Census data is empty. Creating empty snapshot.")
            return pd.DataFrame(columns=SNAPSHOT_COL_NAMES).astype(SNAPSHOT_DTYPES)
            
    except Exception as e:
        logger.error("Error reading census file %s: %s", str(census_path), str(e), exc_info=True)
        raise
    
    logger.info("Loaded census data with %d records. Columns: %s", len(census_df), census_df.columns.tolist())
    
    # Handle column name standardization
    logger.info("Standardizing column names in census data")
    
    # Map common variations of column names to the expected names
    column_mapping = {
        # Standard mappings from schema.py
        'ssn': EMP_ID,  # Map ssn to employee_id
        'employee_ssn': EMP_ID,  # Map employee_ssn to employee_id
        'birth_date': EMP_BIRTH_DATE,
        'hire_date': EMP_HIRE_DATE,
        'termination_date': EMP_TERM_DATE,
        'gross_compensation': EMP_GROSS_COMP,
        'role': EMP_ROLE,
        
        # Additional mappings specific to our CSV structure
        'employee_birth_date': EMP_BIRTH_DATE,
        'employee_hire_date': EMP_HIRE_DATE,
        'employee_termination_date': EMP_TERM_DATE,
        'employee_gross_compensation': EMP_GROSS_COMP,
        'employee_role': EMP_ROLE,
        'employee_deferral_rate': EMP_DEFERRAL_RATE
    }
    
    # Apply column mapping
    census_df = census_df.rename(columns=column_mapping)
    
    # Log the column mapping results
    logger.info(f"After column mapping, available columns: {census_df.columns.tolist()}")
    
    # Ensure required columns exist
    required_columns = [EMP_ID, EMP_HIRE_DATE, EMP_BIRTH_DATE, EMP_GROSS_COMP]
    missing_columns = [col for col in required_columns if col not in census_df.columns]
    
    if missing_columns:
        # If employee_id is missing but we have employee_ssn, create employee_id from it
        if EMP_ID in missing_columns and 'employee_ssn' in census_df.columns:
            logger.info(f"Creating {EMP_ID} from employee_ssn column")
            census_df[EMP_ID] = census_df['employee_ssn']
            missing_columns.remove(EMP_ID)
    
    if missing_columns:
        error_msg = f"Census data is missing required columns: {', '.join(missing_columns)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Filter out employees terminated before or at the projection start year
    term_col = EMP_TERM_DATE if EMP_TERM_DATE in census_df.columns else None
    
    if term_col:
        # Convert termination dates to datetime, coercing errors to NaT
        census_df[term_col] = pd.to_datetime(census_df[term_col], errors='coerce')
        
        # Only include employees with no termination date or termination after Jan 1 of the start year
        before_filter = len(census_df)
        census_df = census_df[
            census_df[term_col].isna() | 
            (census_df[term_col] > pd.Timestamp(f'{start_year}-01-01'))
        ]
        after_filter = len(census_df)
        
        logger.info(
            "Filtered out %d employees terminated before or at %d-01-01. Remaining: %d",
            before_filter - after_filter, start_year, after_filter
        )
    else:
        logger.info("No '%s' column found in census. Assuming all employees are active.", EMP_TERM_DATE)
    
    # Always add simulation_year column at creation
    census_df[SIMULATION_YEAR] = start_year
    logger.debug("Set simulation_year=%d for all employees", start_year)

    # Data for the snapshot at the beginning of the start_year
    # First, check if the census has termination dates
    has_term_dates = EMP_TERM_DATE in census_df.columns
    
    # If termination dates exist, set active status based on them
    if has_term_dates:
        # An employee is active if they have no termination date
        active_status = census_df[EMP_TERM_DATE].isna()
        logger.info(f"Setting active status based on termination dates: {active_status.sum()} active out of {len(census_df)}")
    else:
        # If no termination dates, assume all are active
        active_status = pd.Series(True, index=census_df.index)
        logger.info("No termination dates found in census, assuming all employees are active")
    
    # Initialize all required columns from SNAPSHOT_COL_NAMES with appropriate defaults
    initial_data = {
        EMP_ID: census_df[EMP_ID].astype('string'),
        EMP_HIRE_DATE: pd.to_datetime(census_df[EMP_HIRE_DATE]),
        EMP_BIRTH_DATE: pd.to_datetime(census_df[EMP_BIRTH_DATE]),
        EMP_GROSS_COMP: census_df[EMP_GROSS_COMP].astype('float64'),
        EMP_TERM_DATE: pd.NaT,  # Will be updated below if termination dates exist
        EMP_ACTIVE: active_status,
        EMP_DEFERRAL_RATE: census_df.get(EMP_DEFERRAL_RATE, 0.0).astype('float64'),
        # Initialize with default values that will be calculated below
        EMP_TENURE: 0.0,
        EMP_TENURE_BAND: pd.NA,
        # Initialize with default values for employee level and role
        EMP_LEVEL: pd.Series([pd.NA] * len(census_df), dtype='Int64'),
        EMP_LEVEL_SOURCE: pd.Series([pd.NA] * len(census_df), dtype='string'),
        EMP_EXITED: False,  # Will be updated based on termination status
        EMP_ROLE: pd.NA,  # Will be set to a default value if not in census
        SIMULATION_YEAR: start_year  # Set simulation year for all employees
    }
    
    # Set tenure_band based on employee_tenure
    if EMP_TENURE in initial_data:
        # Define tenure bands
        bands = {
            '0-1': (0, 1),
            '1-3': (1, 3),
            '3-5': (3, 5),
            '5+': (5, float('inf'))
        }
        
        # Create tenure_band column
        def get_tenure_band(tenure: float) -> str:
            for band, (min_val, max_val) in bands.items():
                if min_val <= tenure < max_val:
                    return band
            return '5+'
        
        # Convert initial_data to DataFrame first
        snapshot_df = pd.DataFrame(initial_data, columns=SNAPSHOT_COL_NAMES)
        
        # Ensure all required columns are present and have the correct type
        for col in SNAPSHOT_COL_NAMES:
            if col not in snapshot_df.columns:
                snapshot_df[col] = pd.NA
        
        # Convert to the correct dtypes
        snapshot_df = snapshot_df.astype(SNAPSHOT_DTYPES)
        
        # Compute tenure band using schema constant
        snapshot_df[EMP_TENURE_BAND] = snapshot_df[EMP_TENURE].apply(get_tenure_band)
        initial_data[EMP_TENURE_BAND] = snapshot_df[EMP_TENURE_BAND].values
        
        # Remove any stray 'tenure_band' columns (if present)
        # (No longer needed: all code now uses EMP_TENURE_BAND)
        if 'tenure_band' in snapshot_df.columns and EMP_TENURE_BAND != 'tenure_band':
            snapshot_df = snapshot_df.drop(columns=['tenure_band'])
        # Defensive: ensure only EMP_TENURE_BAND is present, never 'tenure_band'
        assert 'tenure_band' not in snapshot_df.columns or EMP_TENURE_BAND == 'tenure_band', (
            f"Stray 'tenure_band' column found. All code should use EMP_TENURE_BAND: {EMP_TENURE_BAND}")
    
    # Set termination dates if they exist in the census
    if has_term_dates:
        initial_data[EMP_TERM_DATE] = census_df[term_col]
        initial_data[EMP_EXITED] = ~census_df[term_col].isna()
        
        # Infer job levels using compensation data if not already present
        if EMP_LEVEL in census_df.columns and not census_df[EMP_LEVEL].isna().all():
            initial_data[EMP_LEVEL] = census_df[EMP_LEVEL].astype('Int64')
            logger.info(f"Using existing '{EMP_LEVEL}' column from census data")
            
            # Set the level source
            initial_data[EMP_LEVEL_SOURCE] = 'census'
        else:
            logger.info("Inferring job levels from compensation data...")
            # Create a temporary DataFrame with compensation data
            temp_df = pd.DataFrame({
                EMP_ID: census_df[EMP_ID],
                EMP_GROSS_COMP: census_df[EMP_GROSS_COMP]
            })
            
            # Ensure job levels are initialized
            from cost_model.state.job_levels import init_job_levels
            init_job_levels()
            
            # Infer job levels using the job levels module
            target_level_col = EMP_LEVEL
            temp_df = ingest_with_imputation(temp_df, comp_col=EMP_GROSS_COMP, target_level_col=target_level_col)
            
            # Ensure we have the required columns in temp_df
            required_cols = [EMP_ID, target_level_col, EMP_LEVEL_SOURCE]
            missing_cols = [col for col in required_cols if col not in temp_df.columns]
            if missing_cols:
                logger.error(f"Missing required columns in temp_df: {missing_cols}")
                raise ValueError(f"Failed to create job levels column in temp_df")
            
            # Update initial_data dictionary directly from temp_df
            initial_data[target_level_col] = temp_df[target_level_col].values
            initial_data[EMP_LEVEL_SOURCE] = temp_df[EMP_LEVEL_SOURCE].values
            
            # Log the inferred job levels
            logger.info(f"Inferred job levels: {temp_df[target_level_col].value_counts(dropna=False).to_dict()}")
            logger.debug(f"Job levels distribution: {temp_df[target_level_col].value_counts(dropna=False)}")
    else:
        logger.warning("Failed to infer job levels. Defaulting to level 1.")
        initial_data[EMP_LEVEL] = pd.Series([1] * len(initial_data[EMP_ID]), dtype='Int64')
        initial_data[EMP_LEVEL_SOURCE] = pd.Series(['default'] * len(initial_data[EMP_ID]), dtype='string')
    
    # Set employee role if it exists in census, otherwise use default 'Regular'
    if EMP_ROLE in census_df.columns:
        initial_data[EMP_ROLE] = census_df[EMP_ROLE].astype('string')
        logger.info(f"Using '{EMP_ROLE}' column from census data")
    else:
        logger.warning(f"No {EMP_ROLE} column found in census. Initializing with default role 'Regular'.")
        initial_data[EMP_ROLE] = 'Regular'

    # Create snapshot_df with all required columns, ensuring simulation_year is included
    snapshot_df = pd.DataFrame(initial_data)
    
    # Explicitly ensure simulation_year is present and set to start_year
    if SIMULATION_YEAR not in snapshot_df.columns:
        logger.debug(f"Adding missing {SIMULATION_YEAR} column to snapshot_df with value {start_year}")
        snapshot_df[SIMULATION_YEAR] = start_year
    
    # Ensure 'active' column is broadcasted if it was a scalar
    if 'active' in snapshot_df.columns and len(snapshot_df) > 1 and isinstance(snapshot_df['active'].iloc[0], bool):
        snapshot_df['active'] = [snapshot_df['active'].iloc[0]] * len(snapshot_df)

    # Calculate tenure in years
    current_date_for_tenure = pd.Timestamp(f"{start_year}-01-01")
    snapshot_df[EMP_TENURE] = (current_date_for_tenure - snapshot_df[EMP_HIRE_DATE]).dt.days / 365.25
    
    # Calculate tenure bands
    bins = [0, 1, 3, 5, 10, float('inf')]  # Tenure bins in years
    labels = ['0-1yr', '1-3yrs', '3-5yrs', '5-10yrs', '10+yrs']
    snapshot_df[EMP_TENURE_BAND] = pd.cut(
        snapshot_df[EMP_TENURE], 
        bins=bins, 
        labels=labels, 
        right=False
    ).astype(pd.StringDtype())
    
    # Ensure all columns in SNAPSHOT_COL_NAMES exist with correct dtypes
    for col in SNAPSHOT_COL_NAMES:
        if col not in snapshot_df.columns:
            logger.warning(f"Snapshot column '{col}' defined in SNAPSHOT_COL_NAMES is missing. Will be added with default value.")
            # Get the expected dtype from SNAPSHOT_DTYPES
            dtype = SNAPSHOT_DTYPES.get(col)
            
            # Set appropriate default value based on dtype
            if pd.api.types.is_datetime64_any_dtype(dtype):
                snapshot_df[col] = pd.NaT
            elif pd.api.types.is_bool_dtype(dtype):
                snapshot_df[col] = pd.NA
            elif pd.api.types.is_integer_dtype(dtype):
                snapshot_df[col] = pd.NA
                snapshot_df[col] = snapshot_df[col].astype('Int64')
            elif pd.api.types.is_float_dtype(dtype):
                snapshot_df[col] = np.nan
            elif pd.api.types.is_string_dtype(dtype):
                snapshot_df[col] = pd.NA
                snapshot_df[col] = snapshot_df[col].astype('string')
            elif isinstance(dtype, pd.CategoricalDtype):
                snapshot_df[col] = pd.Categorical(
                    [pd.NA] * len(snapshot_df),
                    categories=dtype.categories,
                    ordered=dtype.ordered
                )
            else:  # Default to StringDtype for others like 'object' or 'string'
                snapshot_df[col] = pd.NA
                if pd.api.types.is_string_dtype(dtype) or str(dtype) == 'object':
                    snapshot_df[col] = snapshot_df[col].astype(pd.StringDtype())  # Ensure it's nullable string
    
    # Select and order columns according to SNAPSHOT_COL_NAMES
    # Ensure EMP_ID is present before trying to set it as index
    if EMP_ID not in snapshot_df.columns and EMP_ID in SNAPSHOT_COL_NAMES:
        # This case should ideally be handled by the loop above, but as a safeguard:
        logger.error(f"{EMP_ID} column is missing but listed in SNAPSHOT_COL_NAMES. Adding as NA.")
        snapshot_df[EMP_ID] = pd.NA 
        snapshot_df[EMP_ID] = snapshot_df[EMP_ID].astype(SNAPSHOT_DTYPES.get(EMP_ID, pd.StringDtype()))
    
    # Explicitly ensure all SNAPSHOT_COL_NAMES are in the snapshot_df
    # and have the correct data types from SNAPSHOT_DTYPES
    for col in SNAPSHOT_COL_NAMES:
        dtype = SNAPSHOT_DTYPES.get(col)
        if col not in snapshot_df.columns:
            logger.warning(f"Column '{col}' not in snapshot_df but is in SNAPSHOT_COL_NAMES. Adding it with NAs.")
            if pd.api.types.is_datetime64_any_dtype(dtype):
                snapshot_df[col] = pd.NaT
            elif isinstance(dtype, pd.CategoricalDtype):
                snapshot_df[col] = pd.Categorical(
                    [pd.NA] * len(snapshot_df),
                    categories=dtype.categories,
                    ordered=dtype.ordered
                )
            else:
                snapshot_df[col] = pd.NA
        
        # Ensure the column has the correct dtype
        if dtype is not None and not pd.isna(dtype):
            try:
                snapshot_df[col] = snapshot_df[col].astype(dtype)
            except Exception as e:
                logger.warning(f"Could not convert column '{col}' to dtype {dtype}: {e}")
    
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
    
    # Final validation for EMP_LEVEL - ensure no NaN values remain
    if EMP_LEVEL in snapshot_df.columns:
        nan_levels_mask = snapshot_df[EMP_LEVEL].isna()
        nan_count = nan_levels_mask.sum()
        
        if nan_count > 0:
            # Get employee IDs with NaN levels for logging
            nan_emp_ids = snapshot_df.loc[nan_levels_mask, EMP_ID].tolist()
            log_sample = nan_emp_ids[:5]  # Show first 5 IDs in log
            
            logger.warning(
                f"Found {nan_count} employees with NaN values in {EMP_LEVEL}. "
                f"Setting to default level 1. Sample IDs: {log_sample}"
                f"{' (plus more...)' if nan_count > 5 else ''}"
            )
            
            # Fill NaN values with default level 1
            snapshot_df.loc[nan_levels_mask, EMP_LEVEL] = 1
            
            # Also update the level source to indicate this was a fallback
            if EMP_LEVEL_SOURCE in snapshot_df.columns:
                snapshot_df.loc[nan_levels_mask, EMP_LEVEL_SOURCE] = 'default-validation'

    # Ensure EMP_LEVEL is properly typed as Int64
    if EMP_LEVEL in snapshot_df.columns:
        try:
            snapshot_df[EMP_LEVEL] = snapshot_df[EMP_LEVEL].astype('Int64')
        except Exception as e:
            logger.error(f"Failed to convert {EMP_LEVEL} to Int64 after validation: {e}")
    
    return snapshot_df

def update_snapshot_with_events(
    prev_snapshot: pd.DataFrame,
    events_df: pd.DataFrame,
    as_of: pd.Timestamp,
    event_priority: Dict[str, int]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply events up to 'as_of' date to the previous snapshot.
    Returns the updated snapshot and list of active employee IDs as of 'as_of'.
    """
    # Filter events by timestamp
    if 'event_time' in events_df.columns:
        filtered = events_df[events_df[EVENT_TIME] <= as_of]
    else:
        filtered = events_df.copy()
    # Apply update from state snapshot
    from cost_model.state.snapshot import update as _update_snapshot

    updated_snapshot = _update_snapshot(prev_snapshot, filtered, as_of.year)
    # Determine active employees at year-end
    if 'active' in updated_snapshot.columns:
        year_end_employee_ids = updated_snapshot[updated_snapshot[EMP_ACTIVE]].index.tolist()
    else:
        year_end_employee_ids = []
    return updated_snapshot, year_end_employee_ids


def consolidate_snapshots_to_parquet(snapshots_dir: Union[str, Path], output_path: Union[str, Path]) -> None:
    """
    Combine all yearly snapshots into a single parquet file with a 'year' column.
    
    This function also ensures that unnecessary columns are removed and that simulation_year is properly set.
    
    Args:
        snapshots_dir: Directory containing yearly snapshot parquet files
        output_path: Path where to save the consolidated parquet file
    """
    import pandas as pd
    from pathlib import Path
    import re
    from cost_model.state.schema import EMP_ID

    # Columns to remove from the final output
    columns_to_remove = [
        'term_rate',
        'comp_raise_pct',
        'new_hire_term_rate',
        'cola_pct',
        'cfg'
    ]

    snapshots_dir = Path(snapshots_dir)
    output_path = Path(output_path)
    
    # Find all parquet files in the directory
    parquet_files = sorted(snapshots_dir.glob('*.parquet'))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {snapshots_dir}")

    all_snapshots = []
    for file in parquet_files:
        # 1) extract the year (defaults to 0 if none found)
        match = re.search(r'(\d{4})', file.stem)
        year = int(match.group(1)) if match else 0
        logger.info(f"Reading snapshot {file.name} → setting simulation_year={year}")

        # 2) load
        df = pd.read_parquet(file)

        # 3) drop the unwanted fields if they exist
        to_drop = [c for c in columns_to_remove if c in df.columns]
        if to_drop:
            df.drop(columns=to_drop, inplace=True)
            logger.debug(f"Dropped columns {to_drop} from {file.name}")

        # 4) (re)assign simulation_year unconditionally
        df['simulation_year'] = year

        # 5) add a separate 'year' column for clarity
        df['year'] = year

        # 6) bring EMP_ID back as a column (in case it's still the index)
        if EMP_ID in df.index.names:
            df = df.reset_index()

        all_snapshots.append(df)

    # concatenate
    if all_snapshots:  # Only proceed if we have snapshots to concatenate
        combined = pd.concat(all_snapshots, ignore_index=True)

        # fill any remaining NaNs in simulation_year from the 'year' column
        combined['simulation_year'] = combined['simulation_year'].fillna(combined['year']).astype(int)

        # final sort & write
        combined = combined.sort_values(['year', EMP_ID])
        combined.to_parquet(output_path, index=False)

        logger.info(f"Consolidated {len(parquet_files)} snapshots → {output_path}")
        logger.info(f"Final combined snapshot has {len(combined)} records")
        logger.info(f"Years covered: {sorted(combined['year'].unique())}")
        logger.info(f"simulation_year values: {sorted(combined['simulation_year'].dropna().unique())}")
    else:
        logger.warning("No valid snapshots found to consolidate")