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
        from cost_model.state.schema import EVENT_TIME
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


def _extract_compensation_for_employee(
    emp_id: str,
    year_events: pd.DataFrame,
    simulation_year: int,
    logger: logging.Logger
) -> float:
    """
    Extract compensation for a terminated new hire employee using priority-based approach.

    Priority order:
    1. Most recent EVT_COMP event within the simulation year up to termination date
    2. Compensation from EVT_HIRE event value_json payload
    3. Level-based default compensation
    4. Global default compensation (50000.0)

    Args:
        emp_id: Employee ID to extract compensation for
        year_events: Events that occurred during the simulation year
        simulation_year: The simulation year being processed
        logger: Logger instance for debugging

    Returns:
        Compensation value as float
    """
    from cost_model.state.schema import EVT_COMP, EVT_HIRE, EMP_ID, EMP_TERM_DATE
    from cost_model.state.snapshot_utils import parse_hire_payload
    import json

    # Default compensation values
    DEFAULT_COMPENSATION = 50000.0
    LEVEL_BASED_DEFAULTS = {
        1: 50000.0,
        2: 75000.0,
        3: 100000.0,
        4: 150000.0
    }

    # Get termination date for this employee to limit compensation event search
    term_events = year_events[
        (year_events['event_type'].isin(['EVT_TERM', 'EVT_NEW_HIRE_TERM'])) &
        (year_events[EMP_ID] == emp_id)
    ]
    term_date = None
    if not term_events.empty:
        term_date = term_events['event_time'].iloc[0]

    # Priority 1: Look for most recent EVT_COMP event within the year up to termination
    comp_events = year_events[
        (year_events['event_type'] == EVT_COMP) &
        (year_events[EMP_ID] == emp_id)
    ]

    if not comp_events.empty:
        # Filter by termination date if available
        if term_date is not None:
            comp_events = comp_events[comp_events['event_time'] <= term_date]

        if not comp_events.empty:
            # Get the most recent compensation event
            latest_comp_event = comp_events.sort_values('event_time').iloc[-1]
            if pd.notna(latest_comp_event['value_num']) and latest_comp_event['value_num'] > 0:
                logger.info(f"Using EVT_COMP compensation for employee {emp_id}: {latest_comp_event['value_num']}")
                return float(latest_comp_event['value_num'])

    # Priority 2: Extract compensation from EVT_HIRE event value_json
    hire_events = year_events[
        (year_events['event_type'] == EVT_HIRE) &
        (year_events[EMP_ID] == emp_id)
    ]

    if not hire_events.empty:
        hire_event = hire_events.iloc[0]

        # Try to extract from value_json
        if pd.notna(hire_event.get('value_json')):
            try:
                if isinstance(hire_event['value_json'], str):
                    payload = json.loads(hire_event['value_json'])
                else:
                    payload = hire_event['value_json']

                if isinstance(payload, dict) and 'compensation' in payload:
                    comp_value = float(payload['compensation'])
                    if comp_value > 0:
                        logger.info(f"Using hire event compensation for employee {emp_id}: {comp_value}")
                        return comp_value
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                logger.warning(f"Failed to parse compensation from hire event for employee {emp_id}: {e}")

        # Also try to extract level for fallback
        level = 1  # default
        if pd.notna(hire_event.get('value_json')):
            try:
                payload = parse_hire_payload(hire_event['value_json'])
                level = payload.get('level', 1)
            except Exception:
                pass

        # Priority 3: Use level-based default
        level_comp = LEVEL_BASED_DEFAULTS.get(int(level), DEFAULT_COMPENSATION)
        logger.info(f"Using level-based default compensation for employee {emp_id} (level={level}): {level_comp}")
        return level_comp

    # Priority 4: Global default
    logger.info(f"Using global default compensation for employee {emp_id}: {DEFAULT_COMPENSATION}")
    return DEFAULT_COMPENSATION


def build_enhanced_yearly_snapshot(
    start_of_year_snapshot: pd.DataFrame,
    end_of_year_snapshot: pd.DataFrame,
    year_events: pd.DataFrame,
    simulation_year: int
) -> pd.DataFrame:
    """
    Build an enhanced yearly snapshot that includes all employees who were active
    at any point during the specified simulation year.

    This implements the approved solution for yearly snapshot generation:
    - Includes employees active at start of year
    - Includes employees hired during the year
    - Includes employees who terminated during the year
    - Sets employee_status_eoy based on their status at end of year
    - Populates employee_termination_date for terminated employees
    - FIXED: Properly populates compensation for terminated new hires

    Args:
        start_of_year_snapshot: Snapshot at the beginning of the year
        end_of_year_snapshot: Snapshot at the end of the year (after all events)
        year_events: Events that occurred during this simulation year
        simulation_year: The simulation year being processed

    Returns:
        Enhanced yearly snapshot DataFrame with all employees active during the year
    """
    from cost_model.state.schema import (
        EMP_ID, EMP_ACTIVE, EMP_TERM_DATE, EMP_STATUS_EOY,
        EVT_HIRE, EVT_TERM, EVT_NEW_HIRE_TERM, SIMULATION_YEAR, EMP_HIRE_DATE
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Building enhanced yearly snapshot for year {simulation_year}")

    # Step 1: Get employees active at start of year
    soy_active_employees = start_of_year_snapshot[
        start_of_year_snapshot[EMP_ACTIVE] == True
    ].copy() if EMP_ACTIVE in start_of_year_snapshot.columns else start_of_year_snapshot.copy()
    soy_active_ids = set(soy_active_employees[EMP_ID].unique()) if EMP_ID in soy_active_employees.columns else set()

    # Step 2: Get employees hired during the year from events
    hired_this_year = set()
    if year_events is not None and not year_events.empty and 'event_type' in year_events.columns:
        hire_events = year_events[year_events['event_type'] == EVT_HIRE]
        if not hire_events.empty and EMP_ID in hire_events.columns:
            hired_this_year = set(hire_events[EMP_ID].unique())
            logger.info(f"Found {len(hired_this_year)} employees hired during year {simulation_year}")

    # Step 3: Get employees terminated during the year from events
    # CRITICAL FIX: Include both regular terminations (EVT_TERM) and new hire terminations (EVT_NEW_HIRE_TERM)
    terminated_this_year = set()
    if year_events is not None and not year_events.empty and 'event_type' in year_events.columns:
        # Include both types of termination events
        term_events = year_events[year_events['event_type'].isin([EVT_TERM, EVT_NEW_HIRE_TERM])]
        if not term_events.empty and EMP_ID in term_events.columns:
            terminated_this_year = set(term_events[EMP_ID].unique())
            # Log breakdown of termination types for debugging
            regular_terms = len(year_events[year_events['event_type'] == EVT_TERM][EMP_ID].unique()) if EVT_TERM in year_events['event_type'].values else 0
            new_hire_terms = len(year_events[year_events['event_type'] == EVT_NEW_HIRE_TERM][EMP_ID].unique()) if EVT_NEW_HIRE_TERM in year_events['event_type'].values else 0
            logger.info(f"Found {len(terminated_this_year)} employees terminated during year {simulation_year} "
                       f"(regular: {regular_terms}, new hire: {new_hire_terms})")

    # Step 4: Form the "Active During Year" set
    # CRITICAL FIX: Include ALL employees who were active at any point during the year:
    # - Employees active at start of year
    # - Employees hired during the year
    # - Employees terminated during the year (they were active for part of the year)
    active_during_year_ids = soy_active_ids.union(hired_this_year).union(terminated_this_year)

    logger.info(f"Total employees active during year {simulation_year}: {len(active_during_year_ids)} "
                f"(SOY active: {len(soy_active_ids)}, hired: {len(hired_this_year)}, terminated: {len(terminated_this_year)})")

    # Step 5: Build the yearly snapshot from multiple sources
    # CRITICAL FIX: The EOY snapshot excludes terminated employees, so we need to reconstruct
    # the full set of employees who were active during the year from multiple sources

    logger.info(f"Building yearly snapshot from SOY snapshot + EOY snapshot + events for {len(active_during_year_ids)} employees")

    # Start with employees from EOY snapshot (survivors + new hires)
    if EMP_ID in end_of_year_snapshot.columns:
        eoy_employees = end_of_year_snapshot[
            end_of_year_snapshot[EMP_ID].isin(active_during_year_ids)
        ].copy()
    else:
        logger.warning(f"EMP_ID column not found in end_of_year_snapshot. Using full snapshot.")
        eoy_employees = end_of_year_snapshot.copy()

    # Add terminated employees from SOY snapshot who are not in EOY snapshot
    eoy_employee_ids = set(eoy_employees[EMP_ID].unique()) if EMP_ID in eoy_employees.columns else set()
    missing_employee_ids = active_during_year_ids - eoy_employee_ids

    if missing_employee_ids:
        logger.info(f"Found {len(missing_employee_ids)} employees active during year but missing from EOY snapshot (likely terminated)")

        # CRITICAL FIX: Handle both employees from SOY snapshot and new hires who terminated
        missing_employees_list = []

        # First, get terminated employees who were in SOY snapshot
        # IMPORTANT: Only include SOY employees who actually have termination events
        if EMP_ID in start_of_year_snapshot.columns:
            soy_missing_candidates = start_of_year_snapshot[
                start_of_year_snapshot[EMP_ID].isin(missing_employee_ids)
            ].copy()

            # Filter to only include employees who actually terminated (have termination events)
            soy_missing = soy_missing_candidates[
                soy_missing_candidates[EMP_ID].isin(terminated_this_year)
            ].copy()

            if not soy_missing.empty:
                missing_employees_list.append(soy_missing)
                logger.info(f"Found {len(soy_missing)} terminated employees from SOY snapshot")

            # Log any employees who were in SOY but missing from EOY without termination events
            soy_missing_no_term = soy_missing_candidates[
                ~soy_missing_candidates[EMP_ID].isin(terminated_this_year)
            ]
            if not soy_missing_no_term.empty:
                logger.warning(f"Found {len(soy_missing_no_term)} employees in SOY but missing from EOY without termination events: {soy_missing_no_term[EMP_ID].tolist()}")

        # Second, reconstruct new hires who terminated from events
        # These employees won't be in SOY snapshot since they were hired during the year
        new_hire_terminated_ids = missing_employee_ids.intersection(hired_this_year).intersection(terminated_this_year)
        if new_hire_terminated_ids and year_events is not None and not year_events.empty:
            logger.info(f"Reconstructing {len(new_hire_terminated_ids)} terminated new hires from events")

            # Reconstruct employee data from hire and termination events
            from cost_model.state.snapshot_utils import parse_hire_payload

            for emp_id in new_hire_terminated_ids:
                # Get hire event for this employee
                hire_events = year_events[
                    (year_events['event_type'] == EVT_HIRE) &
                    (year_events[EMP_ID] == emp_id)
                ]

                # Get termination event for this employee
                term_events = year_events[
                    (year_events['event_type'].isin([EVT_TERM, EVT_NEW_HIRE_TERM])) &
                    (year_events[EMP_ID] == emp_id)
                ]

                if not hire_events.empty and not term_events.empty:
                    hire_event = hire_events.iloc[0]
                    term_event = term_events.iloc[0]

                    # CRITICAL FIX: Extract compensation using priority-based approach
                    compensation = _extract_compensation_for_employee(
                        emp_id, year_events, simulation_year, logger
                    )

                    # Extract additional details from hire event
                    birth_date = pd.NaT
                    level = 1
                    role = 'Regular'

                    if pd.notna(hire_event.get('value_json')):
                        try:
                            payload = parse_hire_payload(hire_event['value_json'])
                            birth_date = pd.to_datetime(payload.get('birth_date'), errors='coerce') if payload.get('birth_date') else pd.NaT
                            level = payload.get('level', 1)
                            role = payload.get('role', 'Regular')
                        except Exception as e:
                            logger.warning(f"Failed to parse hire event details for employee {emp_id}: {e}")

                    # CRITICAL FIX: Calculate accurate tenure for new hires who terminate within the same year
                    # Use the duration between hire_date and termination_date for tenure calculation
                    hire_date = hire_event['event_time']
                    term_date = term_event['event_time']

                    # Calculate tenure in years using the existing utility function
                    from cost_model.utils.date_utils import calculate_tenure
                    tenure_years = calculate_tenure(hire_date, term_date)

                    # Ensure tenure is at least 0 and round to 3 decimal places for consistency
                    tenure_years = max(0.0, round(tenure_years, 3))

                    # Calculate tenure band using existing utility function
                    from cost_model.state.snapshot.tenure import assign_tenure_band
                    tenure_band = assign_tenure_band(tenure_years)

                    logger.info(f"Calculated tenure for terminated new hire {emp_id}: {tenure_years:.3f} years ({tenure_band}) from {hire_date.date()} to {term_date.date()}")

                    # Create comprehensive employee record from hire event
                    emp_data = {
                        EMP_ID: emp_id,
                        EMP_HIRE_DATE: hire_date,
                        EMP_TERM_DATE: term_date,
                        EMP_ACTIVE: False,
                        SIMULATION_YEAR: simulation_year
                    }

                    # Add other required columns with properly populated values
                    emp_data.update({
                        'employee_birth_date': birth_date,
                        'employee_gross_compensation': float(compensation),  # FIXED: Now properly populated
                        'employee_deferral_rate': 0.0,
                        'employee_tenure': tenure_years,  # FIXED: Now properly calculated
                        'employee_tenure_band': tenure_band,  # FIXED: Now properly calculated
                        'employee_level': int(level),
                        'job_level_source': 'hire',
                        'exited': True,
                        'employee_role': role
                    })

                    # Create DataFrame for this employee
                    emp_df = pd.DataFrame([emp_data])
                    missing_employees_list.append(emp_df)

                    logger.debug(f"Reconstructed terminated new hire {emp_id} with compensation {compensation}")
                else:
                    logger.warning(f"Could not find both hire and termination events for employee {emp_id}")

        # Combine all missing employees
        if missing_employees_list:
            missing_employees = pd.concat(missing_employees_list, ignore_index=True)

            # Update termination information for all missing employees
            for emp_id in missing_employee_ids:
                if emp_id in terminated_this_year:
                    # Find termination event for this employee (both regular and new hire terminations)
                    if year_events is not None and not year_events.empty:
                        emp_term_events = year_events[
                            (year_events['event_type'].isin([EVT_TERM, EVT_NEW_HIRE_TERM])) &
                            (year_events[EMP_ID] == emp_id)
                        ]
                        if not emp_term_events.empty:
                            term_date = emp_term_events['event_time'].iloc[0]
                            missing_employees.loc[missing_employees[EMP_ID] == emp_id, EMP_TERM_DATE] = term_date
                            missing_employees.loc[missing_employees[EMP_ID] == emp_id, EMP_ACTIVE] = False

            # Combine EOY employees with missing (terminated) employees
            yearly_snapshot = pd.concat([eoy_employees, missing_employees], ignore_index=True)
            logger.info(f"Added {len(missing_employees)} terminated employees to yearly snapshot")
        else:
            logger.warning("No missing employees could be reconstructed")
            yearly_snapshot = eoy_employees
    else:
        yearly_snapshot = eoy_employees

    # Step 6: Determine employee_status_eoy for each employee
    yearly_snapshot[EMP_STATUS_EOY] = yearly_snapshot.apply(
        lambda row: _determine_employee_status_eoy(row, simulation_year), axis=1
    )

    # Step 7: Ensure simulation_year is set correctly
    yearly_snapshot[SIMULATION_YEAR] = simulation_year

    # Step 7.5: Calculate tenure for ALL employees in the yearly snapshot
    # This ensures that both terminated and active employees have proper tenure data
    from cost_model.state.snapshot.tenure import compute_tenure

    # For terminated employees, calculate tenure from hire date to termination date
    # For active employees, calculate tenure from hire date to end of year
    as_of_eoy = pd.Timestamp(f"{simulation_year}-12-31")

    # Create a copy to work with
    yearly_snapshot_with_tenure = yearly_snapshot.copy()

    # Calculate tenure for each employee based on their status
    for idx, row in yearly_snapshot_with_tenure.iterrows():
        hire_date = row.get(EMP_HIRE_DATE)
        term_date = row.get(EMP_TERM_DATE)

        if pd.notna(hire_date):
            # For terminated employees, use termination date; for active employees, use end of year
            if pd.notna(term_date):
                # Terminated employee - use termination date
                tenure_end_date = term_date
            else:
                # Active employee - use end of year
                tenure_end_date = as_of_eoy

            # Calculate tenure using existing utility
            from cost_model.utils.date_utils import calculate_tenure
            tenure_years = calculate_tenure(hire_date, tenure_end_date)
            tenure_years = max(0.0, round(tenure_years, 3))

            # Calculate tenure band
            from cost_model.state.snapshot.tenure import assign_tenure_band
            tenure_band = assign_tenure_band(tenure_years)

            # Update the row
            yearly_snapshot_with_tenure.loc[idx, 'employee_tenure'] = tenure_years
            yearly_snapshot_with_tenure.loc[idx, 'employee_tenure_band'] = tenure_band
        else:
            # No hire date - set to 0
            yearly_snapshot_with_tenure.loc[idx, 'employee_tenure'] = 0.0
            yearly_snapshot_with_tenure.loc[idx, 'employee_tenure_band'] = '0-1'

    # Replace the original snapshot with the one that has tenure data
    yearly_snapshot = yearly_snapshot_with_tenure

    logger.info(f"Applied tenure calculations to all {len(yearly_snapshot)} employees in yearly snapshot")

    # Step 8: Log summary statistics including compensation and tenure validation
    if EMP_STATUS_EOY in yearly_snapshot.columns:
        status_counts = yearly_snapshot[EMP_STATUS_EOY].value_counts()
        logger.info(f"Year {simulation_year} snapshot status distribution: {status_counts.to_dict()}")

    # CRITICAL: Validate compensation data for terminated employees
    if 'employee_gross_compensation' in yearly_snapshot.columns:
        terminated_employees = yearly_snapshot[yearly_snapshot[EMP_STATUS_EOY] == 'Terminated']
        if not terminated_employees.empty:
            comp_stats = terminated_employees['employee_gross_compensation'].describe()
            zero_comp_count = (terminated_employees['employee_gross_compensation'] == 0).sum()
            null_comp_count = terminated_employees['employee_gross_compensation'].isna().sum()

            logger.info(f"Terminated employees compensation validation:")
            logger.info(f"  - Total terminated: {len(terminated_employees)}")
            logger.info(f"  - Zero compensation: {zero_comp_count}")
            logger.info(f"  - Null compensation: {null_comp_count}")
            logger.info(f"  - Compensation stats: mean={comp_stats['mean']:.2f}, min={comp_stats['min']:.2f}, max={comp_stats['max']:.2f}")

            if zero_comp_count > 0 or null_comp_count > 0:
                logger.warning(f"Found {zero_comp_count + null_comp_count} terminated employees with missing compensation data!")
            else:
                logger.info("✓ All terminated employees have valid compensation data")

    # CRITICAL: Validate tenure data for all employees, especially terminated new hires
    if 'employee_tenure' in yearly_snapshot.columns and 'employee_tenure_band' in yearly_snapshot.columns:
        tenure_stats = yearly_snapshot['employee_tenure'].describe()
        zero_tenure_count = (yearly_snapshot['employee_tenure'] == 0).sum()
        null_tenure_count = yearly_snapshot['employee_tenure'].isna().sum()

        # Check specifically for terminated new hires
        terminated_employees = yearly_snapshot[yearly_snapshot[EMP_STATUS_EOY] == 'Terminated']
        if not terminated_employees.empty:
            terminated_tenure_stats = terminated_employees['employee_tenure'].describe()
            terminated_zero_tenure = (terminated_employees['employee_tenure'] == 0).sum()
            terminated_null_tenure = terminated_employees['employee_tenure'].isna().sum()

            logger.info(f"Tenure validation for all employees:")
            logger.info(f"  - Total employees: {len(yearly_snapshot)}")
            logger.info(f"  - Zero tenure: {zero_tenure_count}")
            logger.info(f"  - Null tenure: {null_tenure_count}")
            logger.info(f"  - Tenure stats: mean={tenure_stats['mean']:.3f}, min={tenure_stats['min']:.3f}, max={tenure_stats['max']:.3f}")

            logger.info(f"Tenure validation for terminated employees:")
            logger.info(f"  - Total terminated: {len(terminated_employees)}")
            logger.info(f"  - Zero tenure: {terminated_zero_tenure}")
            logger.info(f"  - Null tenure: {terminated_null_tenure}")
            logger.info(f"  - Terminated tenure stats: mean={terminated_tenure_stats['mean']:.3f}, min={terminated_tenure_stats['min']:.3f}, max={terminated_tenure_stats['max']:.3f}")

            # Check tenure band distribution for terminated employees
            if 'employee_tenure_band' in terminated_employees.columns:
                tenure_band_counts = terminated_employees['employee_tenure_band'].value_counts()
                logger.info(f"  - Tenure band distribution: {tenure_band_counts.to_dict()}")

            if terminated_null_tenure > 0:
                logger.warning(f"Found {terminated_null_tenure} terminated employees with missing tenure data!")
            else:
                logger.info("✓ All terminated employees have valid tenure data")

    logger.info(f"Enhanced yearly snapshot for {simulation_year} contains {len(yearly_snapshot)} employees")

    return yearly_snapshot


def _determine_employee_status_eoy(employee_row: pd.Series, simulation_year: int) -> str:
    """
    Determine the employee status at end of year based on their data.

    Args:
        employee_row: Series containing employee data
        simulation_year: The simulation year being processed

    Returns:
        Status string: 'Active', 'Terminated', or 'Inactive'
    """
    from cost_model.state.schema import EMP_ACTIVE, EMP_TERM_DATE

    # Check if employee is active at EOY
    is_active = employee_row.get(EMP_ACTIVE, False)
    if is_active:
        return "Active"

    # Check if employee terminated during this year
    term_date = employee_row.get(EMP_TERM_DATE)
    if pd.notna(term_date):
        try:
            term_date_parsed = pd.to_datetime(term_date)
            if term_date_parsed.year == simulation_year:
                return "Terminated"
        except (ValueError, TypeError):
            pass  # Invalid date format, continue to default

    # Default to Inactive for employees who are not active and didn't terminate this year
    return "Inactive"


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