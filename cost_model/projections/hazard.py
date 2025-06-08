# cost_model/projections/hazard.py
"""
Hazard module for generating hazard tables used in workforce projections.
QuickStart: see docs/cost_model/projections/hazard.md
"""

import pandas as pd
import logging
import os
from typing import List, Dict, Optional, Union, Tuple
# Removed standardize_tenure_band import - hazard table should have correct format
from cost_model.state.schema import (
    EMP_LEVEL,
    EMP_TENURE_BAND,
    SIMULATION_YEAR,
    TERM_RATE,
    MERIT_RAISE_PCT,  # CRITICAL FIX: Use correct column for merit raises
    COMP_RAISE_PCT,   # Keep for backward compatibility
    NEW_HIRE_TERMINATION_RATE,  # CRITICAL FIX: Use correct constant
    COLA_PCT,
    CFG
)

logger = logging.getLogger(__name__)


def _get_cola_rate_for_year(global_params, year: int) -> float:
    """
    Get COLA rate for a specific year from global_params configuration.
    
    Looks for cola_hazard.by_year.{year} configuration.
    Falls back to 0.02 (2%) default if not found.
    """
    try:
        if hasattr(global_params, 'cola_hazard') and hasattr(global_params.cola_hazard, 'by_year'):
            year_rates = global_params.cola_hazard.by_year
            if hasattr(year_rates, str(year)):
                return getattr(year_rates, str(year))
            elif isinstance(year_rates, dict) and year in year_rates:
                return year_rates[year]
            elif isinstance(year_rates, dict) and str(year) in year_rates:
                return year_rates[str(year)]
        
        # Fallback: look for global cola_pct attribute
        if hasattr(global_params, 'cola_pct'):
            return global_params.cola_pct
            
        # Default fallback
        logger.warning(f"No COLA rate found for year {year}. Using default 2%.")
        return 0.02
        
    except Exception as e:
        logger.warning(f"Error getting COLA rate for year {year}: {e}. Using default 2%.")
        return 0.02


def build_hazard_table(
    years: List[int],
    initial_snapshot: pd.DataFrame,
    global_params,
    plan_rules_config
) -> pd.DataFrame:
    """Generates the hazard table based on configuration and initial snapshot."""
    logger.info("Generating hazard table...")
    from cost_model.state.schema import EMP_TENURE_BAND
    if EMP_LEVEL in initial_snapshot.columns and EMP_TENURE_BAND in initial_snapshot.columns:
        unique_levels_tenures = initial_snapshot[[EMP_LEVEL, EMP_TENURE_BAND]].drop_duplicates().to_dict('records')
    else:
        logger.warning(f"'{EMP_LEVEL}' or '{EMP_TENURE_BAND}' not in initial snapshot. Using default '1'/'all'.")
        unique_levels_tenures = [{EMP_LEVEL: 1, EMP_TENURE_BAND: 'all'}]

    # Robustly check for annual_termination_rate, comp_raise_pct, and nh_term_rate
    if hasattr(global_params, 'annual_termination_rate'):
        global_term_rate = global_params.annual_termination_rate
    else:
        logger.warning("global_params missing 'annual_termination_rate'. Using default 0.10. Available attributes: %s", dir(global_params))
        global_term_rate = 0.10
    if hasattr(global_params, 'annual_compensation_increase_rate'):
        global_comp_raise_pct = global_params.annual_compensation_increase_rate
    else:
        logger.warning("global_params missing 'annual_compensation_increase_rate'. Using default 0.03. Available attributes: %s", dir(global_params))
        global_comp_raise_pct = 0.03
    # Look for new_hire_termination_rate in attrition section first, then root level, then use default
    if hasattr(global_params, 'attrition') and hasattr(global_params.attrition, 'new_hire_termination_rate'):
        global_nh_term_rate = global_params.attrition.new_hire_termination_rate
    elif hasattr(global_params, 'new_hire_termination_rate'):
        global_nh_term_rate = global_params.new_hire_termination_rate
    else:
        logger.warning("global_params missing 'new_hire_termination_rate' in both root and attrition. Using default 0.25. Available attributes: %s", dir(global_params))
        global_nh_term_rate = 0.25
    logger.info(f"Using global rates: Term={global_term_rate}, CompPct={global_comp_raise_pct}, NH_Term={global_nh_term_rate}")

    # Define all possible employee levels (0-4) and standard tenure bands
    all_employee_levels = set(range(5))  # 0, 1, 2, 3, 4
    standard_tenure_bands = {'<1', '1-3', '3-5', '5+'}

    # Create a set of all possible (level, tenure_band) combinations
    all_combinations = set()

    # First, add all combinations from the initial snapshot
    for combo in unique_levels_tenures:
        # Use tenure band as-is (no standardization needed)
        tenure_band = combo[EMP_TENURE_BAND]
        all_combinations.add((combo[EMP_LEVEL], tenure_band))

    # Then add all missing combinations with standard tenure bands
    for level in all_employee_levels:
        for tenure_band in standard_tenure_bands:
            all_combinations.add((level, tenure_band))

    # Convert back to list of dicts for the rest of the function
    unique_levels_tenures = [
        {EMP_LEVEL: level, EMP_TENURE_BAND: tenure_band}
        for level, tenure_band in all_combinations
    ]

    logger.info(f"Generated hazard table with {len(unique_levels_tenures)} unique (level, tenure_band) combinations")

    records = []
    for year in years:
        for combo in unique_levels_tenures:
            records.append({
                SIMULATION_YEAR: year,
                EMP_LEVEL: combo[EMP_LEVEL],
                EMP_TENURE_BAND: combo[EMP_TENURE_BAND],
                TERM_RATE: global_term_rate,
                MERIT_RAISE_PCT: global_comp_raise_pct,  # CRITICAL FIX: Use correct column name
                NEW_HIRE_TERMINATION_RATE: global_nh_term_rate,
                COLA_PCT: _get_cola_rate_for_year(global_params, year),  # CRITICAL FIX: Look up year-specific COLA rate
                CFG: plan_rules_config
            })
    if records:
        df = pd.DataFrame(records)

        # Remove any potential duplicates that might have been created
        initial_rows = len(df)
        df = df.drop_duplicates(subset=[SIMULATION_YEAR, EMP_LEVEL, EMP_TENURE_BAND])
        final_rows = len(df)

        if final_rows < initial_rows:
            removed_count = initial_rows - final_rows
            logger.warning(f"Removed {removed_count} duplicate (year, level, tenure_band) combinations during hazard table generation")

        logger.info(f"Generated hazard table with {final_rows} rows.")
    else:
        cols = [SIMULATION_YEAR, EMP_LEVEL, EMP_TENURE_BAND, TERM_RATE, MERIT_RAISE_PCT, NEW_HIRE_TERMINATION_RATE, COLA_PCT, CFG]
        df = pd.DataFrame(columns=cols)
        logger.warning("Empty hazard table created.")
    return df


def load_and_expand_hazard_table(path: str = 'data/hazard_table.parquet') -> pd.DataFrame:
    """
    Load hazard table from parquet file. Assumes 'employee_level' is present and correct.

    Args:
        path: Path to the hazard table parquet file

    Returns:
        DataFrame with standardized hazard table entries
    """
    logger.info(f"Loading hazard table from {path}")

    if not os.path.exists(path):
        logger.error(f"Hazard table file not found at {path}")
        return pd.DataFrame()

    # Load the hazard table
    try:
        df = pd.read_parquet(path)
        logger.info(f"Loaded hazard table with {len(df)} rows")
    except Exception as e:
        logger.error(f"Error loading hazard table: {e}")
        return pd.DataFrame()

    # Log the actual string values of constants to verify definitions
    # These constants are assumed to be imported e.g., from cost_model.constants
    logger.info(f"DEBUG CONSTANTS: SIMULATION_YEAR='{SIMULATION_YEAR}', EMP_LEVEL='{EMP_LEVEL}', EMP_TENURE_BAND='{EMP_TENURE_BAND}'")
    logger.info(f"DEBUG CONSTANTS: TERM_RATE='{TERM_RATE}', MERIT_RAISE_PCT='{MERIT_RAISE_PCT}', NEW_HIRE_TERMINATION_RATE='{NEW_HIRE_TERMINATION_RATE}', COLA_PCT='{COLA_PCT}'")

    logger.info(f"Initial columns from Parquet: {df.columns.tolist()}")

    # Define rename map: Parquet_column_name -> internal_constant_name
    rename_map = {
        'simulation_year': SIMULATION_YEAR,
        'employee_level': EMP_LEVEL,
        'tenure_band': EMP_TENURE_BAND,
        'term_rate': TERM_RATE,
        'merit_raise_pct': MERIT_RAISE_PCT,  # Corrected key based on new schema
        'new_hire_termination_rate': NEW_HIRE_TERMINATION_RATE,
        'cola_pct': COLA_PCT  # Added key based on error log's 'Available columns'
    }
    logger.info(f"Defined rename_map: {rename_map}")

    # Filter rename_map to only include columns present in the DataFrame and where source != destination
    actual_rename_map = {k: v for k, v in rename_map.items() if k in df.columns and k != v}
    logger.info(f"Actual rename_map to be applied: {actual_rename_map}")

    if actual_rename_map:
        logger.info(f"Columns BEFORE rename operation: {df.columns.tolist()}")
        df = df.rename(columns=actual_rename_map)
        logger.info(f"Columns AFTER rename operation: {df.columns.tolist()}")

        # Specific verification for critical renames
        if 'tenure_band' in actual_rename_map: # Check if 'tenure_band' was intended for renaming
            if EMP_TENURE_BAND in df.columns and 'tenure_band' not in df.columns:
                logger.info(f"Verification: Rename of 'tenure_band' to '{EMP_TENURE_BAND}' appears successful.")
            elif EMP_TENURE_BAND in df.columns and 'tenure_band' in df.columns:
                 logger.warning(f"Verification: Rename of 'tenure_band' to '{EMP_TENURE_BAND}' PARTIALLY FAILED. Both '{EMP_TENURE_BAND}' and 'tenure_band' exist. Check constant value of EMP_TENURE_BAND.")
            elif EMP_TENURE_BAND not in df.columns and 'tenure_band' in df.columns:
                 logger.warning(f"Verification: Rename of 'tenure_band' to '{EMP_TENURE_BAND}' FAILED. Original 'tenure_band' still present, '{EMP_TENURE_BAND}' is not. Check constant value of EMP_TENURE_BAND or rename_map.")
            else: # EMP_TENURE_BAND not in df.columns and 'tenure_band' not in df.columns
                 logger.warning(f"Verification: Both 'tenure_band' and '{EMP_TENURE_BAND}' are MISSING after rename attempt. This is unexpected.")

        if 'new_hire_termination_rate' in actual_rename_map: # Check if 'new_hire_termination_rate' was intended for renaming
            if NEW_HIRE_TERMINATION_RATE in df.columns and 'new_hire_termination_rate' not in df.columns:
                logger.info(f"Verification: Rename of 'new_hire_termination_rate' to '{NEW_HIRE_TERMINATION_RATE}' appears successful.")
            elif NEW_HIRE_TERMINATION_RATE in df.columns and 'new_hire_termination_rate' in df.columns:
                logger.warning(f"Verification: Rename of 'new_hire_termination_rate' to '{NEW_HIRE_TERMINATION_RATE}' PARTIALLY FAILED. Both '{NEW_HIRE_TERMINATION_RATE}' and 'new_hire_termination_rate' exist. Check constant value of NEW_HIRE_TERMINATION_RATE.")
            elif NEW_HIRE_TERMINATION_RATE not in df.columns and 'new_hire_termination_rate' in df.columns:
                logger.warning(f"Verification: Rename of 'new_hire_termination_rate' to '{NEW_HIRE_TERMINATION_RATE}' FAILED. Original 'new_hire_termination_rate' still present, '{NEW_HIRE_TERMINATION_RATE}' is not. Check constant value of NEW_HIRE_TERMINATION_RATE or rename_map.")
            else: # NEW_HIRE_TERMINATION_RATE not in df.columns and 'new_hire_termination_rate' not in df.columns
                logger.warning(f"Verification: Both 'new_hire_termination_rate' and '{NEW_HIRE_TERMINATION_RATE}' are MISSING after rename attempt. This is unexpected.")
    else:
        logger.info("No columns to rename: actual_rename_map is empty or no keys matched Parquet columns.")

    # Define required columns using internal constant names.
    # Updated to handle both old and new compensation schemas
    base_required_cols = [
        SIMULATION_YEAR,
        EMP_LEVEL,
        EMP_TENURE_BAND,
        TERM_RATE,
        NEW_HIRE_TERMINATION_RATE
    ]

    # Check for compensation columns - either old schema (comp_raise_pct) or new schema (merit_raise_pct)
    has_old_comp_schema = 'comp_raise_pct' in df.columns
    has_new_comp_schema = 'merit_raise_pct' in df.columns

    if has_old_comp_schema:
        required_cols = base_required_cols + ['comp_raise_pct']
        logger.info("Using old compensation schema with comp_raise_pct")
    elif has_new_comp_schema:
        required_cols = base_required_cols + ['merit_raise_pct']
        logger.info("Using new compensation schema with merit_raise_pct")
    else:
        # Neither schema found - this is an error
        logger.error(f"Hazard table missing compensation columns. Expected either 'comp_raise_pct' or 'merit_raise_pct'. Available columns: {df.columns.tolist()}")
        return pd.DataFrame()

    logger.info(f"Checking for required_cols (using constant values): {required_cols}")
    logger.info(f"Current df.columns for check: {df.columns.tolist()}")

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        logger.error(f"Hazard table missing required columns AFTER RENAME AND VERIFICATION: {missing_cols}. Available columns: {df.columns.tolist()}")
        return pd.DataFrame()
    else:
        logger.info("All required columns successfully found in hazard table after renaming.")

    # The DataFrame 'df' is now the result, no role expansion needed.
    result = df.copy() # Work on a copy to avoid SettingWithCopyWarning

    # Ensure correct column types
    if SIMULATION_YEAR in result.columns:
        result[SIMULATION_YEAR] = result[SIMULATION_YEAR].astype(int)
    if EMP_LEVEL in result.columns:
        result[EMP_LEVEL] = result[EMP_LEVEL].astype(int)

    # Log tenure bands in hazard table (no standardization needed - hazard table should have correct format)
    if EMP_TENURE_BAND in result.columns:
        logger.info(f"Tenure bands in hazard table: {sorted(result[EMP_TENURE_BAND].unique())}")
    else:
        logger.warning(f"'{EMP_TENURE_BAND}' column not found in hazard table.")

    # Ensure all key rate columns are numeric and handle potential NaNs by filling with 0
    # This is a safeguard, ideally the Parquet file should have clean data.
    # Updated to include new granular compensation columns
    rate_columns = [TERM_RATE, NEW_HIRE_TERMINATION_RATE, COLA_PCT]

    # Handle both old and new compensation column schemas
    if 'comp_raise_pct' in result.columns:
        rate_columns.append('comp_raise_pct')
    else:
        # New schema with granular raise columns
        granular_raise_columns = ['merit_raise_pct', 'promotion_raise_pct', 'promotion_rate']
        for col in granular_raise_columns:
            if col in result.columns:
                rate_columns.append(col)

    for col in rate_columns:
        if col in result.columns:
            # Use infer_objects before fillna to avoid downcasting warning
            numeric_series = pd.to_numeric(result[col], errors='coerce')
            inferred_series = numeric_series.infer_objects(copy=False)
            result[col] = inferred_series.fillna(0.0)
            logger.info(f"Processed column '{col}': ensured numeric, NaNs filled with 0.0")
        else:
            logger.warning(f"Rate column '{col}' not found in hazard table. It might be optional or missing.")

    # For backward compatibility, create comp_raise_pct if it doesn't exist but granular columns do
    if 'comp_raise_pct' not in result.columns and 'merit_raise_pct' in result.columns:
        # Use merit_raise_pct as the primary annual raise component
        # promotion_raise_pct is handled separately in promotion logic
        result['comp_raise_pct'] = result['merit_raise_pct']
        logger.info("Created comp_raise_pct column from merit_raise_pct for backward compatibility")

    logger.info(f"Final processed hazard table has {len(result)} rows")

    # Check for and remove duplicates in the processed hazard table
    if EMP_LEVEL in result.columns and EMP_TENURE_BAND in result.columns and SIMULATION_YEAR in result.columns:
        # Check for duplicates within each year
        initial_rows = len(result)

        # Group by year and check for duplicates within each year
        years_with_duplicates = []
        for year in result[SIMULATION_YEAR].unique():
            year_data = result[result[SIMULATION_YEAR] == year]
            year_unique = year_data.drop_duplicates(subset=[EMP_LEVEL, EMP_TENURE_BAND])
            if len(year_unique) < len(year_data):
                years_with_duplicates.append(year)
                duplicates_count = len(year_data) - len(year_unique)
                logger.debug(f"Found {duplicates_count} duplicate (level, tenure_band) combinations in hazard table for year {year}")

        # Remove duplicates across the entire table
        result = result.drop_duplicates(subset=[SIMULATION_YEAR, EMP_LEVEL, EMP_TENURE_BAND])
        final_rows = len(result)

        if final_rows < initial_rows:
            removed_count = initial_rows - final_rows
            logger.info(f"Removed {removed_count} duplicate (year, level, tenure_band) combinations from hazard table")
            if years_with_duplicates:
                logger.debug(f"Years with duplicates: {years_with_duplicates}")

        unique_combos = result[[EMP_LEVEL, EMP_TENURE_BAND]].drop_duplicates()
        logger.info(f"Processed hazard table has {len(unique_combos)} unique (employee_level, tenure_band) combinations")
        logger.debug(f"Unique combinations: {unique_combos.to_dict('records')}")

    return result
