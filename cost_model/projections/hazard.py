# cost_model/projections/hazard.py
"""
Hazard module for generating hazard tables used in workforce projections.
QuickStart: see docs/cost_model/projections/hazard.md
"""

import pandas as pd
import logging
import os
from typing import List, Dict, Optional, Union, Tuple
from cost_model.utils.tenure_utils import standardize_tenure_band
from cost_model.state.schema import (
    EMP_LEVEL,
    EMP_TENURE_BAND,
    SIMULATION_YEAR,
    TERM_RATE,
    COMP_RAISE_PCT,
    NEW_HIRE_TERM_RATE,
    COLA_PCT,
    CFG
)

logger = logging.getLogger(__name__)

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
    standard_tenure_bands = {'0-1', '1-3', '3-5', '5+'}
    
    # Create a set of all possible (level, tenure_band) combinations
    all_combinations = set()
    
    # First, add all combinations from the initial snapshot
    for combo in unique_levels_tenures:
        # Ensure tenure band is standardized
        std_tenure = standardize_tenure_band(combo[EMP_TENURE_BAND])
        all_combinations.add((combo[EMP_LEVEL], std_tenure))
    
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
                COMP_RAISE_PCT: global_comp_raise_pct,
                NEW_HIRE_TERM_RATE: global_nh_term_rate,
                COLA_PCT: getattr(global_params, 'cola_pct', 0.0),
                CFG: plan_rules_config
            })
    if records:
        df = pd.DataFrame(records)
        logger.info(f"Hazard table with {len(records)} rows.")
    else:
        cols = [SIMULATION_YEAR, EMP_LEVEL, EMP_TENURE_BAND, TERM_RATE, COMP_RAISE_PCT, NEW_HIRE_TERM_RATE, COLA_PCT, CFG]
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
    
    logger.info(f"Initial columns from Parquet: {df.columns.tolist()}")

    # Define rename map for columns that differ from constants used internally.
    # Assumes constants (EMP_TENURE_BAND, NEW_HIRE_TERM_RATE, etc.) are imported from cost_model.constants or similar.
    rename_map = {
        # Parquet_column_name: internal_constant_name
        'tenure_band': EMP_TENURE_BAND,                 # e.g., 'tenure_band' -> 'employee_tenure_band'
        'new_hire_termination_rate': NEW_HIRE_TERM_RATE, # e.g., 'new_hire_termination_rate' -> 'new_hire_term_rate'
        'termination_rate': TERM_RATE,                  # e.g., 'termination_rate' -> 'termination_rate'
        'compensation_raise_percentage': COMP_RAISE_PCT, # e.g., 'compensation_raise_percentage' -> 'compensation_raise_percentage'
        'employee_level': EMP_LEVEL,                    # e.g., 'employee_level' -> 'employee_level'
        'simulation_year': SIMULATION_YEAR              # e.g., 'simulation_year' -> 'simulation_year'
        # Add other mappings if Parquet names differ from internal constants or to standardize (e.g. case differences).
    }
    
    # Filter rename_map to only include columns actually present in the DataFrame's current columns.
    # This avoids errors if a mapped Parquet column is unexpectedly missing.
    actual_rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    
    if actual_rename_map:
        logger.info(f"Applying rename map: {actual_rename_map}")
        df = df.rename(columns=actual_rename_map)
        logger.info(f"Columns after renaming: {df.columns.tolist()}")
    else:
        logger.info("No columns to rename based on the current DataFrame and rename_map. Ensure Parquet column names match keys in rename_map if renaming is expected.")

    # Now, check for required columns using the standardized internal constant names.
    required_cols = [
        SIMULATION_YEAR, 
        EMP_LEVEL,
        EMP_TENURE_BAND, 
        TERM_RATE, 
        COMP_RAISE_PCT,
        NEW_HIRE_TERM_RATE
    ]
    # Ensure consistency in column name expectations.
    
    # Check for missing required columns after the rename attempt.
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.error(f"Hazard table missing required columns AFTER RENAME ATTEMPT: {missing_cols}. Available columns: {df.columns.tolist()}")
        return pd.DataFrame()

    # The DataFrame 'df' is now the result, no role expansion needed.
    result = df.copy() # Work on a copy to avoid SettingWithCopyWarning
    
    # Ensure correct column types
    if SIMULATION_YEAR in result.columns:
        result[SIMULATION_YEAR] = result[SIMULATION_YEAR].astype(int)
    if EMP_LEVEL in result.columns:
        result[EMP_LEVEL] = result[EMP_LEVEL].astype(int)
        
    # Standardize tenure bands to ensure consistent format
    if EMP_TENURE_BAND in result.columns:
        logger.info("Standardizing tenure band formats in hazard table")
        result[EMP_TENURE_BAND] = result[EMP_TENURE_BAND].map(standardize_tenure_band)
        logger.info(f"Unique tenure bands after standardization: {result[EMP_TENURE_BAND].unique().tolist()}")
    else:
        logger.warning(f"'{EMP_TENURE_BAND}' column not found in hazard table. Skipping standardization.")

    # Ensure all key rate columns are numeric and handle potential NaNs by filling with 0
    # This is a safeguard, ideally the Parquet file should have clean data.
    rate_columns = [TERM_RATE, COMP_RAISE_PCT, NEW_HIRE_TERM_RATE, COLA_PCT]
    for col in rate_columns:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors='coerce').fillna(0.0)
            logger.info(f"Processed column '{col}': ensured numeric, NaNs filled with 0.0")
        else:
            logger.warning(f"Rate column '{col}' not found in hazard table. It might be optional or missing.")
            # If a critical rate column like TERM_RATE is missing and not handled, it could cause issues downstream.
            # However, NEW_HIRE_TERM_RATE was added to required_cols, so it should be present.
            # COLA_PCT might be optional.

    logger.info(f"Final processed hazard table has {len(result)} rows")
    
    # Log the unique combinations in the processed hazard table
    if EMP_LEVEL in result.columns and EMP_TENURE_BAND in result.columns:
        unique_combos = result[[EMP_LEVEL, EMP_TENURE_BAND]].drop_duplicates()
        logger.info(f"Processed hazard table has {len(unique_combos)} unique (employee_level, tenure_band) combinations")
        logger.debug(f"Unique combinations: {unique_combos.to_dict('records')}")
    
    return result
