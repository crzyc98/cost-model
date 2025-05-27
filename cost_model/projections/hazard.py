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

    # Ensure all required employee levels (0-4) are represented in the hazard table
    # This is a safety net to make sure we don't miss any levels
    all_employee_levels = set(range(5))  # 0, 1, 2, 3, 4
    existing_levels = set(combo[EMP_LEVEL] for combo in unique_levels_tenures)
    missing_levels = all_employee_levels - existing_levels
    
    if missing_levels:
        logger.warning(f"Initial snapshot missing employee levels: {missing_levels}. Adding to hazard table.")
        for level in missing_levels:
            for tenure_band in set(combo[EMP_TENURE_BAND] for combo in unique_levels_tenures):
                unique_levels_tenures.append({EMP_LEVEL: level, EMP_TENURE_BAND: tenure_band})
    
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
    Load hazard table from parquet file and expand role='*' entries to all employee levels.
    
    Args:
        path: Path to the hazard table parquet file
        
    Returns:
        DataFrame with expanded hazard table entries for all employee levels
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
    
    # Check if we have the required columns
    required_cols = ['simulation_year', 'role', 'tenure_band', 'term_rate', 'comp_raise_pct']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        logger.error(f"Hazard table missing required columns: {missing_cols}")
        return pd.DataFrame()
    
    # Find rows with role='*' which need to be expanded
    wildcard_mask = df['role'] == '*'
    wildcard_rows = df[wildcard_mask].copy()
    non_wildcard_rows = df[~wildcard_mask].copy()
    
    if wildcard_rows.empty:
        logger.warning("No role='*' entries found in hazard table to expand")
        # If no employee_level column, add it with default value
        if 'employee_level' not in df.columns:
            logger.warning("Adding default employee_level=1 to all hazard table entries")
            df['employee_level'] = 1
        return df
    
    logger.info(f"Found {len(wildcard_rows)} role='*' entries to expand across employee levels")
    
    # Create expanded rows for all employee levels (0-4)
    all_levels = list(range(5))  # 0, 1, 2, 3, 4
    expanded_rows = []
    
    for _, row in wildcard_rows.iterrows():
        for level in all_levels:
            new_row = row.copy()
            new_row['employee_level'] = level
            expanded_rows.append(new_row)
    
    # Create DataFrame from expanded rows
    expanded_df = pd.DataFrame(expanded_rows)
    
    # If non_wildcard_rows doesn't have employee_level, add it with default value
    if 'employee_level' not in non_wildcard_rows.columns and not non_wildcard_rows.empty:
        non_wildcard_rows['employee_level'] = 1
    
    # Combine expanded and non-wildcard rows
    result = pd.concat([expanded_df, non_wildcard_rows], ignore_index=True)
    
    # Ensure correct column types
    if 'simulation_year' in result.columns:
        result['simulation_year'] = result['simulation_year'].astype(int)
    if 'employee_level' in result.columns:
        result['employee_level'] = result['employee_level'].astype(int)
        
    # Standardize tenure bands to ensure consistent format
    if 'tenure_band' in result.columns:
        logger.info("Standardizing tenure band formats in hazard table")
        result['tenure_band'] = result['tenure_band'].map(standardize_tenure_band)
        logger.info(f"Unique tenure bands after standardization: {result['tenure_band'].unique().tolist()}")
    
    logger.info(f"Final expanded hazard table has {len(result)} rows")
    
    # Log the unique combinations in the expanded hazard table
    if 'employee_level' in result.columns and 'tenure_band' in result.columns:
        unique_combos = result[['employee_level', 'tenure_band']].drop_duplicates()
        logger.info(f"Expanded hazard table has {len(unique_combos)} unique (employee_level, tenure_band) combinations")
        logger.debug(f"Unique combinations: {unique_combos.to_dict('records')}")
    
    return result
