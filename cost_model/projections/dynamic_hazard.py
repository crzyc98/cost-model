# cost_model/projections/dynamic_hazard.py
"""
Dynamic hazard table generation for auto-tuning system.

This module provides functionality to generate hazard tables at runtime using
detailed hazard parameters from global_params, replacing static file loading.

QuickStart: see docs/auto_calibration/epic_x.md
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional
import itertools

from cost_model.config.models import GlobalParameters
from cost_model.state.schema import (
    SIMULATION_YEAR,
    EMP_LEVEL,
    TERM_RATE,
    COLA_PCT,
    CFG,
    PROMOTION_RATE,
    MERIT_RAISE_PCT,
    PROMOTION_RAISE_PCT,
    NEW_HIRE_TERMINATION_RATE,
    EMP_TENURE_BAND  # Use EMP_TENURE_BAND instead of TENURE_BAND for engine compatibility
)

logger = logging.getLogger(__name__)


def build_dynamic_hazard_table(
    global_params: GlobalParameters,
    simulation_years: List[int],
    job_levels: List[int],
    tenure_bands: List[str],
    cfg_scenario_name: str = "baseline"
) -> pd.DataFrame:
    """
    Dynamically generate hazard table DataFrame at runtime using global_params.
    
    This function replaces static hazard table loading by calculating all hazard
    rates and multipliers from the detailed parameters in global_params that are
    modified by the auto-tuning system.
    
    Args:
        global_params: GlobalParameters object containing detailed hazard configurations
        simulation_years: List of simulation years to generate hazard table for
        job_levels: List of job levels (numeric)
        tenure_bands: List of tenure band strings (e.g., ["<1", "1-3", "3-5", ...])
        cfg_scenario_name: Scenario name for the 'cfg' column
        
    Returns:
        DataFrame with hazard table schema matching what simulation engines expect:
        - simulation_year, employee_level, employee_tenure_band
        - term_rate, promotion_rate, merit_raise_pct, promotion_raise_pct
        - cola_pct, new_hire_termination_rate, cfg
        
    Raises:
        ValueError: If required hazard configurations are missing from global_params
    """
    logger.info(f"Building dynamic hazard table for {len(simulation_years)} years, "
                f"{len(job_levels)} levels, {len(tenure_bands)} tenure bands")
    
    # Validate required configurations exist
    _validate_global_params(global_params)
    
    # Generate all combinations
    all_combinations = list(itertools.product(simulation_years, job_levels, tenure_bands))
    logger.debug(f"Generated {len(all_combinations)} year-level-tenure combinations")
    
    # Create base DataFrame
    df = pd.DataFrame(all_combinations, columns=[SIMULATION_YEAR, EMP_LEVEL, EMP_TENURE_BAND])
    
    # Calculate each hazard rate column
    df = _calculate_termination_rates(df, global_params)
    df = _calculate_promotion_rates(df, global_params)
    df = _calculate_merit_raise_rates(df, global_params)
    df = _calculate_promotion_raise_rates(df, global_params)
    df = _calculate_cola_rates(df, global_params)
    df = _calculate_new_hire_termination_rates(df, global_params)
    
    # Add configuration column
    df[CFG] = cfg_scenario_name

    # Note: Using EMP_TENURE_BAND to match what simulation engines expect
    # (load_and_expand_hazard_table normally renames tenure_band to employee_tenure_band)

    # Debug: Check columns before finalization
    logger.debug(f"Columns before finalization: {df.columns.tolist()}")

    # Ensure proper column order and data types
    df = _finalize_hazard_table_schema(df)
    
    logger.info(f"Successfully built dynamic hazard table with {len(df)} rows")
    return df


def _validate_global_params(global_params: GlobalParameters) -> None:
    """Validate that required hazard configurations exist in global_params."""
    missing_configs = []
    
    if not hasattr(global_params, 'termination_hazard') or global_params.termination_hazard is None:
        missing_configs.append('termination_hazard')
    if not hasattr(global_params, 'promotion_hazard') or global_params.promotion_hazard is None:
        missing_configs.append('promotion_hazard')
    if not hasattr(global_params, 'raises_hazard') or global_params.raises_hazard is None:
        missing_configs.append('raises_hazard')
    if not hasattr(global_params, 'cola_hazard') or global_params.cola_hazard is None:
        missing_configs.append('cola_hazard')
        
    if missing_configs:
        raise ValueError(f"Missing required hazard configurations in global_params: {missing_configs}")
    
    logger.debug("Global params validation passed - all required hazard configurations present")


def _calculate_termination_rates(df: pd.DataFrame, global_params: GlobalParameters) -> pd.DataFrame:
    """
    Calculate term_rate column using termination_hazard configuration.
    
    Formula: base_rate_for_new_hire * tenure_multipliers[band] * level_discount_effect
    """
    logger.debug("Calculating termination rates from global_params.termination_hazard")
    
    term_config = global_params.termination_hazard
    base_rate = term_config.base_rate_for_new_hire
    tenure_multipliers = term_config.tenure_multipliers
    level_discount_factor = term_config.level_discount_factor
    min_level_discount_multiplier = term_config.min_level_discount_multiplier
    
    logger.debug(f"Using base_rate_for_new_hire: {base_rate}")
    logger.debug(f"Using tenure_multipliers: {tenure_multipliers}")
    logger.debug(f"Using level_discount_factor: {level_discount_factor}, min_multiplier: {min_level_discount_multiplier}")
    
    # Apply tenure multipliers
    df['tenure_multiplier'] = df[EMP_TENURE_BAND].map(tenure_multipliers)

    # Handle missing tenure bands with warning
    missing_tenure_mask = df['tenure_multiplier'].isna()
    if missing_tenure_mask.any():
        missing_bands = df.loc[missing_tenure_mask, EMP_TENURE_BAND].unique()
        logger.warning(f"Missing tenure multipliers for bands: {missing_bands}. Using 1.0 as default.")
        df['tenure_multiplier'] = df['tenure_multiplier'].fillna(1.0)
    
    # Calculate level discount effect
    # level_discount_effect = max(min_multiplier, 1 - level_discount_factor * (level - 1))
    level_discount_effect = np.maximum(
        min_level_discount_multiplier,
        1 - level_discount_factor * (df[EMP_LEVEL] - 1)
    )
    
    # Calculate final termination rate
    df[TERM_RATE] = base_rate * df['tenure_multiplier'] * level_discount_effect
    
    # Clean up temporary column
    df = df.drop(columns=['tenure_multiplier'])
    
    logger.debug(f"Calculated termination rates - range: [{df[TERM_RATE].min():.4f}, {df[TERM_RATE].max():.4f}]")
    return df


def _calculate_new_hire_termination_rates(df: pd.DataFrame, global_params: GlobalParameters) -> pd.DataFrame:
    """
    Calculate new_hire_termination_rate column.

    Uses global_params.new_hire_termination_rate if available,
    otherwise checks attrition.new_hire_termination_rate,
    otherwise falls back to termination_hazard.base_rate_for_new_hire.
    """
    logger.debug("Calculating new hire termination rates")

    # Try multiple locations for new_hire_termination_rate (matching hazard.py logic)
    nh_term_rate = None

    # First try: direct attribute at root level
    if hasattr(global_params, 'new_hire_termination_rate') and global_params.new_hire_termination_rate is not None:
        nh_term_rate = global_params.new_hire_termination_rate
        logger.debug(f"Using global_params.new_hire_termination_rate: {nh_term_rate}")
    # Second try: nested under attrition
    elif hasattr(global_params, 'attrition') and hasattr(global_params.attrition, 'new_hire_termination_rate'):
        nh_term_rate = global_params.attrition.new_hire_termination_rate
        logger.debug(f"Using global_params.attrition.new_hire_termination_rate: {nh_term_rate}")
    # Fallback: use termination_hazard base rate
    else:
        nh_term_rate = global_params.termination_hazard.base_rate_for_new_hire
        logger.debug(f"Fallback to termination_hazard.base_rate_for_new_hire: {nh_term_rate}")

    df[NEW_HIRE_TERMINATION_RATE] = nh_term_rate

    logger.debug(f"Set new_hire_termination_rate to {nh_term_rate} for all rows")
    return df


def _calculate_promotion_rates(df: pd.DataFrame, global_params: GlobalParameters) -> pd.DataFrame:
    """
    Calculate promotion_rate column using promotion_hazard configuration.

    Formula: base_rate * tenure_multipliers[band] * level_dampener_effect
    """
    logger.debug("Calculating promotion rates from global_params.promotion_hazard")

    promo_config = global_params.promotion_hazard
    base_rate = promo_config.base_rate
    tenure_multipliers = promo_config.tenure_multipliers
    level_dampener_factor = promo_config.level_dampener_factor

    logger.debug(f"Using promotion base_rate: {base_rate}")
    logger.debug(f"Using promotion tenure_multipliers: {tenure_multipliers}")
    logger.debug(f"Using level_dampener_factor: {level_dampener_factor}")

    # Apply tenure multipliers
    df['promo_tenure_multiplier'] = df[EMP_TENURE_BAND].map(tenure_multipliers)

    # Handle missing tenure bands with warning
    missing_tenure_mask = df['promo_tenure_multiplier'].isna()
    if missing_tenure_mask.any():
        missing_bands = df.loc[missing_tenure_mask, EMP_TENURE_BAND].unique()
        logger.warning(f"Missing promotion tenure multipliers for bands: {missing_bands}. Using 1.0 as default.")
        df['promo_tenure_multiplier'] = df['promo_tenure_multiplier'].fillna(1.0)

    # Calculate level dampener effect
    # level_dampener_effect = max(0, 1 - level_dampener_factor * (level - 1))
    level_dampener_effect = np.maximum(
        0.0,  # Promotion rate can be dampened to 0
        1 - level_dampener_factor * (df[EMP_LEVEL] - 1)
    )

    # Calculate base promotion rate
    df[PROMOTION_RATE] = base_rate * df['promo_tenure_multiplier'] * level_dampener_effect

    # Set promotion rate to 0 for the highest level (assuming max level can't be promoted)
    max_level = df[EMP_LEVEL].max()
    df.loc[df[EMP_LEVEL] == max_level, PROMOTION_RATE] = 0.0
    logger.debug(f"Set promotion_rate to 0 for highest level: {max_level}")

    # Clean up temporary column
    df = df.drop(columns=['promo_tenure_multiplier'])

    logger.debug(f"Calculated promotion rates - range: [{df[PROMOTION_RATE].min():.4f}, {df[PROMOTION_RATE].max():.4f}]")
    return df


def _calculate_merit_raise_rates(df: pd.DataFrame, global_params: GlobalParameters) -> pd.DataFrame:
    """
    Calculate merit_raise_pct column using raises_hazard configuration.

    Formula: merit_base + tenure_bump + level_bump
    """
    logger.debug("Calculating merit raise rates from global_params.raises_hazard")

    raises_config = global_params.raises_hazard
    merit_base = raises_config.merit_base
    merit_tenure_bump_bands = raises_config.merit_tenure_bump_bands
    merit_tenure_bump_value = raises_config.merit_tenure_bump_value
    merit_low_level_cutoff = raises_config.merit_low_level_cutoff
    merit_low_level_bump_value = raises_config.merit_low_level_bump_value

    logger.debug(f"Using merit_base: {merit_base}")
    logger.debug(f"Using tenure bump bands: {merit_tenure_bump_bands}, value: {merit_tenure_bump_value}")
    logger.debug(f"Using low level cutoff: {merit_low_level_cutoff}, bump: {merit_low_level_bump_value}")

    # Start with base merit rate
    df[MERIT_RAISE_PCT] = merit_base

    # Add tenure bump for eligible bands
    if merit_tenure_bump_bands:
        tenure_bump_condition = df[EMP_TENURE_BAND].isin(merit_tenure_bump_bands)
        df.loc[tenure_bump_condition, MERIT_RAISE_PCT] += merit_tenure_bump_value
        logger.debug(f"Applied tenure bump to {tenure_bump_condition.sum()} rows")

    # Add low-level bump
    level_bump_condition = df[EMP_LEVEL] <= merit_low_level_cutoff
    df.loc[level_bump_condition, MERIT_RAISE_PCT] += merit_low_level_bump_value
    logger.debug(f"Applied low-level bump to {level_bump_condition.sum()} rows")

    logger.debug(f"Calculated merit raise rates - range: [{df[MERIT_RAISE_PCT].min():.4f}, {df[MERIT_RAISE_PCT].max():.4f}]")
    return df


def _calculate_promotion_raise_rates(df: pd.DataFrame, global_params: GlobalParameters) -> pd.DataFrame:
    """
    Calculate promotion_raise_pct column using raises_hazard configuration.

    Uses raises_hazard.promotion_raise as a uniform rate for all promotions.
    """
    logger.debug("Calculating promotion raise rates from global_params.raises_hazard")

    promotion_raise = global_params.raises_hazard.promotion_raise
    logger.debug(f"Using promotion_raise: {promotion_raise}")

    df[PROMOTION_RAISE_PCT] = promotion_raise

    logger.debug(f"Set promotion_raise_pct to {promotion_raise} for all rows")
    return df


def _calculate_cola_rates(df: pd.DataFrame, global_params: GlobalParameters) -> pd.DataFrame:
    """
    Calculate cola_pct column using cola_hazard configuration.

    Maps simulation_year to cola_hazard.by_year values.
    """
    logger.debug("Calculating COLA rates from global_params.cola_hazard")

    cola_by_year = global_params.cola_hazard.by_year
    logger.debug(f"Using COLA by year: {cola_by_year}")

    # Map years to COLA rates
    df[COLA_PCT] = df[SIMULATION_YEAR].map(cola_by_year)

    # Handle missing years with warning and default
    missing_cola_mask = df[COLA_PCT].isna()
    if missing_cola_mask.any():
        missing_years = df.loc[missing_cola_mask, SIMULATION_YEAR].unique()
        default_cola = 0.02  # 2% default COLA
        logger.warning(f"Missing COLA rates for years: {missing_years}. Using default {default_cola}")
        df[COLA_PCT] = df[COLA_PCT].fillna(default_cola)

    logger.debug(f"Calculated COLA rates - range: [{df[COLA_PCT].min():.4f}, {df[COLA_PCT].max():.4f}]")
    return df


def _finalize_hazard_table_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure proper column order and data types for hazard table output.

    Matches the schema of existing hazard_table.parquet file.
    """
    logger.debug("Finalizing hazard table schema and data types")

    # Define expected column order (matching what simulation engines expect)
    expected_columns = [
        SIMULATION_YEAR,           # simulation_year
        EMP_LEVEL,                 # employee_level
        EMP_TENURE_BAND,          # employee_tenure_band (what engines expect after load_and_expand_hazard_table)
        TERM_RATE,                # term_rate
        PROMOTION_RATE,           # promotion_rate
        MERIT_RAISE_PCT,          # merit_raise_pct
        PROMOTION_RAISE_PCT,      # promotion_raise_pct
        COLA_PCT,                 # cola_pct
        NEW_HIRE_TERMINATION_RATE,  # new_hire_termination_rate (NOT new_hire_term_rate)
        CFG                       # cfg
    ]

    # Reorder columns
    df = df[expected_columns]

    # Ensure proper data types
    df[SIMULATION_YEAR] = df[SIMULATION_YEAR].astype('int64')
    df[EMP_LEVEL] = df[EMP_LEVEL].astype('int64')
    df[EMP_TENURE_BAND] = df[EMP_TENURE_BAND].astype('object')
    df[TERM_RATE] = df[TERM_RATE].astype('float64')
    df[PROMOTION_RATE] = df[PROMOTION_RATE].astype('float64')
    df[MERIT_RAISE_PCT] = df[MERIT_RAISE_PCT].astype('float64')
    df[PROMOTION_RAISE_PCT] = df[PROMOTION_RAISE_PCT].astype('float64')
    df[COLA_PCT] = df[COLA_PCT].astype('float64')
    df[NEW_HIRE_TERMINATION_RATE] = df[NEW_HIRE_TERMINATION_RATE].astype('float64')
    df[CFG] = df[CFG].astype('object')

    logger.debug(f"Finalized hazard table with columns: {df.columns.tolist()}")
    logger.debug(f"Data types: {df.dtypes.to_dict()}")

    return df
