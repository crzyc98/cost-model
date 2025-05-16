from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import logging
import yaml
import numpy as np
import pandas as pd

from cost_model.utils.columns import (
    EMP_ID,
    EMP_GROSS_COMP, 
    EMP_LEVEL, 
    EMP_LEVEL_SOURCE,
    EMP_TENURE,
    EMP_TENURE_BAND,
    EMP_ACTIVE
)
from cost_model.state.schema import (
    SNAPSHOT_COLS,
    SNAPSHOT_DTYPES,
    EVENT_COLS
)
from cost_model.state.job_levels.models import JobLevel, ConfigError
from cost_model.state.job_levels.intervals import check_for_overlapping_bands as _check_for_overlapping_bands
from cost_model.state.job_levels.utils import assign_levels_to_dataframe
from cost_model.state.job_levels.engine import infer_job_level_by_percentile

logger = logging.getLogger(__name__)

def load_job_levels_from_config(config: Dict[str, Any], strict_validation: bool = True) -> Dict[int, JobLevel]:
    """Load job levels from a configuration dictionary."""
    levels: Dict[int, JobLevel] = {}
    for data in config.get("job_levels", []):
        level_id = data.get("level_id")
        if level_id is None:
            msg = "Job level missing 'level_id'"
            if strict_validation:
                raise ConfigError(msg)
            else:
                logger.warning(msg)
                continue
        try:
            level = JobLevel(
                level_id=int(level_id),
                name=data.get("name", f"Level {level_id}"),
                description=data.get("description", ""),
                min_compensation=float(data.get("min_compensation", 0)),
                max_compensation=float(data.get("max_compensation", 0)),
                comp_base_salary=float(data.get("comp_base_salary", 0)),
                comp_age_factor=float(data.get("comp_age_factor", 0)),
                comp_stochastic_std_dev=float(data.get("comp_stochastic_std_dev", 0.1)),
                mid_compensation=data.get("mid_compensation"),
                avg_annual_merit_increase=float(data.get("avg_annual_merit_increase", 0.03)),
                promotion_probability=float(data.get("promotion_probability", 0.1)),
                target_bonus_percent=float(data.get("target_bonus_percent", 0.0)),
                source=data.get("source", "band"),
                job_families=data.get("job_families", [])
            )
            levels[level.level_id] = level
        except (TypeError, ValueError) as e:
            msg = f"Invalid data for level {level_id}: {e}"
            if strict_validation:
                raise ConfigError(msg)
            else:
                logger.warning(msg)
    if not levels:
        raise ConfigError("No job levels found in configuration")
    # Validate overlaps
    _check_for_overlapping_bands(levels, strict=strict_validation)
    return levels

def load_from_yaml(path: Union[str, Path], strict_validation: bool = True) -> Dict[int, JobLevel]:
    """Load job levels from a YAML file."""
    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return load_job_levels_from_config(config, strict_validation=strict_validation)
    except (yaml.YAMLError, OSError) as e:
        msg = f"Error loading job levels from {path}: {e}"
        raise ConfigError(msg)

# ---------------------- DataFrame ingestion with imputation ----------------------

def ingest_with_imputation(
    raw_df: 'pd.DataFrame', 
    comp_col: str = EMP_GROSS_COMP,
    level_col: str = 'level_id',
    target_level_col: str = EMP_LEVEL
) -> 'pd.DataFrame':
    """
    Assign static bands, then impute missing levels by compensation percentile.
    
    Args:
        raw_df: Input DataFrame with employee data
        comp_col: Column name for compensation (default: EMP_GROSS_COMP)
        level_col: Source column for job levels if available (default: 'level_id')
        target_level_col: Target column for the final job levels (default: EMP_LEVEL)
        
    Returns:
        DataFrame with employee levels populated in the target_level_col and 'job_level'
    """
    # Make a copy of the input dataframe to avoid modifying the original
    df = raw_df.copy()
    
    # First, assign levels using the static bands
    df = assign_levels_to_dataframe(df, comp_col, target_level_col)
    
    # Then, perform percentile-based imputation for any remaining NAs
    if df[target_level_col].isna().any():
        # Create a temporary column for imputed levels
        df = infer_job_level_by_percentile(df, comp_col)
        
        # Fill any remaining NAs with the imputed levels
        na_mask = df[target_level_col].isna()
        if 'imputed_level' in df.columns and na_mask.any():
            df.loc[na_mask, target_level_col] = df.loc[na_mask, 'imputed_level']
    
    # Ensure the target column is of integer type
    df[target_level_col] = pd.to_numeric(df[target_level_col], errors='coerce').astype('Int64')
    
    # Track source of the level assignment
    if EMP_LEVEL_SOURCE not in df.columns:
        df[EMP_LEVEL_SOURCE] = pd.NA
    
    # Update source information
    assigned_mask = df[target_level_col].notna()
    
    # Mark levels that came from the input level_col
    if level_col in df.columns and not df[level_col].isna().all():
        from_band = df[level_col].notna() & assigned_mask
        df.loc[from_band, EMP_LEVEL_SOURCE] = 'salary-band'
    
    # Mark levels that were imputed
    from_imputed = assigned_mask & (~from_band if 'from_band' in locals() else assigned_mask)
    df.loc[from_imputed, EMP_LEVEL_SOURCE] = 'percentile-impute'
    
    # Clean up any temporary columns
    df = df.drop(columns=['imputed_level'], errors='ignore')
    
    # Ensure we have the required columns
    required_cols = [EMP_ID, target_level_col, EMP_LEVEL_SOURCE]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns after processing: {missing_cols}")
    
    return df[required_cols]
