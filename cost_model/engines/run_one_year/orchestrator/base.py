# cost_model/engines/run_one_year/orchestrator/base.py
"""
Base module for the orchestrator package.

Contains core data structures and stateless helper functions.
This module must not import from other orchestrator modules to prevent circular dependencies.
"""
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np

from cost_model.state.schema import EMP_ID, SIMULATION_YEAR


@dataclass
class YearContext:
    """
    Context object containing all parameters needed for a single year simulation.
    
    This centralizes the year-specific data and configuration to avoid passing
    many parameters between orchestrator methods.
    """
    year: int
    as_of: pd.Timestamp
    end_of_year: pd.Timestamp
    year_rng: np.random.Generator
    hazard_slice: pd.DataFrame
    global_params: Any
    plan_rules: Dict[str, Any]
    census_template_path: Optional[str] = None
    deterministic_term: bool = False
    
    @classmethod
    def create(
        cls,
        year: int,
        hazard_slice: pd.DataFrame,
        global_params: Any,
        plan_rules: Dict[str, Any],
        rng: np.random.Generator,
        rng_seed_offset: int = 0,
        census_template_path: Optional[str] = None,
        deterministic_term: bool = False
    ) -> 'YearContext':
        """
        Factory method to create a YearContext with derived values.
        
        Args:
            year: Simulation year
            hazard_slice: Hazard table slice for this year
            global_params: Global simulation parameters
            plan_rules: Plan configuration rules
            rng: Base random number generator
            rng_seed_offset: Offset for year-specific RNG seeding
            census_template_path: Path to census template
            deterministic_term: Whether to use deterministic terminations
            
        Returns:
            Configured YearContext instance
        """
        # Create year-specific RNG
        year_seed = hash((year, rng_seed_offset)) % (2**32)
        year_rng = np.random.default_rng(year_seed)
        
        # Create timestamps
        as_of = pd.Timestamp(f"{year}-01-01")
        end_of_year = pd.Timestamp(f"{year}-12-31")
        
        return cls(
            year=year,
            as_of=as_of,
            end_of_year=end_of_year,
            year_rng=year_rng,
            hazard_slice=hazard_slice,
            global_params=global_params,
            plan_rules=plan_rules,
            census_template_path=census_template_path,
            deterministic_term=deterministic_term
        )


def safe_get_meta(meta_str: str, key: str, default: Any = None) -> Any:
    """
    Safely extract a value from a JSON metadata string.
    
    Args:
        meta_str: JSON string containing metadata
        key: Key to extract from the JSON
        default: Default value if key is not found or JSON is invalid
        
    Returns:
        The extracted value or the default
    """
    if pd.isna(meta_str) or not meta_str:
        return default
    try:
        meta = json.loads(meta_str)
        return meta.get(key, default)
    except (json.JSONDecodeError, TypeError):
        return default


def is_valid_employee_id(emp_id: Any) -> bool:
    """
    Check if an employee ID is valid (not None, NA, or empty string).
    
    Args:
        emp_id: Employee ID to validate
        
    Returns:
        True if the employee ID is valid, False otherwise
    """
    try:
        return emp_id is not None and not pd.isna(emp_id) and str(emp_id).strip() != ''
    except Exception:
        return False


def filter_valid_employee_ids(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Filter a DataFrame to only include rows with valid employee IDs.
    
    Args:
        df: DataFrame to filter
        logger: Logger for warnings
        
    Returns:
        Filtered DataFrame with only valid employee IDs
    """
    if df.empty:
        return df
        
    # Ensure EMP_ID is a column and not part of the index
    if EMP_ID in df.index.names and EMP_ID not in df.columns:
        df = df.reset_index(level=EMP_ID)
    
    # Create a mask for valid employee IDs
    valid_mask = df[EMP_ID].apply(is_valid_employee_id)
    invalid_count = len(df) - sum(valid_mask)
    
    if invalid_count > 0:
        logger.warning(
            f"Found {invalid_count} records with invalid employee IDs. "
            f"Filtering them out. Sample of invalid IDs: {df[~valid_mask][EMP_ID].head(5).tolist()}"
        )
        df = df[valid_mask].copy()
    
    if df.empty:
        logger.error("No valid employee records left after filtering invalid employee IDs")
        raise ValueError("No valid employee records with valid employee IDs found after filtering")
    
    return df


def generate_unique_employee_ids(count: int, existing_ids: set, prefix: str = "EMP") -> List[str]:
    """
    Generate unique employee IDs that don't conflict with existing ones.
    
    Args:
        count: Number of IDs to generate
        existing_ids: Set of existing employee IDs to avoid
        prefix: Prefix for generated IDs
        
    Returns:
        List of unique employee ID strings
    """
    new_ids = []
    attempts = 0
    max_attempts = count * 10  # Prevent infinite loops
    
    while len(new_ids) < count and attempts < max_attempts:
        new_id = f"{prefix}_{uuid.uuid4().hex[:8]}"
        if new_id not in existing_ids and new_id not in new_ids:
            new_ids.append(new_id)
        attempts += 1
    
    if len(new_ids) < count:
        raise RuntimeError(f"Could not generate {count} unique employee IDs after {max_attempts} attempts")
    
    return new_ids


def standardize_dataframe_dtypes(df: pd.DataFrame, target_dtypes: Dict[str, Any]) -> pd.DataFrame:
    """
    Standardize DataFrame column dtypes to match target schema.
    
    Args:
        df: DataFrame to standardize
        target_dtypes: Dictionary mapping column names to target dtypes
        
    Returns:
        DataFrame with standardized dtypes
    """
    if df.empty:
        return df
        
    df_copy = df.copy()
    
    for col, dtype in target_dtypes.items():
        if col in df_copy.columns:
            try:
                df_copy[col] = df_copy[col].astype(dtype)
            except (ValueError, TypeError) as e:
                logging.warning(f"Could not convert column {col} to {dtype}: {e}")
    
    return df_copy


def ensure_simulation_year_column(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Ensure the DataFrame has a simulation_year column set to the given year.
    
    Args:
        df: DataFrame to update
        year: Simulation year to set
        
    Returns:
        DataFrame with simulation_year column
    """
    if df.empty:
        return df
        
    df_copy = df.copy()
    df_copy[SIMULATION_YEAR] = year
    return df_copy
