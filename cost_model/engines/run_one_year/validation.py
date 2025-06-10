"""
Validation module for run_one_year package.

Contains functions for validating and ensuring consistency of inputs.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from cost_model.state.schema import EMP_ACTIVE  # Use the actual constant name
from cost_model.state.schema import (
    EMP_BIRTH_DATE,
    EMP_EXITED,
    EMP_GROSS_COMP,
    EMP_HIRE_DATE,
    EMP_ID,
    EMP_LEVEL,
    EMP_LEVEL_SOURCE,
)
from cost_model.state.schema import EMP_TENURE_BAND as TENURE_BAND
from cost_model.state.schema import (
    EMP_TERM_DATE,
)
from cost_model.utils.tenure_utils import standardize_tenure_band

logger = logging.getLogger(__name__)
from cost_model.state.schema import EMP_DEFERRAL_RATE, EMP_TENURE

# Required columns with default values for snapshot DataFrame
# This should match the schema defined in cost_model/state/schema.py
REQUIRED_SNAPSHOT_DEFAULTS = {
    # Required fields (no defaults)
    EMP_ID: None,  # must exist, no default
    EMP_HIRE_DATE: None,  # must exist, no default
    EMP_BIRTH_DATE: None,  # must exist, no default
    EMP_GROSS_COMP: None,  # must exist, no default
    # Optional fields with defaults
    EMP_LEVEL: 1,  # Default to level 1 if not specified
    EMP_LEVEL_SOURCE: "manual",  # Default source for job level
    EMP_TERM_DATE: pd.NaT,  # Not terminated by default
    EMP_ACTIVE: True,  # Active by default
    EMP_DEFERRAL_RATE: 0.0,  # No deferral by default
    TENURE_BAND: "0-1",  # Default tenure band
    EMP_TENURE: 0.0,  # 0 years of tenure by default
    EMP_EXITED: False,  # Not exited by default
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

    # Legacy role column migration removed as part of schema refactoring

    # Verify or add required columns
    for col, default in REQUIRED_SNAPSHOT_DEFAULTS.items():
        if col not in snap_copy.columns:

            if default is None:
                raise ValueError(f"Required column {col} missing from snapshot")
            snap_copy[col] = default
            logging.info(f"Added missing required column {col} with default {default}")

    return snap_copy


def validate_and_extract_hazard_slice(hazard_table: pd.DataFrame, year: int) -> pd.DataFrame:
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

    # Role-based logic removed as part of schema refactoring

    # Log available tenure bands (role logging removed)

    # Standardize tenure bands to ensure consistent matching
    if TENURE_BAND in hazard_slice.columns:
        # Before standardization
        original_tenure_bands = list(hazard_slice[TENURE_BAND].unique())
        logging.info(
            f"[RUN_ONE_YEAR YR={year}] hazard_slice original tenure_bands: {original_tenure_bands}"
        )

        # Apply standardization
        hazard_slice[TENURE_BAND] = hazard_slice[TENURE_BAND].apply(standardize_tenure_band)

        # After standardization
        standardized_tenure_bands = list(hazard_slice[TENURE_BAND].unique())
        logging.info(
            f"[RUN_ONE_YEAR YR={year}] hazard_slice standardized tenure_bands: {standardized_tenure_bands}"
        )

    return hazard_slice


def nonempty_frames(*frames: Union[pd.DataFrame, List[pd.DataFrame]]):
    """
    Filters out empty DataFrames from a list of DataFrames.

    Args:
        *frames: Variable number of DataFrames or lists of DataFrames

    Returns:
        List of non-empty DataFrames
    """
    result = []
    for frame in frames:
        if isinstance(frame, pd.DataFrame):
            if not frame.empty:
                result.append(frame)
        elif isinstance(frame, list):
            result.extend([f for f in frame if isinstance(f, pd.DataFrame) and not f.empty])
    return result


def validate_eoy_snapshot(final_snap: pd.DataFrame, target: Optional[int] = None) -> None:
    """
    Validates the end-of-year snapshot for consistency and correctness.

    Args:
        final_snap: The end-of-year snapshot DataFrame to validate
        target: Expected number of active employees. If None, skips headcount validation.

    Raises:
        AssertionError: If any validation checks fail
    """
    logger.info("Validating end-of-year snapshot...")

    # Check for duplicate employee IDs
    duplicate_ids = final_snap[EMP_ID].duplicated()
    if duplicate_ids.any():
        duplicate_count = duplicate_ids.sum()
        duplicates = final_snap[final_snap[EMP_ID].duplicated(keep=False)].sort_values(EMP_ID)
        logger.error(f"Found {duplicate_count} duplicate EMP_IDs in snapshot")
        logger.error(f"Duplicate IDs:\n{duplicates[[EMP_ID, EMP_ACTIVE, EMP_TERM_DATE]].head(10)}")
        if len(duplicates) > 10:
            logger.error(f"... and {len(duplicates) - 10} more")
        raise ValueError(f"Found {duplicate_count} duplicate EMP_IDs in snapshot")

    # Check active employee count if target is provided
    if target is not None:
        active_count = final_snap[EMP_ACTIVE].sum()
        if active_count != target:
            logger.error(
                f"Active employee count mismatch. Expected: {target}, Actual: {active_count}"
            )
            raise ValueError(f"EOY headcount {active_count} does not match target {target}")

    # Check for invalid active/terminated states
    invalid_active = final_snap[final_snap[EMP_ACTIVE] & ~pd.isna(final_snap[EMP_TERM_DATE])]
    if not invalid_active.empty:
        logger.error(f"Found {len(invalid_active)} employees marked as both active and terminated")
        raise ValueError(
            f"Found {len(invalid_active)} employees with both active=True and a termination date"
        )

    logger.info("End-of-year snapshot validation passed")
