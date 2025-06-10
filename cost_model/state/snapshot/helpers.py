"""
Helper functions for snapshot processing.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .constants import EMP_ID, SNAPSHOT_COLS, SNAPSHOT_DTYPES

logger = logging.getLogger(__name__)


def get_first_event(events: pd.DataFrame, event_type: str) -> pd.DataFrame:
    """
    Returns the first occurrence of the given event_type for each employee.

    Args:
        events: DataFrame containing events
        event_type: Type of event to filter for

    Returns:
        DataFrame with the first event of the specified type for each employee
    """
    if events.empty or event_type not in events["event_type"].values:
        return pd.DataFrame(columns=events.columns)

    return events[events["event_type"] == event_type].drop_duplicates(subset=[EMP_ID], keep="first")


def get_last_event(events: pd.DataFrame, event_type: str) -> pd.DataFrame:
    """
    Returns the last occurrence of the given event_type for each employee.

    Args:
        events: DataFrame containing events
        event_type: Type of event to filter for

    Returns:
        DataFrame with the last event of the specified type for each employee
    """
    if events.empty or event_type not in events["event_type"].values:
        return pd.DataFrame(columns=events.columns)

    return events[events["event_type"] == event_type].drop_duplicates(subset=[EMP_ID], keep="last")


def ensure_columns_and_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures DataFrame has all specified columns with correct dtypes, adding NA columns if missing.
    For numeric columns, uses np.nan for missing values to avoid astype errors.

    Args:
        df: DataFrame to ensure columns and types

    Returns:
        DataFrame with all required columns and correct types
    """
    result = df.copy()

    # Add missing columns
    for col, dtype in SNAPSHOT_DTYPES.items():
        if col not in result.columns:
            if pd.api.types.is_datetime64_any_dtype(dtype):
                result[col] = pd.NaT
            elif dtype == pd.BooleanDtype():
                result[col] = pd.NA
            elif pd.api.types.is_numeric_dtype(dtype):
                result[col] = np.nan  # Use np.nan for numeric columns
            else:
                result[col] = pd.NA

    # Select only the columns we need, in the correct order
    result = result[SNAPSHOT_COLS]

    # Replace pd.NA with np.nan in numeric columns before astype
    for col, dtype in SNAPSHOT_DTYPES.items():
        if pd.api.types.is_numeric_dtype(dtype) and col in result.columns:
            result[col] = result[col].replace({pd.NA: np.nan})

    # Apply the correct types
    result = result.astype(SNAPSHOT_DTYPES)

    return result


def validate_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates a snapshot DataFrame, checking for index uniqueness and other requirements.

    Args:
        df: DataFrame to validate

    Returns:
        Validated DataFrame
    """
    result = df.copy()

    # Ensure index is unique
    if not result.index.is_unique:
        dups = result.index[result.index.duplicated()].unique().tolist()
        logger.error(f"Duplicate EMP_IDs in snapshot: {dups} – dropping duplicates (keeping last)")
        result = result[~result.index.duplicated(keep="last")]

    # Check for NaN in index
    if result.index.hasnans:
        logger.error("NaN values found in snapshot index. This should not happen!")
        # Remove rows with NaN in index
        result = result[~result.index.isna()]

    return result
