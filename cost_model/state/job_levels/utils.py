"""Utility functions for job levels module."""

import logging
from typing import Dict, List, Optional, Union

import pandas as pd

from cost_model.state.schema import (
    EMP_GROSS_COMP,
    EMP_LEVEL,
    EMP_LEVEL_SOURCE,
    EMP_TENURE,
    EMP_TENURE_BAND,
    SNAPSHOT_COLS,
    SNAPSHOT_DTYPES,
)

from .models import JobLevel

logger = logging.getLogger(__name__)


def assign_levels_to_dataframe(
    df: pd.DataFrame, comp_column: str = EMP_GROSS_COMP, target_level_col: str = EMP_LEVEL
) -> pd.DataFrame:
    """Assign job levels to employees in a dataframe using vectorized pd.cut.

    Args:
        df: Input DataFrame with employee data
        comp_column: Column name containing compensation data
        target_level_col: Column to store the assigned levels

    Returns:
        DataFrame with assigned levels in the target_level_col

    Note:
        - For compensation below the lowest level's min, assigns the lowest level
        - For compensation above the highest level's max, assigns the highest level
        - For gaps between levels, assigns to the lower level
    """
    if target_level_col not in df.columns:
        df[target_level_col] = pd.NA

    # Only process rows where level is not already set
    # Handle potential duplicate columns by selecting the first one
    if target_level_col in df.columns:
        level_series = df[target_level_col]
        # If duplicate columns exist, pandas returns a DataFrame - take the first column
        if isinstance(level_series, pd.DataFrame):
            level_series = level_series.iloc[:, 0]
        mask = level_series.isna()
    else:
        # Column doesn't exist, all values need to be set
        mask = pd.Series([True] * len(df), index=df.index)

    if len(mask) == 0:
        return df
    if not mask.any():
        return df

    # Get the job levels configuration from the state module
    from . import state

    levels = state.LEVEL_TAXONOMY

    if not levels:
        logger.warning("No job levels configured - cannot assign levels")
        return df

    # Sort levels by min_compensation for proper binning
    sorted_levels = sorted(levels.values(), key=lambda x: x.min_compensation)

    # Get min and max values from the levels
    min_comp = min(level.min_compensation for level in sorted_levels)
    max_comp = max(level.max_compensation for level in sorted_levels)

    # Create bins with proper edges to handle the ranges inclusively
    # We use right=True to make bins closed on the right (i.e., (a, b])
    bins = [float("-inf")] + [level.max_compensation for level in sorted_levels]
    labels = [level.level_id for level in sorted_levels]

    # Create a copy of the compensation column for processing
    # Handle potential duplicate columns by selecting the first one
    comp_series = df[comp_column]
    if isinstance(comp_series, pd.DataFrame):
        comp_series = comp_series.iloc[:, 0]
    comp_values = comp_series.loc[mask].copy()

    # Log any values outside the expected range
    below_min = (comp_values < min_comp).sum()
    above_max = (comp_values > max_comp).sum()

    if below_min > 0:
        logger.warning(
            f"{below_min} employees have compensation below the minimum level. "
            f"Assigning to lowest level (ID: {sorted_levels[0].level_id})."
        )

    if above_max > 0:
        logger.warning(
            f"{above_max} employees have compensation above the maximum level. "
            f"Assigning to highest level (ID: {sorted_levels[-1].level_id})."
        )

    # Handle values below minimum by setting them to minimum
    comp_values = comp_values.clip(lower=min_comp, upper=max_comp)

    # Assign levels based on compensation
    # We use right=True to make bins closed on the right (i.e., (a, b])
    df.loc[mask, target_level_col] = pd.cut(
        comp_values, bins=bins, labels=labels, right=True, include_lowest=False
    )

    # Handle any values that couldn't be binned (shouldn't happen with clip)
    # but just in case, assign them to the closest level
    # Handle potential duplicate columns for level checking
    level_check_series = df[target_level_col]
    if isinstance(level_check_series, pd.DataFrame):
        level_check_series = level_check_series.iloc[:, 0]
    na_mask = level_check_series.isna() & mask

    if na_mask.any():
        logger.warning(
            f"Could not assign levels to {na_mask.sum()} employees. "
            "Assigning to closest level based on compensation."
        )

        for idx in df[na_mask].index:
            # Handle potential duplicate columns for compensation access
            if isinstance(comp_series, pd.DataFrame):
                comp = comp_series.iloc[0].at[idx]
            else:
                comp = comp_series.at[idx]
            # Find the level with the closest min_compensation
            closest_level = min(
                sorted_levels,
                key=lambda l: min(abs(l.min_compensation - comp), abs(l.max_compensation - comp)),
            )
            df.at[idx, target_level_col] = closest_level.level_id

    # Handle any remaining NAs by assigning to the closest level
    # Refresh the level check series
    level_check_series = df[target_level_col]
    if isinstance(level_check_series, pd.DataFrame):
        level_check_series = level_check_series.iloc[:, 0]
    na_mask = level_check_series.isna() & mask

    if na_mask.any():
        logger.warning(
            f"Could not assign levels to {na_mask.sum()} employees. "
            "Assigning to closest level based on compensation."
        )

        # For each NA, find the closest level by min_compensation
        for idx in df[na_mask].index:
            # Handle potential duplicate columns for compensation access
            if isinstance(comp_series, pd.DataFrame):
                comp = comp_series.iloc[0].at[idx]
            else:
                comp = comp_series.at[idx]
            # Find the level with the closest min_compensation
            closest_level = min(sorted_levels, key=lambda l: abs(l.min_compensation - comp))
            df.at[idx, target_level_col] = closest_level.level_id

    return df
