from typing import Sequence, Optional, List, Dict, Any, Union
import pandas as pd
import logging

from cost_model.state.schema import (
    EMP_LEVEL,
    EMP_GROSS_COMP,
    EMP_LEVEL_SOURCE,
    EMP_TENURE,
    EMP_TENURE_BAND
)
from cost_model.state.schema import SNAPSHOT_COLS, SNAPSHOT_DTYPES

logger = logging.getLogger(__name__)

def infer_job_level_by_percentile(
    df: pd.DataFrame,
    salary_col: str = EMP_GROSS_COMP,
    level_percentiles: Sequence[float] = (0.20, 0.50, 0.80, 0.95),
    target_level_col: str = 'imputed_level',
    source_col: str = EMP_LEVEL_SOURCE
) -> pd.DataFrame:
    """
    Impute job levels for rows where they're missing using global compensation percentiles.

    Args:
        df: Input DataFrame with employee data
        salary_col: Column name containing salary/compensation data
        level_percentiles: Sequence of percentiles defining level boundaries
                         (e.g., [0.2, 0.5, 0.8, 0.95] for 5 levels)
        target_level_col: Column name to store the imputed levels
        source_col: Column name to track the source of level assignments

    Returns:
        DataFrame with imputed levels and source tracking

    Note:
        - For edge cases (0 or 1 row), returns level 0
        - Uses quantile interpolation for small datasets to maintain consistency
    """
    # Input validation
    if salary_col not in df.columns:
        raise ValueError(f"Salary column '{salary_col}' not found in dataframe")

    # Handle empty dataframe
    if len(df) == 0:
        out = df.copy()
        out[target_level_col] = pd.Series(dtype="Int64")
        if source_col not in out.columns:
            out[source_col] = pd.Series(dtype="category")
        return out

    # Handle single row case
    if len(df) == 1:
        out = df.copy()
        out[target_level_col] = 0
        # Only set source if not already set
        if source_col not in out.columns or out[source_col].isna().all():
            out[source_col] = "percentile-impute"
        return out

    # Create working copy
    out = df.copy()

    # Calculate percentiles and assign levels
    out["_pct"] = out[salary_col].rank(pct=True)
    bins = [0.0, *sorted(level_percentiles), 1.0]  # Ensure percentiles are sorted
    labels = list(range(1, len(bins)))  # Start from 1 instead of 0 to match hazard table
    out[target_level_col] = pd.cut(
        out["_pct"],
        bins=bins,
        labels=labels,
        include_lowest=True
    ).astype('Int64')

    # Initialize source column if needed
    if source_col not in out.columns:
        out[source_col] = pd.NA

    # Determine which rows need level imputation
    needs_imputation = pd.Series(True, index=out.index)

    # Check for existing level columns
    level_col = None
    level_cols_to_check = [EMP_LEVEL, 'level_id', 'job_level']  # Common column names for job levels
    for col in level_cols_to_check:
        if col in out.columns:
            level_col = col
            break

    if level_col is None:
        # No existing level column found, mark all as imputed
        out[source_col] = "percentile-impute"
    else:
        # Only mark rows with missing levels as imputed
        missing_mask = out[level_col].isna()
        out.loc[missing_mask, source_col] = "percentile-impute"

    # Clean up temporary columns and return
    return out.drop(columns=["_pct"], errors='ignore')


