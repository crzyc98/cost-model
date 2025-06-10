# cost_model/state/tenure.py
"""Tenure‐related helper functions for snapshots.

QuickStart: see docs/cost_model/state/tenure.md
"""
from __future__ import annotations

from enum import Enum
from typing import List, Tuple

import pandas as pd


class TenureBand(Enum):
    """Enumeration of tenure bands for employee categorization."""

    NEW_HIRE = "<1"
    EARLY = "1-3"
    CORE = "3-5"
    EXPERIENCED = "5-10"
    VETERAN = "10-15"
    NEAR_RETIREE = "15+"


# Tenure cutoffs: (min_years, max_years, TenureBand)
# Each tuple defines the range [min_years, max_years) for the given tenure band
TENURE_CUTOFFS: List[Tuple[float, float, TenureBand]] = [
    (0.0, 1.0, TenureBand.NEW_HIRE),
    (1.0, 3.0, TenureBand.EARLY),
    (3.0, 5.0, TenureBand.CORE),
    (5.0, 10.0, TenureBand.EXPERIENCED),
    (10.0, 15.0, TenureBand.VETERAN),
    (15.0, float("inf"), TenureBand.NEAR_RETIREE),
]

# Create categorical dtype for tenure bands
TENURE_BAND_CATEGORICAL_DTYPE = pd.CategoricalDtype(
    categories=["<1", "1-3", "3-5", "5-10", "10-15", "15+"], ordered=True
)

__all__ = [
    "TenureBand",
    "TENURE_CUTOFFS",
    "TENURE_BAND_CATEGORICAL_DTYPE",
    "assign_tenure_band",
    "categorize_tenure",
    "apply_tenure",
]


def categorize_tenure(tenure):
    """Categorize numeric tenure into a TenureBand enum value."""
    if pd.isna(tenure):
        return pd.NA

    for min_years, max_years, band in TENURE_CUTOFFS:
        if min_years <= tenure < max_years:
            return band

    # Fallback (should not happen with proper TENURE_CUTOFFS)
    return TenureBand.NEAR_RETIREE


def assign_tenure_band(tenure):
    """Map numeric tenure (years) to a categorical band string."""
    if pd.isna(tenure):
        return pd.NA

    band = categorize_tenure(tenure)
    if pd.isna(band):
        return pd.NA

    return band.value


def apply_tenure(
    df: pd.DataFrame, hire_col: str, as_of: pd.Timestamp, *, out_tenure_col: str, out_band_col: str
) -> pd.DataFrame:
    """Vectorised tenure + band calculation and assignment.

    Parameters
    ----------
    df : DataFrame to modify in place.
    hire_col : column containing hire dates (datetime64[ns]).
    as_of : timestamp to measure tenure against.
    out_tenure_col : name for numeric tenure column (will be float).
    out_band_col : name for categorical band column.
    """
    hire_dates = pd.to_datetime(df[hire_col], errors="coerce")
    tenure_yrs = (as_of - hire_dates).dt.days / 365.25
    df[out_tenure_col] = tenure_yrs.round(3)
    df[out_band_col] = (
        df[out_tenure_col].map(assign_tenure_band).astype(TENURE_BAND_CATEGORICAL_DTYPE)
    )
    return df
