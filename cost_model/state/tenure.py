"""Tenure‐related helper functions for snapshots.

QuickStart: see docs/cost_model/state/tenure.md
"""
from __future__ import annotations

import pandas as pd

__all__ = [
    "assign_tenure_band",
    "apply_tenure",
]

def assign_tenure_band(tenure: float | int | pd.NA) -> str | pd.NA:
    """Map numeric tenure (years) to a categorical band string."""
    if pd.isna(tenure):
        return pd.NA
    if tenure < 1:
        return "0-1"
    if tenure < 3:
        return "1-3"
    if tenure < 5:
        return "3-5"
    return "5+"

def apply_tenure(df: pd.DataFrame, hire_col: str, as_of: pd.Timestamp, *, out_tenure_col: str, out_band_col: str) -> pd.DataFrame:
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
    df[out_band_col] = df[out_tenure_col].map(assign_tenure_band).astype(pd.StringDtype())
    return df
