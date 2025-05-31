"""Age-related helper functions for snapshots."""

from __future__ import annotations
import pandas as pd
from enum import Enum
from typing import List, Tuple

# ────────────────────────────────────────────────────────────────────────────────
# A.  Age bands – tweak these once and the whole model follows
# ────────────────────────────────────────────────────────────────────────────────
class AgeBand(Enum):
    UNDER_30   = "<30"
    THIRTY     = "30-39"
    FORTY      = "40-49"
    FIFTY      = "50-59"
    PRE_RETIRE = "60-65"
    RETIRED    = "65+"

AGE_CUTOFFS: List[Tuple[int, int, AgeBand]] = [
    (   0, 30, AgeBand.UNDER_30),
    (  30, 40, AgeBand.THIRTY),
    (  40, 50, AgeBand.FORTY),
    (  50, 60, AgeBand.FIFTY),
    (  60, 65, AgeBand.PRE_RETIRE),
    (  65, 200, AgeBand.RETIRED),
]

AGE_BAND_CATEGORICAL_DTYPE = pd.CategoricalDtype(
    categories=["<30", "30-39", "40-49", "50-59", "60-65", "65+"],
    ordered=True,
)

# ────────────────────────────────────────────────────────────────────────────────
# B.  Helpers – identical pattern to tenure.py
# ────────────────────────────────────────────────────────────────────────────────
def categorize_age(age: float | pd.NA) -> AgeBand | pd.NA:
    if pd.isna(age):
        return pd.NA
    for lo, hi, band in AGE_CUTOFFS:
        if lo <= age < hi:
            return band
    return AgeBand.RETIRED

def assign_age_band(age):
    band = categorize_age(age)
    return pd.NA if pd.isna(band) else band.value

def apply_age(df: pd.DataFrame,
              birth_col: str,
              as_of: pd.Timestamp,
              *,
              out_age_col: str,
              out_band_col: str
) -> pd.DataFrame:
    """Vectorised age + band calculation (mutates df)."""
    bdates = pd.to_datetime(df[birth_col], errors="coerce")
    df[out_age_col]  = ((as_of - bdates).dt.days / 365.25).round(1)
    df[out_band_col] = df[out_age_col].map(assign_age_band).astype(AGE_BAND_CATEGORICAL_DTYPE)
    return df