# utils/sampling/terminations.py

import pandas as pd
import numpy as np
from typing import Union, Optional
from numpy.random import Generator, default_rng
import logging
from utils.columns import EMP_HIRE_DATE, EMP_TERM_DATE, STATUS_COL

logger = logging.getLogger(__name__)

def sample_terminations(
    df: pd.DataFrame,
    hire_col: str,
    termination_rate: Union[float, pd.Series],
    year_end: pd.Timestamp,
    rng: Optional[Generator] = None,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Assign EMP_TERM_DATE (and mark STATUS_COL) for a DataFrame of employees.

    Args:
        df: DataFrame containing a hire date column.
        hire_col: Name of the hire‐date column (must be datetime).
        termination_rate: 
          - float: same probability for everyone, or 
          - Series: per‐row probability, aligned on df.index.
        year_end: Timestamp marking the end of the year (inclusive).
        rng: Optional numpy.random.Generator for reproducibility.
        seed: Optional int seed for reproducibility.

    Returns:
        A new DataFrame with:
        - 'EMP_TERM_DATE' set to hire_date + random offset for those who terminate
        - 'STATUS_COL' set to 'Terminated' for those rows
    """
    df = df.copy()
    # Initialize RNG: use provided Generator or seed
    if rng is None:
        rng = default_rng(seed)

    # Ensure hire dates are datetime
    hires = pd.to_datetime(df[hire_col], errors='coerce')

    # Days from hire to end of year (>=1 for same‐day hires)
    days_until = (year_end - hires).dt.days.clip(lower=0).fillna(0).astype(int) + 1

    # Build per-row probabilities
    if isinstance(termination_rate, pd.Series):
        probs = termination_rate.reindex(df.index).fillna(0.0).clip(0, 1)
    else:
        probs = pd.Series(float(termination_rate), index=df.index)

    # Draw uniform [0,1) and mask out old or invalid hires
    draws      = rng.random(len(df))
    already_term = df[EMP_TERM_DATE].notna()
    valid_hire   = hires.notna()
    mask         = ~already_term & valid_hire & (draws < probs.values)
    logger.info("sample_terminations: %d new terminations (of %d rows)", mask.sum(), len(df))

    if mask.any():
        # Random offset (0 to days_until-1)
        offsets = np.floor(rng.random(mask.sum()) * days_until[mask].values).astype(int)
        df.loc[mask, EMP_TERM_DATE] = hires[mask] + pd.to_timedelta(offsets, unit='D')
        df.loc[mask, STATUS_COL] = 'Terminated'
        # Ensure correct dtype
        df[EMP_TERM_DATE] = pd.to_datetime(df[EMP_TERM_DATE])

    return df