import pandas as pd
from numpy.random import Generator
import numpy as np
from typing import Optional

def sample_new_hire_compensation(
    df: pd.DataFrame,
    comp_col: str,
    prev_salaries: np.ndarray,
    rng: Generator
) -> pd.DataFrame:
    """
    Sample starting compensation for new hires by drawing from historical salaries.

    Args:
        df: DataFrame with new hire records.
        comp_col: Name of the compensation column to assign.
        prev_salaries: Array-like of past salaries to sample from.
        rng: Random number generator for reproducibility.

    Returns:
        DataFrame with updated compensation in `comp_col`.
    """
    df = df.copy()
    # draw new hire compensation from historical distribution
    df[comp_col] = rng.choice(prev_salaries, size=len(df), replace=True)
    return df
