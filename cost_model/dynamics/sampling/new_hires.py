# cost_model/dynamics/sampling/new_hires.py
"""
Functions related to sampling new hires during workforce simulations.
QuickStart: see docs/cost_model/dynamics/sampling/new_hires.md
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from numpy.random import Generator, default_rng


def sample_new_hire_compensation(
    df: pd.DataFrame,
    comp_col: str,
    prev_salaries: Sequence[float],
    rng: Optional[Generator] = None,
    *,
    replace: bool = True,
) -> pd.DataFrame:
    """
    Sample starting compensation for new hires by drawing from historical salaries.

    Args:
        df:             DataFrame with new-hire rows (can be empty).
        comp_col:       Name of the column to set (will be created or overwritten).
        prev_salaries:  1D sequence (list/array/Series) of historical salaries.
        rng:            Optional numpy Generator for reproducibility. If None, a new Generator is seeded randomly.
        replace:        Whether to sample with replacement (default True).

    Returns:
        A copy of `df` with `comp_col` filled from `prev_salaries`.
    """
    df = df.copy()

    # ensure we have a Generator
    if rng is None:
        rng = default_rng()

    prev_salaries_arr = np.asarray(prev_salaries, dtype=float)
    if prev_salaries_arr.size == 0:
        raise ValueError("prev_salaries must contain at least one historical salary")

    # draw samples
    sampled = rng.choice(prev_salaries_arr, size=len(df), replace=replace)

    # assign back as a new column
    df[comp_col] = pd.Series(sampled, index=df.index)

    return df
