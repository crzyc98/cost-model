import pandas as pd
import numpy as np
from typing import Union, Optional
from numpy.random import Generator

def sample_terminations(
    df: pd.DataFrame,
    hire_col: str,
    termination_rate: Union[float, pd.Series],
    year_end: pd.Timestamp,
    rng: Generator
) -> pd.DataFrame:
    """
    Assign `termination_date` and `status` to rows in `df` probabilistically based on `termination_rate`.
    `termination_rate` can be a float (constant rate) or a Series of per-row probabilities.
    """
    df = df.copy()
    days_until_end = (year_end - df[hire_col]).dt.days.clip(lower=0) + 1
    # Use provided RNG for reproducible draws
    # Normalize termination_rate to per-row probabilities
    if isinstance(termination_rate, pd.Series):
        probs = termination_rate.reindex(df.index).fillna(0.0)
    else:
        probs = pd.Series(termination_rate, index=df.index)
    # Generate random offsets
    rand_floats = rng.random(len(df))
    rand_off = np.floor(rand_floats * days_until_end.values).astype(int)
    rand_td = pd.to_timedelta(rand_off, unit='D')
    # Sample terminations
    mask = rand_floats < probs.values
    df.loc[mask, 'termination_date'] = df.loc[mask, hire_col] + rand_td[mask]
    df.loc[mask, 'status'] = 'Terminated'
    return df
