# utils/sampling/terminations.py

import pandas as pd
import numpy as np
from typing import Union, Optional
from numpy.random import Generator, default_rng
import logging
from utils.columns import EMP_HIRE_DATE, EMP_TERM_DATE, STATUS_COL # Ensure these are correct

logger = logging.getLogger(__name__)

def sample_terminations(
    df: pd.DataFrame,
    hire_col: str,
    termination_rate: Union[float, int, pd.Series, np.ndarray], # Updated type hint
    year_end: pd.Timestamp,
    rng: Optional[Generator] = None,
    seed: Optional[int] = None
) -> pd.DataFrame:
    # ... (Keep the first part including RNG init and the modified probability handling block) ...

    # --- Restore previous logic for termination date calculation and masking ---
    df_out = df.copy()
    # Initialize RNG
    if rng is None:
        rng = default_rng(seed)

    # Ensure hire dates are datetime
    hires = pd.to_datetime(df_out.get(hire_col), errors='coerce')
    if hires is None:
        logger.error(f"Hire column '{hire_col}' not found. Cannot sample.")
        return df_out

    # Days from hire to end of year (>=1 for same‐day hires)
    days_until = (year_end - hires).dt.days.clip(lower=0).fillna(0).astype(int) + 1

    # --- Keep the modified probability handling block from your latest version here ---
    # Determine probabilities for sampling (copied from your latest version)
    if isinstance(termination_rate, pd.Series):
        if termination_rate.shape[0] != df_out.shape[0]:
             logger.warning(f"Termination probability Series shape {termination_rate.shape} doesn't match DataFrame shape {df_out.shape}. Reindexing.")
        probs = termination_rate.reindex(df_out.index).fillna(0.0)
        logger.debug("Using provided Series of termination probabilities.")
    elif isinstance(termination_rate, np.ndarray):
        if termination_rate.shape[0] != df_out.shape[0]:
             logger.error(f"Termination rate NumPy array shape {termination_rate.shape} incorrect for DataFrame shape {df_out.shape}. Using 0%.")
             probs = pd.Series(0.0, index=df_out.index)
        else:
             probs = pd.Series(termination_rate, index=df_out.index)
             logger.debug("Using provided NumPy array of termination rates.")
    elif isinstance(termination_rate, (float, int)):
        rate = float(termination_rate)
        logger.debug(f"Applying uniform termination rate: {rate:.2%}")
        probs = pd.Series(rate, index=df_out.index)
    else:
         logger.error(f"Unexpected type for termination_rate: {type(termination_rate)}. Using 0%.")
         probs = pd.Series(0.0, index=df_out.index)
    probs = probs.clip(0.0, 1.0)
    # --- End probability handling block ---

    # Draw uniform [0,1)
    draws = rng.random(len(df_out))

    # Mask out invalid hires or those already terminated
    # Ensure EMP_TERM_DATE exists
    if EMP_TERM_DATE not in df_out.columns:
        df_out[EMP_TERM_DATE] = pd.NaT

    already_term = df_out[EMP_TERM_DATE].notna()
    valid_hire   = hires.notna()

    # Determine who terminates using previous robust logic
    terminated_mask = ~already_term & valid_hire & (draws < probs.values)

    n_terminated = terminated_mask.sum()
    # Use more informative log message
    logger.info(f"sample_terminations: {n_terminated} new terminations (of {(~already_term & valid_hire).sum()} currently active, valid hires)")

    if n_terminated > 0:
        # Restore random offset logic for termination date
        offsets = np.floor(rng.random(n_terminated) * days_until[terminated_mask].values).astype(int)
        term_dates = hires[terminated_mask] + pd.to_timedelta(offsets, unit='D')

        # Assign termination date and status using .loc
        df_out.loc[terminated_mask, EMP_TERM_DATE] = term_dates
        # Ensure STATUS_COL exists
        if STATUS_COL not in df_out.columns:
            df_out[STATUS_COL] = 'Active' # Or appropriate default
        df_out.loc[terminated_mask, STATUS_COL] = 'Terminated'

        # Ensure correct dtype after assignment
        df_out[EMP_TERM_DATE] = pd.to_datetime(df_out[EMP_TERM_DATE])
        logger.debug(f"Assigned random termination dates and status for {n_terminated} employees.")

    return df_out