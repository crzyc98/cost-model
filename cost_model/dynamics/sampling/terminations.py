# cost_model/dynamics/sampling/terminations.py
"""
Functions related to sampling terminations during workforce simulations.
QuickStart: see docs/cost_model/dynamics/sampling/terminations.md
"""

import pandas as pd
import numpy as np
from typing import Union, Optional
from numpy.random import Generator, default_rng
import logging
from cost_model.utils.columns import (
    EMP_TERM_DATE,
    STATUS_COL,
)  # Ensure these are correct

logger = logging.getLogger(__name__)


def sample_terminations(
    df: pd.DataFrame,
    hire_col: str,
    termination_rate: Union[float, int, pd.Series, np.ndarray],  # Updated type hint
    year_end: pd.Timestamp,
    rng: Optional[Generator] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    # ... (Keep the first part including RNG init and the modified probability handling block) ...

    # --- Restore previous logic for termination date calculation and masking ---
    df_out = df.copy()
    # Initialize RNG
    if rng is None:
        rng = default_rng(seed)

    # Ensure hire dates are datetime
    hires = pd.to_datetime(df_out.get(hire_col), errors="coerce")
    if hires is None:
        logger.error(f"Hire column '{hire_col}' not found. Cannot sample.")
        return df_out

    # Days from hire to end of year (>=1 for same‐day hires)
    days_until = (year_end - hires).dt.days.clip(lower=0).fillna(0).astype(int) + 1

    # --- Keep the modified probability handling block from your latest version here ---
    # Determine probabilities for sampling (copied from your latest version)
    if isinstance(termination_rate, pd.Series):
        if termination_rate.shape[0] != df_out.shape[0]:
            logger.warning(
                f"Termination probability Series shape {termination_rate.shape} doesn't match DataFrame shape {df_out.shape}. Reindexing."
            )
        probs = termination_rate.reindex(df_out.index).fillna(0.0)
        logger.debug("Using provided Series of termination probabilities.")
    elif isinstance(termination_rate, np.ndarray):
        if termination_rate.shape[0] != df_out.shape[0]:
            logger.error(
                f"Termination rate NumPy array shape {termination_rate.shape} incorrect for DataFrame shape {df_out.shape}. Using 0%."
            )
            probs = pd.Series(0.0, index=df_out.index)
        else:
            probs = pd.Series(termination_rate, index=df_out.index)
            logger.debug("Using provided NumPy array of termination rates.")
    elif isinstance(termination_rate, (float, int)):
        rate = float(termination_rate)
        logger.debug(f"Applying uniform termination rate: {rate:.2%}")
        probs = pd.Series(rate, index=df_out.index)
    else:
        logger.error(
            f"Unexpected type for termination_rate: {type(termination_rate)}. Using 0%."
        )
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
    valid_hire = hires.notna()

    # --- Special-case float/int “annual” termination_rate: pick exactly N = ceil(rate×active), weighted by hazard ---
    import math
    if isinstance(termination_rate, (float, int)):
        valid_idx  = df_out.index[~already_term & valid_hire]
        total_valid= len(valid_idx)
        rate       = float(termination_rate)
        n_to_term  = math.ceil(total_valid * rate)
        logger.info(f"sample_terminations: terminating exactly {n_to_term} of {total_valid} per annual rate {rate:.2%}")

        if n_to_term > 0 and total_valid > 0:
            # — merge in per-row hazard (term_rate) if it’s on df_out —
            if 'term_rate' in df_out.columns:
                weights = df_out.loc[valid_idx, 'term_rate'].fillna(0.0)
                # normalize to sum=1
                probs_w = weights.div(weights.sum()) if weights.sum() > 0 else None
            else:
                probs_w = None

            # weighted sampling without replacement
            if probs_w is not None:
                losers = rng.choice(valid_idx, size=min(n_to_term, total_valid),
                                    replace=False, p=probs_w.values)
            else:
                losers = rng.choice(valid_idx, size=min(n_to_term, total_valid), replace=False)

            terminated_mask = df_out.index.isin(losers)
        else:
            terminated_mask = pd.Series(False, index=df_out.index)
    else:
        # Fallback to original per-row probability logic
        draws = rng.random(len(df_out))
        terminated_mask = ~already_term & valid_hire & (draws < probs.values)

    n_terminated = terminated_mask.sum()
    # Use more informative log message
    logger.info(
        f"sample_terminations: {n_terminated} new terminations (of {(~already_term & valid_hire).sum()} currently active, valid hires)"
    )

    if n_terminated > 0:
        # Restore random offset logic for termination date
        offsets = np.floor(
            rng.random(n_terminated) * days_until[terminated_mask].values
        ).astype(int)
        term_dates = hires[terminated_mask] + pd.to_timedelta(offsets, unit="D")

        # Ensure termination date is not before the start of the year and not after the end of the year
        year_start_dt = pd.Timestamp(year=year_end.year, month=1, day=1)
        term_dates = term_dates.clip(lower=year_start_dt, upper=year_end)

        # Defensive fix: Do not allow termination before hire date
        hire_dates = hires[terminated_mask]
        invalid_mask = term_dates < hire_dates
        if invalid_mask.any():
            logger.warning(f"{invalid_mask.sum()} terminations had term_date < hire_date. Setting term_date to NaT and not marking as terminated for these rows.")
            # For these, set term_date to NaT and do not mark as terminated
            # Only assign valid term dates
            valid_mask = ~invalid_mask
            # Assign only to valid rows
            df_out.loc[terminated_mask[terminated_mask].index[valid_mask], EMP_TERM_DATE] = term_dates[valid_mask]
            # Mark only valid as terminated
            if STATUS_COL not in df_out.columns:
                df_out[STATUS_COL] = "Active"
            df_out.loc[terminated_mask[terminated_mask].index[valid_mask], STATUS_COL] = "Terminated"
            # For invalid, leave as not terminated (Active/NaT)
        else:
            # Assign termination date and status using .loc
            df_out.loc[terminated_mask, EMP_TERM_DATE] = term_dates
            # Ensure STATUS_COL exists
            if STATUS_COL not in df_out.columns:
                df_out[STATUS_COL] = "Active"  # Or appropriate default
            df_out.loc[terminated_mask, STATUS_COL] = "Terminated"

        # Ensure correct dtype after assignment
        df_out[EMP_TERM_DATE] = pd.to_datetime(df_out[EMP_TERM_DATE])
        logger.debug(
            f"Assigned random termination dates and status for {n_terminated} employees."
        )

    return df_out
