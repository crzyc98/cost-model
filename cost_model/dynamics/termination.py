# cost_model/dynamics/termination.py
"""
Functions related to simulating employee terminations.
QuickStart: /docs/cost_model/dynamics/termination.md
"""

import logging
import pandas as pd
import numpy as np
from typing import Union, Optional

# Use relative imports for other modules within the dynamics package if needed
try:
    from .sampling.terminations import sample_terminations  # Assumes this exists
    from .sampling.salary import (
        SalarySampler,
        DefaultSalarySampler,
    )  # If needed for salary sampling
except ImportError:
    print("Warning (termination.py): Could not import sampling helpers.")

    # Define dummy functions/classes if needed
    def sample_terminations(df, *args, **kwargs):
        return df  # Passthrough

    class DefaultSalarySampler:
        def sample_terminations(self, *args, **kwargs):
            return None  # Return None if sampling fails


# Use absolute imports for modules outside dynamics
try:
    from cost_model.utils.columns import EMP_TERM_DATE, EMP_HIRE_DATE, EMP_GROSS_COMP
except ImportError:
    print("Warning (termination.py): Could not import column constants.")
    EMP_TERM_DATE, EMP_HIRE_DATE, EMP_GROSS_COMP = (
        "employee_termination_date",
        "employee_hire_date",
        "employee_gross_compensation",
    )


logger = logging.getLogger(__name__)


# Renamed from _apply_turnover
def apply_turnover(
    df: pd.DataFrame,
    hire_col: str,
    probs_or_rate: Union[float, pd.Series, np.ndarray],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    rng: np.random.Generator,
    # Optional: sampler and prev_term_salaries if sampling term comp
    sampler: Optional[SalarySampler] = None,
    prev_term_salaries: Optional[pd.Series] = None,
    log: Optional[logging.Logger] = None,  # Accept logger instance
) -> pd.DataFrame:
    """
    Applies terminations to a DataFrame based on probabilities or a flat rate.
    Updates the EMP_TERM_DATE column. Optionally samples termination compensation.

    Args:
        df: DataFrame with employee data.
        hire_col: Name of the hire date column.
        probs_or_rate: A float rate, or a Series/ndarray of probabilities per employee.
        start_date: The start date of the period during which terminations occur.
        end_date: The end date of the period during which terminations occur.
        rng: NumPy random number generator.
        sampler: Optional SalarySampler for termination compensation.
        prev_term_salaries: Optional Series of previous termination salaries for sampling base.
        log: Optional logger instance.

    Returns:
        DataFrame with termination dates potentially updated.
    """
    local_log = log or logger  # Use passed logger or module logger

    # Call the underlying sampling function (which should handle masking etc.)
    # sample_terminations should return df with EMP_TERM_DATE updated
    local_log.debug(
        f"Applying turnover for period {start_date.date()} to {end_date.date()}..."
    )
    df_with_terms = sample_terminations(df, hire_col, probs_or_rate, end_date, rng)

    # --- Optional: Sample Termination Compensation ---
    # Check if sampler and previous salaries are provided
    if sampler and prev_term_salaries is not None and not prev_term_salaries.empty:
        term_mask = df_with_terms[EMP_TERM_DATE].between(
            start_date, end_date, inclusive="both"
        )
        # Also check if the term date was newly assigned in this step (optional, depends on sample_terminations logic)
        # Example: term_mask = term_mask & df[EMP_TERM_DATE].isna() # Only sample for newly termed

        n_term_in_period = term_mask.sum()
        if n_term_in_period > 0:
            local_log.debug(
                f"Sampling termination compensation for {n_term_in_period} employees."
            )
            try:
                draws = sampler.sample_terminations(
                    prev_term_salaries, size=n_term_in_period, rng=rng
                )
                # Assign sampled compensation
                # Ensure index alignment if draws is Series
                df_with_terms.loc[term_mask, EMP_GROSS_COMP] = (
                    draws if isinstance(draws, np.ndarray) else draws.values
                )
            except Exception:
                local_log.exception("Error during termination compensation sampling.")
        else:
            local_log.debug(
                "No terminations within the specified period to sample compensation for."
            )
    elif sampler and (prev_term_salaries is None or prev_term_salaries.empty):
        local_log.warning(
            "Termination sampler provided, but prev_term_salaries is missing or empty. Cannot sample termination compensation."
        )

    return df_with_terms
