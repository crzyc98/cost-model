# cost_model/dynamics/compensation.py
"""
Functions related to compensation changes during simulation dynamics.
"""

import logging
import pandas as pd
import numpy as np
from typing import Mapping, Any, Optional

# Attempt to import column constants, provide fallbacks
try:
    from ..utils.columns import EMP_GROSS_COMP
except ImportError:
    print(
        "Warning (compensation.py): Could not import column constants from utils. Using string literals."
    )
    EMP_GROSS_COMP = "employee_gross_compensation"

# Import sampling helpers if needed (e.g., for second-year bumps)
try:
    from .sampling.salary import SalarySampler, DefaultSalarySampler
except ImportError:
    print(
        "Warning (compensation.py): Could not import SalarySampler. Complex bumps might fail."
    )

    class DefaultSalarySampler:
        pass  # Dummy class


logger = logging.getLogger(__name__)


def apply_comp_bump(
    df: pd.DataFrame,
    comp_col: str,
    rate: float,
    rng: np.random.Generator,
    log: logging.Logger,
    dist: Optional[Mapping[str, Any]] = None,  # Optional config for complex bumps
    sampler: Optional[SalarySampler] = None,  # Optional sampler instance
) -> pd.DataFrame:
    """
    Applies annual compensation increase.

    Handles flat rate increases for experienced employees and potentially
    more complex sampling for second-year employees if configured.

    Args:
        df: DataFrame with employee data (must include comp_col and 'tenure').
        comp_col: Name of the compensation column to update.
        rate: The base annual increase rate (e.g., 0.03 for 3%).
        rng: NumPy random number generator.
        log: Logger instance.
        dist: Optional configuration for second-year bump distribution.
        sampler: Optional SalarySampler instance for complex bumps.

    Returns:
        DataFrame with updated compensation.
    """
    df2 = df.copy()
    if comp_col not in df2:
        log.warning(f"Compensation column '{comp_col}' not found, skipping comp bump.")
        return df2
    if "tenure" not in df2.columns:
        log.warning(
            "Column 'tenure' not found, cannot apply tenure-based comp bump logic. Applying flat rate to all."
        )
        df2[comp_col] = pd.to_numeric(df2[comp_col], errors="coerce").fillna(0.0) * (
            1 + rate
        )
        return df2

    log.debug(
        f"Applying comp bump. Base Rate={rate:.2%}. Current avg comp: {df2[comp_col].mean():.0f}"
    )

    # Ensure columns are numeric
    df2[comp_col] = (
        pd.to_numeric(df2[comp_col], errors="coerce").fillna(0.0).astype(float)
    )
    df2["tenure"] = pd.to_numeric(df2["tenure"], errors="coerce").fillna(0.0)

    # Define masks based on tenure
    mask_new_or_first_year = df2["tenure"] < 1.0  # Tenure is float, check < 1
    mask_second_year = (df2["tenure"] >= 1.0) & (df2["tenure"] < 2.0)
    mask_experienced = df2["tenure"] >= 2.0

    n_new = mask_new_or_first_year.sum()
    n_second = mask_second_year.sum()
    n_exp = mask_experienced.sum()
    log.debug(
        f"Comp Bump Groups: New/First={n_new}, Second={n_second}, Experienced={n_exp}"
    )

    # 1) New hires / First year employees: Typically no bump applied in this step
    # (Their salary was set during hiring/onboarding)
    if n_new > 0:
        log.debug(f"Skipping standard comp bump for {n_new} new/first-year employees.")

    # 2) Second-year employees: Potentially use sampler
    if n_second > 0:
        # Use sampler if provided and configured
        if sampler and dist and hasattr(sampler, "sample_second_year"):
            log.debug(f"Applying sampler bump to {n_second} second-year employees.")
            try:
                df2.loc[mask_second_year, comp_col] = sampler.sample_second_year(
                    df2.loc[mask_second_year],
                    comp_col=comp_col,
                    dist=dist,
                    rate=rate,
                    rng=rng,
                )
            except Exception:
                log.exception(
                    "Error during sampler.sample_second_year. Applying flat rate instead."
                )
                df2.loc[mask_second_year, comp_col] *= 1 + rate
        else:
            # Fallback to flat rate if no sampler/config
            log.debug(
                f"Applying flat rate bump ({rate:.1%}) to {n_second} second-year employees (no sampler/config)."
            )
            df2.loc[mask_second_year, comp_col] *= 1 + rate

    # 3) Experienced employees: Apply flat increase rate
    if n_exp > 0:
        log.debug(
            f"Applying flat rate bump ({rate:.1%}) to {n_exp} experienced employees."
        )
        df2.loc[mask_experienced, comp_col] *= 1 + rate

    # Ensure non-negative compensation
    df2[comp_col] = df2[comp_col].clip(lower=0)

    log.info("Comp bump applied. New avg comp: %.0f", df2[comp_col].mean())
    return df2


def apply_onboarding_bump(
    df: pd.DataFrame,
    comp_col: str,
    ob_cfg: Mapping[str, Any],
    baseline_hire_salaries: pd.Series,
    rng: np.random.Generator,
    log: logging.Logger,
) -> pd.DataFrame:
    """
    Apply onboarding bump for new hires based on configuration.

    Args:
        df: DataFrame of new hires.
        comp_col: Name of the compensation column.
        ob_cfg: Configuration dictionary for onboarding bump (keys: 'enabled', 'method', 'rate').
        baseline_hire_salaries: Series of salaries from previous year's hires for sampling.
        rng: NumPy random number generator.
        log: Logger instance.

    Returns:
        DataFrame with onboarding bump applied.
    """
    df2 = df.copy()
    if not ob_cfg or not ob_cfg.get("enabled", False):
        log.debug("Onboarding bump disabled or config missing.")
        return df2
    if comp_col not in df2.columns:
        log.warning(
            f"Onboarding bump enabled, but comp column '{comp_col}' not found. Skipping."
        )
        return df2

    method = ob_cfg.get("method", "")
    rate = ob_cfg.get("rate", ob_cfg.get("flat_rate", 0.0))  # Support legacy key
    log.info(f"Applying onboarding bump: method='{method}', rate={rate:.2%}")

    if method == "flat_rate":
        df2[comp_col] = pd.to_numeric(df2[comp_col], errors="coerce").fillna(0.0) * (
            1 + rate
        )
    elif method == "sample_plus_rate":
        if not baseline_hire_salaries.empty:
            log.debug(f"Sampling {len(df2)} baseline salaries for onboarding bump.")
            # Ensure choice works with Series, might need .values if it expects array
            draws = rng.choice(baseline_hire_salaries.values, size=len(df2))
            df2[comp_col] = draws * (1 + rate)
        else:
            log.warning(
                "Onboarding bump 'sample_plus_rate' method chosen, but baseline_hire_salaries is empty! Applying flat rate to existing comp instead."
            )
            df2[comp_col] = pd.to_numeric(df2[comp_col], errors="coerce").fillna(
                0.0
            ) * (
                1 + rate
            )  # Fallback
    else:
        log.warning(f"Unknown onboarding bump method '{method}', no bump applied.")

    # Ensure non-negative compensation
    df2[comp_col] = df2[comp_col].clip(lower=0)
    log.debug(
        f"Onboarding bump applied. New avg comp for hires: {df2[comp_col].mean():.0f}"
    )
    return df2
