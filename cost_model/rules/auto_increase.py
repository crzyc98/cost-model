# cost_model/rules/auto_increase.py
"""
Auto Increase rule: apply deferral rate increases according to plan rules.
"""
import logging
from typing import Dict, Any

import pandas as pd

from cost_model.utils.columns import (
    EMP_DEFERRAL_RATE,
    EMP_HIRE_DATE,
    IS_PARTICIPATING,
    AI_OPTED_OUT,
    AI_ENROLLED,
    to_nullable_bool,
)
from cost_model.rules.validators import AutoIncreaseRule

logger = logging.getLogger(__name__)


def apply(
    df: pd.DataFrame, ai_rules: Dict[str, Any], simulation_year: int
) -> pd.DataFrame:
    """
    Apply auto-increase rules to the DataFrame.

    Parameters:
      - df: employee-level snapshot for the year
      - ai_rules: AutoIncreaseRule instance or dict
      - simulation_year: calendar year (int) of this snapshot

    Returns:
      - df with AI flags and bumped EMP_DEFERRAL_RATE
    """
    # coerce plain dicts into the validator object
    if isinstance(ai_rules, dict):
        # Fill in defaults if keys are missing
        ai = {"enabled": False, "increase_rate": 0.0, "cap_rate": 0.0, **ai_rules}
        ai_rules = AutoIncreaseRule(**ai)

    if not ai_rules.enabled:
        logger.info("Auto Increase disabled. Skipping.")
        return df

    increase_rate = ai_rules.increase_rate
    cap_rate = ai_rules.cap_rate

    # --- Initialize flags as nullable booleans ---
    df[AI_OPTED_OUT] = to_nullable_bool(df.get(AI_OPTED_OUT, False))
    df[AI_ENROLLED] = to_nullable_bool(df.get(AI_ENROLLED, False))

    # Ensure participation column exists
    df[IS_PARTICIPATING] = to_nullable_bool(df.get(IS_PARTICIPATING, False))

    # --- Seed AI_ENROLLED per policy ---
    if ai_rules.apply_to_new_hires_only:
        # only those hired in the simulation_year
        new_hires = df[EMP_HIRE_DATE].dt.year.eq(simulation_year)
        df.loc[new_hires, AI_ENROLLED] = True

    elif ai_rules.re_enroll_existing_below_cap:
        # re-enroll existing participants whose rate is below cap
        mask = df[IS_PARTICIPATING] & df[EMP_DEFERRAL_RATE].lt(cap_rate)
        df.loc[mask, AI_ENROLLED] = True

    else:
        # default: everyone currently participating
        df.loc[df[IS_PARTICIPATING], AI_ENROLLED] = True

    logger.debug("AI seeding: %d enrolled/%d total", df[AI_ENROLLED].sum(), len(df))

    # --- Build bump mask ---
    bump_mask = df[AI_ENROLLED] & ~df[AI_OPTED_OUT] & df[EMP_DEFERRAL_RATE].lt(cap_rate)
    logger.debug("AI candidates: %d / %d", bump_mask.sum(), len(df))

    if bump_mask.any():
        before = df.loc[bump_mask, EMP_DEFERRAL_RATE]
        bumped = (before + increase_rate).clip(lower=0.0, upper=cap_rate)
        df.loc[bump_mask, EMP_DEFERRAL_RATE] = bumped

        # keep AI_ENROLLED = True for bumped
        df.loc[bump_mask, AI_ENROLLED] = True

        logger.info(
            "Auto Increase applied to %d employees: +%.2f%% up to cap %.2f%%",
            bump_mask.sum(),
            increase_rate * 100,
            cap_rate * 100,
        )
    else:
        logger.info("No eligible participants for Auto Increase.")

    return df
