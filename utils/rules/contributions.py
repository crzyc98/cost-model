# utils/rules/contributions.py
"""
Calculates employee and employer contributions for the simulation year.
"""
import logging
from typing import Dict, Any
from utils.rules.validators import ContributionsRule, MatchRule, NonElectiveRule

import numpy as np
import pandas as pd

from utils.columns import (
    EMP_GROSS_COMP,
    DEFERRAL_RATE,
    EMP_PRE_TAX_CONTR,
    EMPLOYER_MATCH,
    EMPLOYER_NEC,
    PRE_TAX_CONTR,
    CAPPED_COMP,
    PLAN_YEAR_COMP,
    to_nullable_bool,
)
from utils.date_utils import calculate_age
from utils.constants import ACTIVE_STATUSES

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = True

def apply(
    df: pd.DataFrame,
    contrib_rules: ContributionsRule,
    match_rules: MatchRule,
    nec_rules: NonElectiveRule,
    irs_limits: Dict[int, Dict[str, float]],
    simulation_year: int,
    year_start: pd.Timestamp,
    year_end: pd.Timestamp
) -> pd.DataFrame:
    """
    Vectorized contributions calculation:
      1. Prorate compensation for terminations
      2. Calculate plan-year & capped compensation
      3. Determine catch-up eligibility and limits
      4. Compute employee pre-tax, NEC, match, and total contributions
    """
    logger.info(f"Calculating contributions for {simulation_year}")
    year_limits = irs_limits.get(simulation_year, {})
    comp_limit    = year_limits.get('compensation_limit', 345000)
    def_limit     = year_limits.get('deferral_limit',      23000)
    catch_limit   = year_limits.get('catchup_limit',       7500)
    catch_age     = year_limits.get('catchup_eligibility_age', 50)
    overall_limit = year_limits.get('overall_limit',       None)  # if you have it

    # --- Initialize all output cols with defaults ---
    defaults: Dict[str, Any] = {
        PLAN_YEAR_COMP:          0.0,
        CAPPED_COMP:             0.0,
        PRE_TAX_CONTR:           0.0,
        EMPLOYER_MATCH:          0.0,
        EMPLOYER_NEC:            0.0,
        'total_contributions':   0.0,
        'is_catch_up_eligible':  False,
        'effective_deferral_limit': 0.0,
    }
    for col, default in defaults.items():
        if col not in df:
            df[col] = pd.Series(
                default,
                index=df.index,
                dtype='boolean' if isinstance(default, bool) else float
            )
        # coerce numeric columns
        if not isinstance(default, bool):
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        else:
            df[col] = to_nullable_bool(df[col])

    # ensure deferral_rate is numeric
    df[DEFERRAL_RATE] = pd.to_numeric(df.get(DEFERRAL_RATE, 0.0), errors='coerce').fillna(0.0)
    df[EMP_PRE_TAX_CONTR] = pd.to_numeric(df.get(EMP_PRE_TAX_CONTR, 0.0), errors='coerce').fillna(0.0)

    # --- Proration of compensation ---
    year_days = (pd.to_datetime(year_end) - pd.to_datetime(year_start)).days + 1
    df['days_worked'] = year_days  # default full-year
    if 'termination_date' in df:
        df['termination_date'] = pd.to_datetime(df['termination_date'], errors='coerce')
        mask_term = (
            df['termination_date'].notna()
            & (df['termination_date'] >= year_start)
            & (df['termination_date'] <= year_end)
        )
        df.loc[mask_term, 'days_worked'] = (
            (df.loc[mask_term, 'termination_date'] - year_start).dt.days + 1
        ).clip(lower=1, upper=year_days)

    df['proration'] = df['days_worked'] / year_days
    # plan-year and capped compensation
    df[PLAN_YEAR_COMP] = df[EMP_GROSS_COMP] * df['proration']
    df[CAPPED_COMP]    = np.minimum(df[PLAN_YEAR_COMP], comp_limit * df['proration'])

    # --- Catch-up eligibility & effective limits ---
    active_mask = df['status'].isin(ACTIVE_STATUSES)
    df['current_age'] = calculate_age(df.get('birth_date'), year_end)
    catch_mask = active_mask & (df['current_age'] >= catch_age)
    df.loc[catch_mask, 'is_catch_up_eligible'] = True

    df['effective_deferral_limit'] = def_limit
    df.loc[catch_mask, 'effective_deferral_limit'] += catch_limit

    # --- Employee Pre-Tax Contributions ---
    potential = df[CAPPED_COMP] * df[DEFERRAL_RATE]
    df[PRE_TAX_CONTR] = np.minimum(potential, df['effective_deferral_limit'])

    # --- Employer Non-Elective Contributions (NEC) ---
    df[EMPLOYER_NEC] = df[CAPPED_COMP] * nec_rules.rate

    # --- Employer Match Contributions ---
    df[EMPLOYER_MATCH] = 0.0
    tiers = match_rules.tiers
    dollar_cap = match_rules.dollar_cap
    if tiers:
        # vectorized multi-tier calculation
        caps  = np.array([t['cap_deferral_pct'] for t in tiers])
        rates = np.array([t['match_rate']      for t in tiers])
        prev  = np.concatenate(([0.0], caps[:-1]))
        dr    = df[DEFERRAL_RATE].to_numpy()[:, None]
        cc    = df[CAPPED_COMP].to_numpy()[:, None]
        alloc = np.clip(np.minimum(dr, caps) - prev, 0.0, None)
        match_amt = (alloc * cc * rates).sum(axis=1)
        if dollar_cap is not None:
            match_amt = np.minimum(match_amt, dollar_cap)
        df.loc[active_mask, EMPLOYER_MATCH] = match_amt[active_mask]

    # --- Total contributions and overall-limit check ---
    df['total_contributions'] = (
        df[PRE_TAX_CONTR]
        + df[EMPLOYER_MATCH]
        + df[EMPLOYER_NEC]
    )
    if overall_limit is not None:
        over = df['total_contributions'] > overall_limit
        if over.any():
            logger.warning(
                "%d employees exceed overall limit $%s → scaling back ER credits",
                over.sum(), overall_limit
            )
            excess = df.loc[over, 'total_contributions'] - overall_limit
            er_tot = df.loc[over, EMPLOYER_MATCH] + df.loc[over, EMPLOYER_NEC]
            frac = np.where(er_tot>0, 1 - excess/er_tot, 0)
            df.loc[over, EMPLOYER_MATCH] *= frac
            df.loc[over, EMPLOYER_NEC]   *= frac
            df.loc[over, 'total_contributions'] = overall_limit

    # cleanup
    df.drop(columns=['days_worked', 'proration', 'current_age'], inplace=True)
    return df