"""
utils/rules/contributions.py

Calculates employee and employer contributions for the simulation year.
"""
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

from utils.rules.validators import ContributionsRule, MatchRule, NonElectiveRule
from utils.columns import (
    EMP_GROSS_COMP,
    EMP_DEFERRAL_RATE,
    EMP_CONTR,
    EMPLOYER_MATCH,
    EMPLOYER_CORE,
    EMP_CAPPED_COMP,
    EMP_PLAN_YEAR_COMP,
    to_nullable_bool,
)
from utils.date_utils import calculate_age
from utils.constants import ACTIVE_STATUSES

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    # 0) Find and resolve any duplicate column names up front
    dupes = df.columns[df.columns.duplicated()].unique().tolist()
    if dupes:
        logger.warning("Duplicate columns detected: %r", dupes)
        df = df.loc[:, ~df.columns.duplicated(keep='last')]
        logger.info("Columns after deduplication: %r", df.columns.tolist())

    # 1) Check for duplicate indices
    dupes = df.index.duplicated()
    if dupes.any():
        logger.error("Duplicate indices found in input DataFrame: %d duplicates", dupes.sum())
        logger.debug("Duplicate index values: %s", df.index[dupes].unique())

    logger.info(f"Calculating contributions for {simulation_year}")

    # --- Extract IRS limits for the year ---
    year_limits    = irs_limits.get(simulation_year, {})
    comp_limit     = year_limits.get('compensation_limit',      345000)
    def_limit      = year_limits.get('deferral_limit',          23000)
    catch_limit    = year_limits.get('catchup_limit',           7500)
    catch_age      = year_limits.get('catchup_eligibility_age', 50)
    overall_limit  = year_limits.get('overall_limit')

    # --- 1) Initialize output columns with defaults ---
    defaults: Dict[str, Any] = {
        EMP_PLAN_YEAR_COMP:     0.0,
        EMP_CAPPED_COMP:        0.0,
        EMP_CONTR:              0.0,
        EMPLOYER_MATCH:         0.0,
        EMPLOYER_CORE:          0.0,
        'total_contributions':  0.0,
        'is_catch_up_eligible': False,
        'effective_deferral_limit': 0.0,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            dtype = 'boolean' if isinstance(default, bool) else float
            df[col] = pd.Series(default, index=df.index, dtype=dtype)
        # ensure boolean defaults are proper nullable bool
        if isinstance(default, bool):
            df[col] = to_nullable_bool(df[col])

    # --- 2) Defensive numeric coercion for key contribution columns ---
    for col in [EMP_DEFERRAL_RATE, EMP_CONTR, EMPLOYER_MATCH, EMPLOYER_CORE,
                EMP_PLAN_YEAR_COMP, EMP_CAPPED_COMP, 'total_contributions', 'effective_deferral_limit']:
        val = df.get(col, None)
        if isinstance(val, pd.Series):
            df[col] = pd.to_numeric(val, errors='coerce').fillna(0.0)
        elif isinstance(val, (list, tuple, np.ndarray)):
            df[col] = pd.Series(val, index=df.index).pipe(
                pd.to_numeric, errors='coerce'
            ).fillna(0.0)
        else:
            df[col] = 0.0

    # --- 3) Proration of compensation for terminations ---
    total_days = (pd.to_datetime(year_end) - pd.to_datetime(year_start)).days + 1
    df['days_worked'] = total_days
    if 'termination_date' in df.columns:
        df['termination_date'] = pd.to_datetime(df['termination_date'], errors='coerce')
        mask_term = (
            df['termination_date'].notna() &
            df['termination_date'].between(year_start, year_end)
        )
        df.loc[mask_term, 'days_worked'] = (
            (df.loc[mask_term, 'termination_date'] - year_start).dt.days + 1
        ).clip(lower=1, upper=total_days)

    df['proration'] = df['days_worked'] / total_days
    df[EMP_PLAN_YEAR_COMP] = df[EMP_GROSS_COMP] * df['proration']
    df[EMP_CAPPED_COMP]    = np.minimum(
        df[EMP_PLAN_YEAR_COMP], comp_limit * df['proration']
    )

    # --- 4) Catch-up eligibility & effective limits ---
    df['current_age'] = calculate_age(df.get('birth_date'), year_end)
    active_mask       = df['status'].isin(ACTIVE_STATUSES)
    catch_mask        = active_mask & (df['current_age'] >= catch_age)

    df.loc[catch_mask, 'is_catch_up_eligible']      = True
    df['effective_deferral_limit'] = def_limit
    df.loc[catch_mask, 'effective_deferral_limit'] += catch_limit

    # --- 5) Employee pre-tax contributions ---
    potential = df[EMP_CAPPED_COMP] * df[EMP_DEFERRAL_RATE]
    df[EMP_CONTR] = np.minimum(potential, df['effective_deferral_limit'])

    # --- 6) Employer non-elective contributions (NEC) ---
    df[EMPLOYER_CORE] = df[EMP_CAPPED_COMP] * nec_rules.rate

    # --- 7) Employer match contributions ---
    df[EMPLOYER_MATCH] = 0.0
    tiers             = match_rules.tiers
    dollar_cap        = match_rules.dollar_cap
    if tiers:
        caps  = np.array([t.cap_deferral_pct for t in tiers])
        rates = np.array([t.match_rate for t in tiers])
        prev  = np.concatenate(([0.0], caps[:-1]))
        dr    = df[EMP_DEFERRAL_RATE].to_numpy()[:, None]
        cc    = df[EMP_CAPPED_COMP].to_numpy()[:, None]
        alloc = np.clip(np.minimum(dr, caps) - prev, 0.0, None)
        match_amt = (alloc * cc * rates).sum(axis=1)
        if dollar_cap is not None:
            match_amt = np.minimum(match_amt, dollar_cap)
        df.loc[active_mask, EMPLOYER_MATCH] = match_amt[active_mask]

    # --- 8) Total contributions & overall limit check ---
    df['total_contributions'] = (
        df[EMP_CONTR] + df[EMPLOYER_MATCH] + df[EMPLOYER_CORE]
    )
    logger.debug("Null values - EMP_CONTR: %d, EMPLOYER_MATCH: %d, EMPLOYER_CORE: %d",
                 df[EMP_CONTR].isna().sum(),
                 df[EMPLOYER_MATCH].isna().sum(),
                 df[EMPLOYER_CORE].isna().sum())
    if overall_limit is not None:
        over = df['total_contributions'] > overall_limit
        if over.any():
            logger.warning(
                "%d employees exceed overall limit $%s, scaling back ER credits",
                over.sum(), overall_limit
            )
            excess = df.loc[over, 'total_contributions'] - overall_limit
            er_tot = df.loc[over, EMPLOYER_MATCH] + df.loc[over, EMPLOYER_CORE]
            frac   = np.where(er_tot > 0, 1 - excess / er_tot, 0)
            df.loc[over, EMPLOYER_MATCH] *= frac
            df.loc[over, EMPLOYER_CORE]   *= frac
            df.loc[over, 'total_contributions'] = overall_limit

    # --- Cleanup intermediate columns ---
    df.drop(columns=['days_worked', 'proration', 'current_age'], inplace=True)
    return df