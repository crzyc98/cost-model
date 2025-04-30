"""Auto Enrollment rule: window setup, proactive enroll, re-enroll, and outcome application."""
import logging
from datetime import timedelta
from typing import Dict, Any

import pandas as pd
import numpy as np

from utils.columns import (
    EMP_DEFERRAL_RATE,
    IS_ELIGIBLE,
    IS_PARTICIPATING,
    ELIGIBILITY_ENTRY_DATE,
    STATUS_COL,
    AE_OPTED_OUT,
    PROACTIVE_ENROLLED,
    AUTO_ENROLLED,
    ENROLLMENT_DATE,
    AE_WINDOW_START,
    AE_WINDOW_END,
    FIRST_CONTRIBUTION_DATE,
    AE_OPT_OUT_DATE,
    AUTO_REENROLLED,
    ENROLLMENT_METHOD,
    BECAME_ELIGIBLE_DURING_YEAR,
    WINDOW_CLOSED_DURING_YEAR,
)
from utils.constants import ACTIVE_STATUSES
from utils.rules.validators import AutoEnrollmentRule

logger = logging.getLogger(__name__)

def apply(
    df: pd.DataFrame,
    ae_rules: AutoEnrollmentRule,
    simulation_year_start_date: pd.Timestamp,
    simulation_year_end_date: pd.Timestamp
) -> pd.DataFrame:
    """Apply auto-enrollment rules to the DataFrame using validated AE rules."""
    logger.info(f"Applying Auto Enrollment for {simulation_year_end_date.year}")
    if not ae_rules.enabled:
        logger.info("Auto Enrollment disabled. Skipping.")
        return df
    ae_default_rate = ae_rules.default_rate
    ae_outcome_dist = ae_rules.outcome_distribution or {}

    # Check required columns
    required = [
        IS_ELIGIBLE,
        IS_PARTICIPATING,
        EMP_DEFERRAL_RATE,
        AE_OPTED_OUT,
        ELIGIBILITY_ENTRY_DATE,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        logger.warning(f"Required columns missing for Auto Enrollment: {missing}. Skipping.")
        return df

    # Initialize tracking columns
    for col in (ENROLLMENT_DATE, FIRST_CONTRIBUTION_DATE, AE_OPT_OUT_DATE):
        if col not in df.columns:
            df[col] = pd.NaT

    # Setup AE window (default zero days if not configured)
    window_days = ae_rules.window_days or 0
    df[AE_WINDOW_START] = df[ELIGIBILITY_ENTRY_DATE]
    df[AE_WINDOW_END] = df[ELIGIBILITY_ENTRY_DATE] + timedelta(days=window_days)

    # Initialize flags
    df[PROACTIVE_ENROLLED] = False
    df[AUTO_ENROLLED] = False
    df[BECAME_ELIGIBLE_DURING_YEAR] = False
    df[WINDOW_CLOSED_DURING_YEAR] = False

    # Proactive enrollment at eligibility entry
    proactive_p = ae_rules.proactive_enrollment_probability
    logger.debug(f"[Proactive AE] probability={proactive_p:.2%}")
    newly_eligible = (
        (df[ELIGIBILITY_ENTRY_DATE] >= simulation_year_start_date) &
        (df[ELIGIBILITY_ENTRY_DATE] <= simulation_year_end_date)
    )
    logger.debug(f"[Proactive AE] newly eligible count={newly_eligible.sum()}")
    active = df[STATUS_COL] == ACTIVE_STATUSES[0]
    not_part = ~df[IS_PARTICIPATING]
    not_opted = ~df[AE_OPTED_OUT]
    candidates = newly_eligible & active & not_part & not_opted
    logger.debug(f"[Proactive AE] candidate count={candidates.sum()}")
    idxs = df.index[candidates]
    if len(idxs) > 0:
        draws = np.random.rand(len(idxs))
        selected = idxs[draws < proactive_p]
        # Assign deferral rate
        distribution = ae_rules.proactive_rate_range
        if distribution is not None:
            min_r, max_r = distribution
            rates = np.random.uniform(min_r, max_r, size=len(selected))
            df.loc[selected, EMP_DEFERRAL_RATE] = rates
        else:
            df.loc[selected, EMP_DEFERRAL_RATE] = ae_default_rate
        df.loc[selected, IS_PARTICIPATING] = True
        df.loc[selected, PROACTIVE_ENROLLED] = True
        df.loc[selected, ENROLLMENT_DATE] = df.loc[selected, ELIGIBILITY_ENTRY_DATE]
        df.loc[selected, FIRST_CONTRIBUTION_DATE] = df.loc[selected, ELIGIBILITY_ENTRY_DATE]
        logger.info(f"{len(selected)} proactively enrolled at eligibility.")

    # Re-enroll existing participants below default rate
    if ae_rules.re_enroll_existing:
        if AUTO_REENROLLED not in df.columns:
            df[AUTO_REENROLLED] = False
        mask = (
            (df[STATUS_COL] == ACTIVE_STATUSES[0]) &
            (df[IS_ELIGIBLE]) &
            (df[IS_PARTICIPATING]) &
            (df[EMP_DEFERRAL_RATE] > 0) &
            (df[EMP_DEFERRAL_RATE] < ae_default_rate)
        )
        if mask.any():
            df.loc[mask, EMP_DEFERRAL_RATE] = ae_default_rate
            df.loc[mask, IS_PARTICIPATING] = True
            df.loc[mask, ENROLLMENT_METHOD] = 'AE'
            df.loc[mask, AUTO_REENROLLED] = True
            logger.info(f"Re-enrolled {mask.sum()} existing participants at default rate {ae_default_rate:.2%}")

    # AE window closure
    within_window = (
        (df[AE_WINDOW_END] >= simulation_year_start_date) &
        (df[AE_WINDOW_END] <= simulation_year_end_date)
    )
    df.loc[within_window, WINDOW_CLOSED_DURING_YEAR] = True

    # Target for AE outcomes
    ae_target = (
        df[IS_ELIGIBLE] &
        (~df[IS_PARTICIPATING]) &
        (~df[AE_OPTED_OUT]) &
        within_window
    )
    num_targeted = ae_target.sum()
    if num_targeted == 0:
        logger.info("No employees targeted for Auto Enrollment this year.")
        return df
    logger.info(f"Targeting {num_targeted} employees for AE.")

    # Outcome distribution: ae_rules.outcome_distribution
    outcomes = []
    probs = []
    for key, prob in ae_outcome_dist.items():
        if not key.startswith('prob_'):
            continue
        outcome = key[len('prob_'):]
        outcomes.append(outcome)
        probs.append(prob)
    total_p = sum(probs)
    # Normalize probabilities if not 1.0
    if total_p > 0 and not np.isclose(total_p, 1.0):
        logger.warning(f"AE outcome probabilities sum to {total_p:.2%}; normalizing.")
        probs = [p / total_p for p in probs]
    # Default to stay_default if no valid probs
    if not outcomes:
        outcomes = ['stay_default']
        probs = [1.0]
    target_indices = df.index[ae_target]
    assigned = np.random.choice(outcomes, size=num_targeted, p=probs)
    for idx, outcome in zip(target_indices, assigned):
        if outcome == 'stay_default':
            # assign default rate or random within proactive range
            dist = ae_rules.proactive_rate_range
            if dist:
                min_r, max_r = dist
                df.loc[idx, EMP_DEFERRAL_RATE] = np.random.uniform(min_r, max_r)
            else:
                df.loc[idx, EMP_DEFERRAL_RATE] = ae_default_rate
            df.loc[idx, IS_PARTICIPATING] = True
            df.loc[idx, AUTO_ENROLLED] = True
            df.loc[idx, ENROLLMENT_DATE] = df.loc[idx, AE_WINDOW_END]
            df.loc[idx, FIRST_CONTRIBUTION_DATE] = df.loc[idx, AE_WINDOW_END]
        elif outcome == 'opt_out':
            # opted out
            df.loc[idx, AE_OPTED_OUT] = True
            df.loc[idx, AE_OPT_OUT_DATE] = df.loc[idx, AE_WINDOW_END]
            df.loc[idx, IS_PARTICIPATING] = False
        elif outcome == 'opt_down':
            # TODO: implement decrease in deferral rate
            logger.warning('AE outcome opt_down not implemented; defaulting to stay_default behavior')
            df.loc[idx, EMP_DEFERRAL_RATE] = ae_default_rate
        elif outcome == 'increase_to_match':
            # TODO: implement increase to match employer
            logger.warning('AE outcome increase_to_match not implemented; using default_rate')
            df.loc[idx, EMP_DEFERRAL_RATE] = ae_default_rate
        elif outcome == 'increase_high':
            # TODO: implement high deferral increase
            logger.warning('AE outcome increase_high not implemented; using default_rate')
            df.loc[idx, EMP_DEFERRAL_RATE] = ae_default_rate
        else:
            logger.warning(f'Unknown AE outcome: {outcome}')
            df.loc[idx, EMP_DEFERRAL_RATE] = ae_default_rate
            df.loc[idx, IS_PARTICIPATING] = True
    # Log summary
    enrolled_count = df.loc[target_indices, IS_PARTICIPATING].sum()
    opted_out_count = df.loc[target_indices, AE_OPTED_OUT].sum()
    logger.info(f"AE Applied: {enrolled_count} enrolled, {opted_out_count - enrolled_count} opted out.")
    return df
