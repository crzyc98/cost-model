"""Auto Enrollment rule: window setup, proactive enroll, re-enroll, and outcome application."""
import logging
import numpy as np
import pandas as pd

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
    df[AE_WINDOW_END] = df[ELIGIBILITY_ENTRY_DATE] + pd.Timedelta(days=window_days)

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
    active = df[STATUS_COL].isin(ACTIVE_STATUSES)
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
            (df[STATUS_COL].isin(ACTIVE_STATUSES)) &
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

    # Inject AE rule rates into DataFrame
    df = df.copy()
    df['rate_for_max_match'] = ae_rules.increase_to_match_rate
    df['opt_down_rate'] = ae_rules.opt_down_target_rate
    df['increase_high_rate'] = ae_rules.increase_high_rate

    # Outcome distribution: ae_rules.outcome_distribution
    od = ae_rules.outcome_distribution
    
    # Cumulative thresholds
    thresholds = np.cumsum([
        od.prob_opt_out,
        od.prob_stay_default,
        od.prob_opt_down,
        od.prob_increase_to_match,
        od.prob_increase_high
    ])

    # Random draws
    draws = np.random.rand(num_targeted)

    # Apply outcomes
    counts = {'opt_out':0, 'stay_default':0, 'opt_down':0, 'to_match':0, 'increase_high':0}

    target_indices = df.index[ae_target]
    for i, draw in zip(target_indices, draws):
        if draw < thresholds[0]:
            # opt-out
            df.at[i, AE_OPTED_OUT] = True
            counts['opt_out'] += 1
        elif draw < thresholds[1]:
            # stay at default
            df.at[i, EMP_DEFERRAL_RATE] = ae_rules.default_rate
            df.at[i, IS_PARTICIPATING] = True
            df.at[i, ENROLLMENT_METHOD] = 'AE'
            counts['stay_default'] += 1
        elif draw < thresholds[2]:
            # opt-down
            df.at[i, EMP_DEFERRAL_RATE] = df.at[i, 'opt_down_rate']
            df.at[i, IS_PARTICIPATING] = True
            df.at[i, ENROLLMENT_METHOD] = 'AE'
            counts['opt_down'] += 1
        elif draw < thresholds[3]:
            # increase to match
            df.at[i, EMP_DEFERRAL_RATE] = df.at[i, 'rate_for_max_match']
            df.at[i, IS_PARTICIPATING] = True
            df.at[i, ENROLLMENT_METHOD] = 'AE'
            counts['to_match'] += 1
        elif draw < thresholds[4]:
            # increase to high target
            df.at[i, EMP_DEFERRAL_RATE] = df.at[i, 'increase_high_rate']
            df.at[i, IS_PARTICIPATING] = True
            df.at[i, ENROLLMENT_METHOD] = 'AE'
            counts['increase_high'] += 1
        else:
            # fallback to default
            df.at[i, EMP_DEFERRAL_RATE] = ae_rules.default_rate
            df.at[i, IS_PARTICIPATING] = True
            df.at[i, ENROLLMENT_METHOD] = 'AE'
            counts['stay_default'] += 1

    logger.info(
        "AE Applied: %d default, %d opt-down, %d to-match, %d high, %d opt-out",
        counts['stay_default'],
        counts['opt_down'],
        counts['to_match'],
        counts['increase_high'],
        counts['opt_out']
    )

    return df
