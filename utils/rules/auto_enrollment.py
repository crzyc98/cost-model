"""
Auto Enrollment rule: window setup, proactive enroll, re-enroll, and outcome application.
"""
import pandas as pd
import numpy as np
import logging
from datetime import timedelta
from typing import Dict, Any

logger = logging.getLogger(__name__)

def apply(
    df: pd.DataFrame,
    plan_rules: Dict[str, Any],
    simulation_year_start_date: pd.Timestamp,
    simulation_year_end_date: pd.Timestamp
) -> pd.DataFrame:
    """Apply auto-enrollment rules to the DataFrame."""
    logger.info(f"Applying Auto Enrollment for {simulation_year_end_date.year}")
    ae_rules = plan_rules.get('auto_enrollment', {})
    ae_enabled = ae_rules.get('enabled', False)
    ae_default_rate = ae_rules.get('default_rate', 0.0)
    ae_outcome_dist = ae_rules.get('ae_outcome_distribution', {})

    if not ae_enabled:
        logger.info("Auto Enrollment disabled. Skipping.")
        return df

    # Check required columns
    required = ['is_eligible', 'is_participating', 'deferral_rate', 'ae_opted_out', 'eligibility_entry_date']
    missing = [col for col in required if col not in df.columns]
    if missing:
        logger.warning(f"Required columns missing for Auto Enrollment: {missing}. Skipping.")
        return df

    # Initialize tracking columns
    for col in ('enrollment_date', 'first_contribution_date', 'ae_opt_out_date'):
        if col not in df.columns:
            df[col] = pd.NaT

    # Setup AE window (default zero days if not configured)
    window_days = ae_rules.get('window_days', 0)
    df['ae_window_start'] = df['eligibility_entry_date']
    df['ae_window_end'] = df['eligibility_entry_date'] + timedelta(days=window_days)

    # Initialize flags
    df['proactive_enrolled'] = False
    df['auto_enrolled'] = False
    df['became_eligible_during_year'] = False
    df['window_closed_during_year'] = False

    # Proactive enrollment at eligibility entry
    proactive_p = ae_rules.get('proactive_enrollment_probability', 0.0)
    logger.debug(f"[Proactive AE] probability={proactive_p:.2%}")
    newly_eligible = (
        (df['eligibility_entry_date'] >= simulation_year_start_date) &
        (df['eligibility_entry_date'] <= simulation_year_end_date)
    )
    logger.debug(f"[Proactive AE] newly eligible count={newly_eligible.sum()}")
    active = df['status'] == 'Active'
    not_part = ~df['is_participating']
    not_opted = ~df['ae_opted_out']
    candidates = newly_eligible & active & not_part & not_opted
    logger.debug(f"[Proactive AE] candidate count={candidates.sum()}")
    idxs = df.index[candidates]
    if len(idxs) > 0:
        draws = np.random.rand(len(idxs))
        selected = idxs[draws < proactive_p]
        # Assign deferral rate
        distribution = ae_rules.get('proactive_rate_range', None)
        if distribution is not None:
            min_r, max_r = distribution
            rates = np.random.uniform(min_r, max_r, size=len(selected))
            df.loc[selected, 'deferral_rate'] = rates
        else:
            df.loc[selected, 'deferral_rate'] = ae_default_rate
        df.loc[selected, 'is_participating'] = True
        df.loc[selected, 'proactive_enrolled'] = True
        df.loc[selected, 'enrollment_date'] = df.loc[selected, 'eligibility_entry_date']
        df.loc[selected, 'first_contribution_date'] = df.loc[selected, 'eligibility_entry_date']
        logger.info(f"{len(selected)} proactively enrolled at eligibility.")

    # Re-enroll existing participants below default rate
    re_enroll_existing = ae_rules.get('re_enroll_existing', False)
    if re_enroll_existing:
        if 'auto_reenrolled' not in df.columns:
            df['auto_reenrolled'] = False
        mask = (
            (df['status'] == 'Active') &
            (df['is_eligible']) &
            (df['is_participating']) &
            (df['deferral_rate'] > 0) &
            (df['deferral_rate'] < ae_default_rate)
        )
        if mask.any():
            df.loc[mask, 'deferral_rate'] = ae_default_rate
            df.loc[mask, 'is_participating'] = True
            df.loc[mask, 'enrollment_method'] = 'AE'
            df.loc[mask, 'auto_reenrolled'] = True
            logger.info(f"Re-enrolled {mask.sum()} existing participants at default rate {ae_default_rate:.2%}")

    # AE window closure
    within_window = (
        (df['ae_window_end'] >= simulation_year_start_date) &
        (df['ae_window_end'] <= simulation_year_end_date)
    )
    df.loc[within_window, 'window_closed_during_year'] = True

    # Target for AE outcomes
    ae_target = (
        df['is_eligible'] &
        (~df['is_participating']) &
        (~df['ae_opted_out']) &
        within_window
    )
    num_targeted = ae_target.sum()
    if num_targeted == 0:
        logger.info("No employees targeted for Auto Enrollment this year.")
        return df
    logger.info(f"Targeting {num_targeted} employees for AE.")

    # Outcome distribution
    stay_p = ae_outcome_dist.get('stay_default', 0.0)
    opt_out_p = ae_outcome_dist.get('opt_out', 0.0)
    total_p = stay_p + opt_out_p
    # Normalize if sum not 1.0
    if total_p > 0 and not np.isclose(total_p, 1.0):
        logger.warning(f"AE outcome probabilities sum to {total_p:.2%}; normalizing.")
        stay_p /= total_p
        opt_out_p /= total_p
    # Default invalid or empty distribution to stay_default
    if total_p <= 0:
        logger.info("AE outcome distribution missing or zero; defaulting all to stay_default.")
        stay_p, opt_out_p = 1.0, 0.0
    
    outcomes = ['stay_default', 'opt_out']
    probs = [stay_p, opt_out_p]
    target_indices = df.index[ae_target]
    assigned = np.random.choice(outcomes, size=num_targeted, p=probs)
    for idx, outcome in zip(target_indices, assigned):
        if outcome == 'stay_default':
            distribution = ae_rules.get('proactive_rate_range', None)
            if distribution is not None:
                min_r, max_r = distribution
                rate = np.random.uniform(min_r, max_r)
                df.loc[idx, 'deferral_rate'] = rate
            else:
                df.loc[idx, 'deferral_rate'] = ae_default_rate
            df.loc[idx, 'is_participating'] = True
            df.loc[idx, 'auto_enrolled'] = True
            df.loc[idx, 'enrollment_date'] = df.loc[idx, 'ae_window_end']
            df.loc[idx, 'first_contribution_date'] = df.loc[idx, 'ae_window_end']
        else:
            df.loc[idx, 'ae_opted_out'] = True
            df.loc[idx, 'ae_opt_out_date'] = df.loc[idx, 'ae_window_end']
            df.loc[idx, 'is_participating'] = False

    enrolled_count = df.loc[target_indices, 'is_participating'].sum()
    opted_out_count = df.loc[target_indices, 'ae_opted_out'].sum()
    logger.info(f"AE Applied: {enrolled_count} enrolled, {opted_out_count - enrolled_count} opted out.")

    return df
