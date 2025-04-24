"""
Auto Increase rule: apply deferral rate increases according to plan rules.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def apply(df, plan_rules, simulation_year):
    """Apply auto-increase rules to the DataFrame."""
    ai_config = plan_rules.get('auto_increase', {})
    ai_enabled = ai_config.get('enabled', False)
    if not ai_enabled:
        logger.info("Auto Increase disabled. Skipping.")
        return df

    ai_increase_rate = ai_config.get('increase_rate', 0.01)
    ai_max_deferral_rate = ai_config.get('cap_rate', ai_config.get('max_deferral_rate', 0.10))

    # Initialize flags
    if 'ai_opted_out' not in df.columns:
        df['ai_opted_out'] = False
    if 'ai_enrolled' not in df.columns:
        df['ai_enrolled'] = False
    else:
        df['ai_enrolled'] = df['ai_enrolled'].fillna(False)

    # Apply auto-increase to all employees enrolled in AI until they hit the cap
    mask = (
        df['ai_enrolled'] &
        (~df['ai_opted_out']) &
        (df['deferral_rate'] < ai_max_deferral_rate)
    )
    logger.debug(f"AI candidates: {mask.sum()} / {len(df)}")

    # Apply increase
    if mask.any():
        new_rates = np.minimum(df.loc[mask, 'deferral_rate'] + ai_increase_rate, ai_max_deferral_rate)
        df.loc[mask, 'deferral_rate'] = new_rates
        df.loc[mask, 'ai_enrolled'] = True
        logger.info(f"Auto Increase applied to {mask.sum()} employees: +{ai_increase_rate*100:.1f}% (cap {ai_max_deferral_rate*100:.1f}%)")
    else:
        logger.info("No eligible participants for Auto Increase.")
    return df
