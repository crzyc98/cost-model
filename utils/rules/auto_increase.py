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
    if not ai_config.get('enabled', False):
        logger.info("Auto Increase disabled. Skipping.")
        return df

    ai_increase_rate     = ai_config.get('increase_rate', 0.01)
    ai_max_deferral_rate = ai_config.get('cap_rate', ai_config.get('max_deferral_rate', 0.10))

    # --- Initialize flags ---
    if 'ai_opted_out' not in df.columns:
        df['ai_opted_out'] = False

    if 'ai_enrolled' not in df.columns:
        df['ai_enrolled'] = pd.Series(False, index=df.index, dtype='boolean')
    else:
        df['ai_enrolled'] = df['ai_enrolled'].astype('boolean').fillna(False)

    # ensure participating flag exists for seeding logic
    if 'is_participating' not in df.columns:
        df['is_participating'] = False

    # --- Seed ai_enrolled per scenario flags ---
    # 1) only bump new hires (hired this year)
    if ai_config.get('apply_to_new_hires_only', False):
        df.loc[
            df['employee_hire_date'].dt.year == simulation_year,
            'ai_enrolled'
        ] = True

    # 2) re-enroll existing participants under the cap
    elif ai_config.get('re_enroll_existing_below_cap', False):
        df.loc[
            df['is_participating'] & (df['employee_deferral_rate'] < ai_max_deferral_rate),
            'ai_enrolled'
        ] = True

    # 3) default: everyone participating
    else:
        df.loc[df['is_participating'], 'ai_enrolled'] = True

    logger.debug("AI-enrolled after seeding: %d/%d", df['ai_enrolled'].sum(), len(df))

    # --- Build bump mask and apply increase ---
    mask = (
        df['ai_enrolled']
      & ~df['ai_opted_out']
      & (df['employee_deferral_rate'] < ai_max_deferral_rate)
    )
    logger.debug("AI candidates: %d / %d", mask.sum(), len(df))

    if mask.any():
        df.loc[mask, 'employee_deferral_rate'] = (
            df.loc[mask, 'employee_deferral_rate'] + ai_increase_rate
        ).clip(upper=ai_max_deferral_rate)
        # Ensure deferral rate is not negative
        df.loc[mask, 'employee_deferral_rate'] = df.loc[mask, 'employee_deferral_rate'].clip(lower=0.0)
        # keep ai_enrolled True
        df.loc[mask, 'ai_enrolled'] = True
        logger.info(
            "Auto Increase applied to %d employees: +%.1f%% (cap %.1f%%)",
            mask.sum(),
            ai_increase_rate * 100,
            ai_max_deferral_rate * 100
        )
    else:
        logger.info("No eligible participants for Auto Increase.")

    return df