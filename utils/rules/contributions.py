# utils/rules/contributions.py

"""
Calculates employee and employer contributions for the simulation year.
Extracted from utils.plan_rules.calculate_contributions.
"""

import pandas as pd
import numpy as np
import logging
from utils.date_utils import calculate_age

DEBUG_SAMPLE = False  # Set via CLI to throttle proration examples

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Ensure DEBUG logs are not filtered
logger.propagate = True        # Ensure logs propagate to root logger

def apply(df, plan_rules: dict, simulation_year, year_start_date, year_end_date) -> pd.DataFrame:
    """Calculate contributions using provided plan_rules mapping."""
    # Entry debug: show plan_rules keys and status counts
    logger.debug(
        "contributions.apply: entering with plan_rules keys=%s, status counts=\n%s",
        list(plan_rules.keys()),
        df.get('status', pd.Series()).value_counts()
    )
    logger.info(f"Calculating contributions for {simulation_year}...")
    # ensure start/end dates are timestamps for proration math
    year_start_date = pd.to_datetime(year_start_date)
    year_end_date = pd.to_datetime(year_end_date)

    # Validate plan_rules shape
    for key in ['irs_limits', 'employer_match', 'employer_nec']:
        if key not in plan_rules:
            logger.warning("plan_rules missing '%s'; related logic may be skipped", key)

    irs_limits = plan_rules.get('irs_limits', {})
    year_limits = irs_limits.get(simulation_year, {})
    logger.debug("contributions.apply: year_limits=%s", year_limits)
    logger.debug(f"IRS Limits for {simulation_year}: {year_limits}")

    statutory_comp_limit = year_limits.get('comp_limit', 345000)
    deferral_limit = year_limits.get('deferral_limit', 23000)
    catch_up_limit = year_limits.get('catch_up', 7500)
    overall_limit = year_limits.get('overall_limit', 69000)
    catch_up_age = year_limits.get(
        'catchup_eligibility_age',
        plan_rules.get('catch_up_age', 50)
    )

    required_input_cols = ['gross_compensation', 'deferral_rate', 'birth_date', 'status', 'hire_date']
    missing_cols = [col for col in required_input_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Required columns missing: {', '.join(missing_cols)}. Cannot calculate contributions.")
        return df

    output_cols = {
        'plan_year_compensation': 0.0,
        'capped_compensation': 0.0,
        'pre_tax_contributions': 0.0,
        'employer_match_contribution': 0.0,
        'employer_non_elective_contribution': 0.0,
        'total_contributions': 0.0,
        'is_catch_up_eligible': False,
        'effective_deferral_limit': 0.0
    }
    for col, default in output_cols.items():
        if col not in df.columns:
            # initialize with pandas BooleanArray for eligibility flag
            if col == 'is_catch_up_eligible':
                df[col] = pd.Series(default, index=df.index, dtype='boolean')
            else:
                df[col] = default
        if 'contribution' in col or 'compensation' in col or 'limit' in col:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        elif col == 'is_catch_up_eligible':
            # ensure extension boolean dtype and fill without warning
            df[col] = df[col].astype('boolean').fillna(False)
    # Optionally downcast other object dtypes after numeric and boolean casts
    df = df.infer_objects(copy=False)

    df['deferral_rate'] = pd.to_numeric(df['deferral_rate'], errors='coerce').fillna(0.0)

    # --- Prorate compensation based on days worked, inclusive of termination ---
    total_days = (year_end_date - year_start_date).days + 1
    logger.debug(f"Plan year total days: {total_days} ({year_start_date} to {year_end_date})")

    # compute days worked for everyone; default full year
    df['days_worked'] = total_days

    # Check if termination_date column exists and process it
    if 'termination_date' not in df.columns:
        logger.warning("termination_date column not found. Assuming no terminations.")
    else:
        # Ensure termination_date is datetime
        if not pd.api.types.is_datetime64_dtype(df['termination_date']):
            df['termination_date'] = pd.to_datetime(df['termination_date'], errors='coerce')
            logger.debug("Converted termination_date to datetime")
        
        # Log sample of termination dates for debugging
        term_sample = df[df['termination_date'].notna()].head(5)
        if not term_sample.empty:
            logger.debug(f"Sample termination dates before processing:\n{term_sample['termination_date'].to_string()}")
        
        # Extra diagnostics for proration logic
        logger.debug(f"DEBUG: Checking for proration, year_start_date={year_start_date}, year_end_date={year_end_date}")
        logger.debug(f"DEBUG: Number with termination_date notna: {df['termination_date'].notna().sum()}")
        logger.debug(f"DEBUG: Number with termination_date >= year_start_date: {(df['termination_date'] >= year_start_date).sum()}")
        logger.debug(f"DEBUG: Number with termination_date <= year_end_date: {(df['termination_date'] <= year_end_date).sum()}")
        # Identify terminations that occurred during the plan year
        logger.debug("!!! ENTERED PRORATION BLOCK !!!")  # Diagnostic print
        term_mask = (
            df['termination_date'].notna() & 
            (df['termination_date'] >= year_start_date) & 
            (df['termination_date'] <= year_end_date)
        )
        
        term_count = term_mask.sum()
        logger.info(f"Found {term_count} participants terminated during plan year {simulation_year}")
        
        if term_count > 0:
            # Calculate days worked for terminated participants (inclusive of termination day)
            df.loc[term_mask, 'days_worked'] = (
                (df.loc[term_mask, 'termination_date'] - year_start_date).dt.days + 1
            )
            
            # Ensure days_worked is at least 1 and not greater than total_days
            df['days_worked'] = np.maximum(df['days_worked'], 1)
            df['days_worked'] = np.minimum(df['days_worked'], total_days)
            
            # Log some examples for verification
            logger.debug("Termination proration example(s):")
            num_examples = 3 if DEBUG_SAMPLE else 1
            for i, (idx, row) in enumerate(df[term_mask].head(num_examples).iterrows()):
                logger.debug(
                    "Example %d: Termination %s, Days worked: %s/%s, Gross comp: $%.2f",
                    i+1,
                    row['termination_date'].date(),
                    row['days_worked'],
                    total_days,
                    row['gross_compensation']
                )

    # 🔍 Year 1 Proration Stats
    if simulation_year == year_start_date.year:
        pr = df['days_worked'] / total_days
        logger.debug(
            f"[Year {simulation_year} Proration Stats] "
            f"min={pr.min():.4f}, max={pr.max():.4f}, mean={pr.mean():.4f}"
        )
    # Calculate proration factor based on days worked
    df['proration_factor'] = df['days_worked'] / total_days
    
    # Debug log proration factors
    logger.debug(f"Proration factor stats: min={df['proration_factor'].min():.4f}, "
                f"max={df['proration_factor'].max():.4f}, "
                f"mean={df['proration_factor'].mean():.4f}")
    
    # Apply proration to compensation
    df['plan_year_compensation'] = df['gross_compensation'] * df['proration_factor']
    df['capped_compensation'] = np.minimum(
        df['plan_year_compensation'], 
        statutory_comp_limit * df['proration_factor']
    )
    
    # Debug logging for verification
    if 'termination_date' in df.columns:
        logger.debug("Compensation calculation example for terminated participants:")
        for i, (idx, row) in enumerate(df[df['termination_date'].notna()].head(3).iterrows()):
            logger.debug(
                f"Example {i+1}: Termination {row.get('termination_date', 'N/A')}, "
                f"Proration: {row.get('proration_factor', 0):.4f}, "
                f"Gross: ${row.get('gross_compensation', 0):.2f}, "
                f"Plan year: ${row.get('plan_year_compensation', 0):.2f}"
            )

    # Clean up temporary columns
    df.drop(['days_worked', 'proration_factor'], axis=1, errors='ignore', inplace=True)

    active_mask = df['status'].isin(['Active', 'Unknown'])
    if 'is_eligible' not in df.columns:
        df['is_eligible'] = False
        eligible_mask = active_mask
    else:
        df['is_eligible'] = df['is_eligible'].fillna(False).astype(bool)
        eligible_mask = df['is_eligible']

    if 'deferral_rate' in df.columns and 'is_participating' in df.columns:
        df.loc[eligible_mask & (df['deferral_rate'] > 0), 'is_participating'] = True

    # contributions for employees who were ever eligible, regardless of status
    calc_mask = eligible_mask
    zero_contrib_cols = [
        'pre_tax_contributions',
        'employer_match_contribution',
        'employer_non_elective_contribution',
        'total_contributions'
    ]
    df.loc[~calc_mask, zero_contrib_cols] = 0.0

    if calc_mask.any():
        df.loc[calc_mask, 'current_age'] = calculate_age(
            df.loc[calc_mask, 'birth_date'], year_end_date)
        df['is_catch_up_eligible'] = False
        df.loc[calc_mask, 'is_catch_up_eligible'] = (
            df.loc[calc_mask, 'current_age'] >= catch_up_age)

        df['effective_deferral_limit'] = deferral_limit
        df.loc[calc_mask & df['is_catch_up_eligible'], 'effective_deferral_limit'] = (
            deferral_limit + catch_up_limit)

        potential_deferral = (
            df.loc[calc_mask, 'capped_compensation'] *
            df.loc[calc_mask, 'deferral_rate']
        )
        df.loc[calc_mask, 'pre_tax_contributions'] = np.minimum(
            potential_deferral,
            df.loc[calc_mask, 'effective_deferral_limit']
        )

        nec_rate = plan_rules.get('employer_nec', {}).get('rate', 0.0)
        df.loc[calc_mask, 'employer_non_elective_contribution'] = (
            df.loc[calc_mask, 'capped_compensation'] * nec_rate
        )

        if plan_rules.get('last_day_work_rule', False):
            term_mask = df['status'] == 'Terminated'
            df.loc[
                term_mask,
                ['employer_match_contribution', 'employer_non_elective_contribution']
            ] = 0.0
            logger.info(
                f"Last day work rule: zeroed match/NEC for {term_mask.sum()} terminated employees."
            )

        match_tiers = plan_rules.get('employer_match', {}).get('tiers', [])
        logger.debug("contributions.apply: match_tiers=%s", match_tiers)
        match_dollar_cap = plan_rules.get('employer_match', {}).get('dollar_cap')
        df.loc[calc_mask, 'employer_match_contribution'] = 0.0
        if match_tiers:
            # Vectorized multi-tier match calculation
            caps = np.array([t.get('cap_deferral_pct', 0.0) for t in match_tiers])
            rates = np.array([t.get('match_rate', 0.0) for t in match_tiers])
            prev_caps = np.concatenate(([0.0], caps[:-1]))
            # reshape for broadcasting: rows × tiers
            def_rates = df.loc[calc_mask, 'deferral_rate'].to_numpy()[:, None]
            comp_caps = df.loc[calc_mask, 'capped_compensation'].to_numpy()[:, None]
            # compute deferral amount in each tier
            def_in_tiers = np.clip(np.minimum(def_rates, caps) - prev_caps, 0.0, None)
            # compute match per tier and sum
            match_amounts = (def_in_tiers * comp_caps * rates).sum(axis=1)
            if match_dollar_cap is not None:
                match_amounts = np.minimum(match_amounts, match_dollar_cap)
            calculated_match = pd.Series(match_amounts, index=df.loc[calc_mask].index)
            df.loc[calc_mask, 'employer_match_contribution'] = calculated_match

        total_calc = (
            df.loc[calc_mask, 'pre_tax_contributions'] +
            df.loc[calc_mask, 'employer_match_contribution'] +
            df.loc[calc_mask, 'employer_non_elective_contribution']
        )

        exceeds = total_calc > overall_limit
        if exceeds.any():
            logger.warning(
                f"{exceeds.sum()} employees exceeded overall contribution limit of ${overall_limit}. Reducing employer contributions."
            )
            excess_amount = total_calc[exceeds] - overall_limit
            er_total = (
                df.loc[exceeds, 'employer_match_contribution'] +
                df.loc[exceeds, 'employer_non_elective_contribution']
            )
            reduction = pd.Series(1.0, index=excess_amount.index)
            nonzero = er_total > 0
            reduction[nonzero] = 1.0 - (
                excess_amount[nonzero] / er_total[nonzero]
            )
            reduction = np.maximum(reduction, 0)
            df.loc[exceeds, 'employer_match_contribution'] *= reduction
            df.loc[exceeds, 'employer_non_elective_contribution'] *= reduction
            total_calc[exceeds] = (
                df.loc[exceeds, 'pre_tax_contributions'] +
                df.loc[exceeds, 'employer_match_contribution'] +
                df.loc[exceeds, 'employer_non_elective_contribution']
            )

        df.loc[calc_mask, 'total_contributions'] = total_calc

    if 'current_age' in df.columns:
        df.drop(columns=['current_age'], inplace=True, errors='ignore')

    return df