"""
Functions for applying retirement plan rules within the simulation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re # Added for parsing match formula
from dateutil.relativedelta import relativedelta # Added for eligibility date calc
from utils.date_utils import calculate_age, calculate_tenure  # Use date utilities from utils
from utils.rules.formula_parsers import parse_match_formula, parse_match_tiers
from utils.rules.eligibility import apply as apply_eligibility
from utils.rules.auto_enrollment import apply as apply_auto_enrollment
from utils.rules.auto_increase import apply as apply_auto_increase
from utils.rules.contributions import apply as apply_contributions
from utils.rules.response import apply as _response_apply

def determine_eligibility(df, scenario_config, simulation_year_end_date):
    """Facade: delegate to utils.rules.eligibility.apply"""
    plan_rules = scenario_config.get('plan_rules', {})
    return apply_eligibility(df, plan_rules, simulation_year_end_date)

def apply_contributions(df, scenario_config, simulation_year, year_start_date, year_end_date):
    """
    Calculates employee and employer contributions for the simulation year.
    Updates contribution-related columns inplace.
    """
    print(f"  Calculating contributions for {simulation_year}...")
    plan_rules = scenario_config.get('plan_rules', {})
    irs_limits = scenario_config.get('irs_limits', {})
    # Use integer simulation_year for dictionary lookup
    year_limits = irs_limits.get(simulation_year, {}) # <-- Changed: Use integer year
    print(f"  DEBUG: IRS Limits for {simulation_year}: {year_limits}")

    # Use consistent keys (e.g., 'comp_limit' vs 'statutory_comp_limit')
    statutory_comp_limit = year_limits.get('comp_limit', 345000) # Example default
    deferral_limit = year_limits.get('deferral_limit', 23000) # Example default
    catch_up_limit = year_limits.get('catch_up', 7500)       # Example default ('catch_up' used in config)
    overall_limit = year_limits.get('overall_limit', 69000) # Example default (415 limit)
    catch_up_age = plan_rules.get('catch_up_age', 50)

    employer_match_formula = plan_rules.get('employer_match_formula', "")
    employer_non_elective_formula = plan_rules.get('employer_non_elective_formula', "")

    # --- Ensure required columns exist and initialize output columns ---
    # Check for essential input columns
    required_input_cols = ['gross_compensation', 'deferral_rate', 'birth_date', 'status', 'hire_date']
    missing_cols = [col for col in required_input_cols if col not in df.columns]
    if missing_cols:
        print(f"  Error: Required columns missing: {', '.join(missing_cols)}. Cannot calculate contributions.")
        return df # Return the DataFrame as is, even if calculations can't proceed

    # Initialize output columns if they don't exist
    output_cols = {
        'plan_year_compensation': 0.0,
        'capped_compensation': 0.0,
        'pre_tax_contributions': 0.0,
        'employer_match_contribution': 0.0,
        'employer_non_elective_contribution': 0.0,
        'total_contributions': 0.0,
        'is_catch_up_eligible': False, # Add this for clarity
        'effective_deferral_limit': 0.0 # Add this for clarity
    }
    for col, default_val in output_cols.items():
        if col not in df.columns:
            df[col] = default_val
        # Ensure correct dtype for calculations
        if 'contribution' in col or 'compensation' in col or 'limit' in col:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        elif col == 'is_catch_up_eligible':
            # Fill NA with False before converting to bool to avoid TypeError
            df[col] = df[col].fillna(False).astype(bool)

    # Ensure deferral_rate is numeric
    df['deferral_rate'] = pd.to_numeric(df['deferral_rate'], errors='coerce').fillna(0.0)

    # --- Filter for Eligible Employees (including terminated for prorated contributions) ---
    # Include Active, Unknown, and Terminated to calculate prorated contributions
    active_mask = df['status'].isin(['Active', 'Unknown', 'Terminated'])
    # Ensure is_eligible column exists and is boolean
    if 'is_eligible' not in df.columns:
        print("  Warning: 'is_eligible' column missing. Assuming all active employees are eligible for calculation.")
        df['is_eligible'] = False # Create if missing
        eligible_mask = active_mask # Treat all active as eligible if column missing
    else:
        # Fill NA with False then convert to bool safely
        df['is_eligible'] = df['is_eligible'].fillna(False).astype(bool)
        eligible_mask = df['is_eligible']

    # --- NEW: Mark eligible employees with non-zero deferral rates as participating ---
    # This catches employees contributing from the start (census data) or via AE/AI
    if 'deferral_rate' in df.columns and 'is_participating' in df.columns:
        df.loc[eligible_mask & (df['deferral_rate'] > 0), 'is_participating'] = True
    # --- END NEW ---

    calc_mask = active_mask & eligible_mask # Mask for rows where contributions should be calculated

    # Zero out contributions for non-applicable rows *before* calculations
    cols_to_zero_out = [
        'plan_year_compensation',
        'capped_compensation',
        'pre_tax_contributions',
        'employer_match_contribution',
        'employer_non_elective_contribution',
        'total_contributions'
    ]
    df.loc[~calc_mask, cols_to_zero_out] = 0.0

    # Proceed with calculations only for the masked rows
    if calc_mask.any():
        # --- Calculate Plan Year Compensation & Capped Compensation ---
        # Use gross_compensation for simplicity (or refine if needed)
        # Full-year compensation for active & eligible
        df.loc[calc_mask, 'plan_year_compensation'] = df.loc[calc_mask, 'gross_compensation'].fillna(0)
        df.loc[calc_mask, 'capped_compensation'] = np.minimum(df.loc[calc_mask, 'plan_year_compensation'], statutory_comp_limit)
        # Prorate for terminated employees within the year
        term_mask = (df['status'] == 'Terminated') & df['termination_date'].notna() \
            & df['termination_date'].between(year_start_date, year_end_date)
        if term_mask.any():
            total_days = (year_end_date - year_start_date).days
            days_worked = (df.loc[term_mask, 'termination_date'] - year_start_date).dt.days.clip(lower=0, upper=total_days)
            frac = days_worked / total_days
            # Prorate compensation and cap
            df.loc[term_mask, 'plan_year_compensation'] = df.loc[term_mask, 'gross_compensation'] * frac
            df.loc[term_mask, 'capped_compensation'] = np.minimum(df.loc[term_mask, 'plan_year_compensation'], statutory_comp_limit * frac)

        # --- Calculate Employee Deferrals (including Catch-up) ---
        # Calculate age only where needed
        df.loc[calc_mask, 'current_age'] = calculate_age(df.loc[calc_mask, 'birth_date'], year_end_date)
        df['is_catch_up_eligible'] = False # Initialize
        df.loc[calc_mask, 'is_catch_up_eligible'] = (df.loc[calc_mask, 'current_age'] >= catch_up_age)

        # Calculate effective limits for relevant employees
        df['effective_deferral_limit'] = deferral_limit
        df.loc[calc_mask & df['is_catch_up_eligible'], 'effective_deferral_limit'] = deferral_limit + catch_up_limit

        # Calculate potential and actual deferrals
        potential_deferral = df.loc[calc_mask, 'capped_compensation'] * df.loc[calc_mask, 'deferral_rate']
        df.loc[calc_mask, 'pre_tax_contributions'] = np.minimum(potential_deferral, df.loc[calc_mask, 'effective_deferral_limit'])

        # --- Calculate Employer Non-Elective Contribution ---
        nec_rate = 0.0
        if employer_non_elective_formula:
            # Simple parsing: assumes 'X%'
            match = re.match(r"\s*(\d+\.?\d*)\s*%\s*", employer_non_elective_formula, re.IGNORECASE)
            if match:
                nec_rate = float(match.group(1)) / 100.0
            else:
                print(f"  Warning: Could not parse non-elective formula: '{employer_non_elective_formula}'. Assuming 0%.")

        df.loc[calc_mask, 'employer_non_elective_contribution'] = df.loc[calc_mask, 'capped_compensation'] * nec_rate

        # Last Day Work Rule: zero match & NEC for terminated employees
        if plan_rules.get('last_day_work_rule', False):
            term_mask = df['status'] == 'Terminated'
            df.loc[term_mask, ['employer_match_contribution','employer_non_elective_contribution']] = 0.0
            print(f"  Last day work rule: zeroed match/NEC for {term_mask.sum()} terminated employees.")

        # --- Calculate Employer Match Contribution ---
        df.loc[calc_mask, 'employer_match_contribution'] = 0.0 # Initialize for the eligible group
        if employer_match_formula:
            # Parse tiered match formula using helper
            tiers = parse_match_tiers(employer_match_formula)

            if tiers:
                # Apply tiers cumulatively
                # Get employee deferral rate and capped compensation for the eligible group
                eligible_deferral_rate = df.loc[calc_mask, 'deferral_rate']
                eligible_capped_comp = df.loc[calc_mask, 'capped_compensation']
                calculated_match = pd.Series(0.0, index=df[calc_mask].index) # Initialize match for eligible
                last_deferral_cap = 0.0

                for tier in tiers:
                    current_deferral_cap = tier['deferral_cap_pct']
                    tier_match_rate = tier['match_pct']

                    # Deferral percentage within this tier's range
                    deferral_in_tier_range = np.minimum(eligible_deferral_rate, current_deferral_cap) - last_deferral_cap
                    deferral_in_tier_range = np.maximum(deferral_in_tier_range, 0) # Ensure non-negative

                    # Match applied to compensation based on deferral in this tier
                    match_for_tier = (deferral_in_tier_range * eligible_capped_comp) * tier_match_rate
                    calculated_match += match_for_tier

                    last_deferral_cap = current_deferral_cap

                df.loc[calc_mask, 'employer_match_contribution'] = calculated_match

        # --- Calculate Total Contributions and check 415 limit ---
        # Sum components for the eligible group
        total_calc = (df.loc[calc_mask, 'pre_tax_contributions'] +
                      df.loc[calc_mask, 'employer_match_contribution'] +
                      df.loc[calc_mask, 'employer_non_elective_contribution'])

        # Check overall 415 limit (Employer + Employee contributions)
        # Note: This simple check assumes pre-tax only; adjust if Roth or after-tax exists
        # We reduce employer contributions proportionally if the limit is exceeded.
        exceeds_limit = total_calc > overall_limit
        if exceeds_limit.any():
            print(f"  Warning: {exceeds_limit.sum()} employees exceeded the overall contribution limit of ${overall_limit}. Reducing employer contributions.")
            
            # Calculate the excess amount
            excess_amount = total_calc[exceeds_limit] - overall_limit
            
            # Calculate total ER contributions for those exceeding
            er_total = df.loc[exceeds_limit, 'employer_match_contribution'] + df.loc[exceeds_limit, 'employer_non_elective_contribution']
            
            # Calculate reduction factor (avoid division by zero)
            reduction_factor = pd.Series(1.0, index=excess_amount.index)
            non_zero_er = er_total > 0
            reduction_factor[non_zero_er] = 1.0 - (excess_amount[non_zero_er] / er_total[non_zero_er])
            reduction_factor = np.maximum(reduction_factor, 0) # Ensure factor is not negative
            
            # Apply reduction factor to ER contributions
            df.loc[exceeds_limit, 'employer_match_contribution'] *= reduction_factor
            df.loc[exceeds_limit, 'employer_non_elective_contribution'] *= reduction_factor
            
            # Recalculate total contributions for those affected
            total_calc[exceeds_limit] = (df.loc[exceeds_limit, 'pre_tax_contributions'] +
                                         df.loc[exceeds_limit, 'employer_match_contribution'] +
                                         df.loc[exceeds_limit, 'employer_non_elective_contribution'])
        
        # Assign final total contributions
        df.loc[calc_mask, 'total_contributions'] = total_calc

    # --- Final clean up and reporting ---
    # Drop temporary age column if added
    if 'current_age' in df.columns and 'current_age' not in required_input_cols and 'current_age' not in output_cols:
         df = df.drop(columns=['current_age'], errors='ignore')
    
    # Report summary statistics (only for those calculated)
    if calc_mask.any():
        avg_deferral = df.loc[calc_mask, 'pre_tax_contributions'].mean()
        avg_match = df.loc[calc_mask & (df['employer_match_contribution'] > 0), 'employer_match_contribution'].mean() # Avg only for those receiving match
        avg_nec = df.loc[calc_mask & (df['employer_non_elective_contribution'] > 0), 'employer_non_elective_contribution'].mean() # Avg only for those receiving NEC
        print(f"  Contributions calculated. Avg Deferral: ${avg_deferral:,.2f}, Avg Match: ${avg_match:,.2f}, Avg NEC: ${avg_nec:,.2f}")
    else:
        print("  No active and eligible employees found for contribution calculation.")

    return df # <-- Added: Explicitly return the modified DataFrame

def calculate_contributions(df, scenario_config, simulation_year, year_start_date, year_end_date):
    return apply_contributions(df, scenario_config, simulation_year, year_start_date, year_end_date)

def apply_plan_change_deferral_response(df, current_scenario_config, baseline_scenario_config, simulation_year, start_year):
    return _response_apply(df, current_scenario_config, baseline_scenario_config, simulation_year, start_year)

def process_year(df, scenario_cfg, baseline_cfg, year_dates):
    df = apply_plan_change_deferral_response(df, scenario_cfg, baseline_cfg, year_dates.year, scenario_cfg['start_year'])
    df = apply_contributions(
        df,
        scenario_cfg,
        year_dates.year,
        year_dates.start_date,
        year_dates.end_date
    )
    return df
