"""
Functions for applying retirement plan rules within the simulation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re # Added for parsing match formula
from dateutil.relativedelta import relativedelta # Added for eligibility date calc
from utils.date_utils import calculate_age, calculate_tenure  # Use date utilities from utils

def determine_eligibility(df, scenario_config, simulation_year_end_date):
    """
    Determines plan eligibility based on age and service rules defined in the scenario.

    Updates the 'is_eligible' column in the DataFrame inplace.

    Args:
        df (pd.DataFrame): The census DataFrame for the current simulation year.
                           Must include 'birth_date' and 'hire_date'.
        scenario_config (dict): The configuration dictionary for the simulation scenario.
                                Must contain 'plan_rules' with 'eligibility_age' and 
                                'eligibility_service_months'.
        simulation_year_end_date (pd.Timestamp): The end date of the current simulation year.
    """
    print(f"  Determining eligibility for {simulation_year_end_date.year}...")
    plan_rules = scenario_config.get('plan_rules', {})
    eligibility_rules = plan_rules.get('eligibility', {}) # Access nested dict
    min_age = eligibility_rules.get('min_age', 21)
    min_service_months = eligibility_rules.get('min_service_months', 12)

    if 'birth_date' not in df.columns or 'hire_date' not in df.columns:
        print("  Warning: 'birth_date' or 'hire_date' columns missing. Cannot determine eligibility.")
        if 'is_eligible' not in df.columns:
             df['is_eligible'] = False # Initialize column if missing
        if 'eligibility_entry_date' not in df.columns:
             df['eligibility_entry_date'] = pd.NaT
        return df # <-- Modified: Return df even on early exit

    # Ensure dates are in datetime format
    df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')
    df['hire_date'] = pd.to_datetime(df['hire_date'], errors='coerce')

    # Handle potential NaT dates after coercion
    if df['birth_date'].isnull().any() or df['hire_date'].isnull().any():
        print("  Warning: Some 'birth_date' or 'hire_date' values could not be parsed. Affected rows may not be marked eligible.")
        # Optionally, fill NaNs in calculated columns if needed downstream
        # df[['current_age', 'current_tenure_months']] = df[['current_age', 'current_tenure_months']].fillna(-1) # Example

    # Calculate age and tenure as of the simulation year end date
    # Check for NaT before calculation to avoid errors/warnings
    valid_birth_date = df['birth_date'].notna()
    valid_hire_date = df['hire_date'].notna()

    df['current_age'] = pd.NA # Initialize
    # df['current_tenure_months'] = pd.NA # Initialize

    df.loc[valid_birth_date, 'current_age'] = calculate_age(df.loc[valid_birth_date, 'birth_date'], simulation_year_end_date)
    # df.loc[valid_hire_date, 'current_tenure_months'] = (calculate_tenure(df.loc[valid_hire_date, 'hire_date'], simulation_year_end_date) * 12).round()

    # --- Calculate Eligibility Entry Date --- #
    # (This logic now happens *before* determining the 'is_eligible' flag for the current year end)
    if 'eligibility_entry_date' not in df.columns:
        df['eligibility_entry_date'] = pd.NaT
    else: # Ensure it's datetime
        df['eligibility_entry_date'] = pd.to_datetime(df['eligibility_entry_date'], errors='coerce')

    # Calculate date service requirement is met
    service_met_date = df['hire_date'] + df['hire_date'].apply(lambda x: relativedelta(months=min_service_months) if pd.notna(x) else pd.NaT)

    # Calculate date age requirement is met
    age_met_date = df['birth_date'] + df['birth_date'].apply(lambda x: relativedelta(years=min_age) if pd.notna(x) else pd.NaT)

    # Eligibility date is the LATER of the two requirement dates
    # Use .loc for assignment to avoid SettingWithCopyWarning
    df.loc[:, 'eligibility_entry_date'] = np.maximum(service_met_date.fillna(pd.Timestamp.min), 
                                                     age_met_date.fillna(pd.Timestamp.min))
    # Handle cases where one date is NaT - np.maximum treats NaT like negative infinity
    df.loc[service_met_date.isna() | age_met_date.isna(), 'eligibility_entry_date'] = pd.NaT 
    df.loc[df['eligibility_entry_date'] == pd.Timestamp.min, 'eligibility_entry_date'] = pd.NaT # Clean up potential min timestamps

    # Convert back to datetime just in case operations changed dtype
    df['eligibility_entry_date'] = pd.to_datetime(df['eligibility_entry_date'], errors='coerce')
    # --- End Eligibility Entry Date Calc ---

    # --- Determine Eligibility Status (as of simulation_year_end_date) ---
    # Use .loc to avoid SettingWithCopyWarning
    # An employee is eligible at year-end if their entry date is on or before the year-end 
    # and they are currently active (or unknown status).
    is_eligible_based_on_date = (df['eligibility_entry_date'] <= simulation_year_end_date) & df['eligibility_entry_date'].notna()

    # Consider active status ('Unknown' counts as potentially active for eligibility)
    is_active = df['status'].isin(['Active', 'Unknown'])

    df['is_eligible'] = (is_eligible_based_on_date & is_active)
    # Apply hours_worked requirement if configured
    min_hours = plan_rules.get('min_hours_worked', 0)
    if min_hours > 0:
        if 'hours_worked' in df.columns:
            below = df['hours_worked'] < min_hours
            df.loc[below, 'is_eligible'] = False
            print(f"  Hours requirement: set {below.sum()} employees ineligible (<{min_hours} hours).")
        else:
            print("  Warning: 'hours_worked' missing; skipping hours requirement.")
    eligible_count = df['is_eligible'].sum()
    print(f"  Eligibility determined: {eligible_count} eligible employees.")

    return df


# --- Auto Enrollment --- #
def apply_auto_enrollment(df, scenario_config, simulation_year_start_date, simulation_year_end_date):
    """
    Applies auto-enrollment rules to employees who became eligible during the simulation year.

    Args:
        df (pd.DataFrame): The employee census dataframe.
        scenario_config (dict): Configuration for the scenario.
        simulation_year_start_date (pd.Timestamp): Start date of the simulation year.
        simulation_year_end_date (pd.Timestamp): End date of the simulation year.

    Returns:
        pd.DataFrame: DataFrame with updated participation status and deferral rates.
    """
    simulation_year = simulation_year_end_date.year
    print(f"  Applying Auto Enrollment for {simulation_year}...")
    plan_rules = scenario_config.get('plan_rules', {})
    ae_rules = plan_rules.get('auto_enrollment', {})
    ae_enabled = ae_rules.get('enabled', False)
    ae_default_rate = ae_rules.get('default_rate', 0.0)
    ae_outcome_dist = ae_rules.get('ae_outcome_distribution', {})

    if not ae_enabled:
        print("  Auto Enrollment disabled. Skipping.")
        return df

    if 'is_eligible' not in df.columns or 'is_participating' not in df.columns or \
       'deferral_rate' not in df.columns or 'ae_opted_out' not in df.columns or \
       'eligibility_entry_date' not in df.columns: # Added check
        print("  Warning: Required columns missing for Auto Enrollment. Skipping.")
        return df

    # Initialize tracking columns if missing
    if 'enrollment_date' not in df.columns:
        df['enrollment_date'] = pd.NaT
    if 'first_contribution_date' not in df.columns:
        df['first_contribution_date'] = pd.NaT
    if 'ae_opt_out_date' not in df.columns:
        df['ae_opt_out_date'] = pd.NaT

    # --- Setup AE window and enrollment flags ---
    window_days = ae_rules.get('window_days', None)
    df['ae_window_start'] = df['eligibility_entry_date']
    if window_days is not None:
        df['ae_window_end'] = df['eligibility_entry_date'] + timedelta(days=window_days)
    else:
        df['ae_window_end'] = pd.NaT

    df['proactive_enrolled'] = False
    df['auto_enrolled'] = False
    # Debug booleans for AE process
    df['became_eligible_during_year'] = False
    df['window_closed_during_year'] = False
    # --- Proactive Enrollment at Eligibility Entry (Debug) ---
    proactive_p = ae_rules.get('proactive_enrollment_probability', 0.0)
    print(f"  [Proactive AE Debug] probability={proactive_p:.2%}")
    if 'eligibility_entry_date' in df.columns:
        newly_eligible = (
            (df['eligibility_entry_date'] >= simulation_year_start_date) &
            (df['eligibility_entry_date'] <= simulation_year_end_date)
        )
        print(f"  [Proactive AE Debug] newly eligible count={newly_eligible.sum()}")
        active = df['status'] == 'Active'
        not_part = ~df['is_participating']
        not_opted = ~df['ae_opted_out']
        candidates = newly_eligible & active & not_part & not_opted
        print(f"  [Proactive AE Debug] candidate count={candidates.sum()}")
        idxs = df.index[candidates]
        if len(idxs) > 0:
            draws = np.random.rand(len(idxs))
            selected = idxs[draws < proactive_p]
            # Determine deferral rate using configured distribution for proactive enrollment
            distribution = ae_rules.get('proactive_rate_range', None)
            if distribution is not None:
                min_rate, max_rate = distribution
                rates = np.random.uniform(min_rate, max_rate, size=len(selected))
                df.loc[selected, 'deferral_rate'] = rates
            else:
                df.loc[selected, 'deferral_rate'] = ae_default_rate
            # Set enrollment flags and dates
            df.loc[selected, 'is_participating'] = True
            df.loc[selected, 'proactive_enrolled'] = True
            df.loc[selected, 'enrollment_date'] = df.loc[selected, 'eligibility_entry_date']
            df.loc[selected, 'first_contribution_date'] = df.loc[selected, 'eligibility_entry_date']
            print(f"  {len(selected)} proactively enrolled at eligibility (p={proactive_p:.2%})")

    # NEW: Re-enroll existing active eligible participants below default rate
    re_enroll_existing = ae_rules.get('re_enroll_existing', False)
    if re_enroll_existing:
        if 'auto_reenrolled' not in df.columns:
            df['auto_reenrolled'] = False
        mask_reenroll = (
            (df.get('status') == 'Active') &
            (df.get('is_eligible', False)) &
            (df.get('is_participating', False)) &
            (df['deferral_rate'] > 0) &
            (df['deferral_rate'] < ae_default_rate)
        )
        if mask_reenroll.any():
            df.loc[mask_reenroll, 'deferral_rate'] = ae_default_rate
            df.loc[mask_reenroll, 'is_participating'] = True
            df.loc[mask_reenroll, 'enrollment_method'] = 'AE'
            df.loc[mask_reenroll, 'auto_reenrolled'] = True
            print(f"  Re-enrolled {mask_reenroll.sum()} existing participants at default rate {ae_default_rate:.2%}")

    # --- Define AE Target Window ---
    # Employees whose AE window closes in this simulation year
    within_window_closure = (
        (df['ae_window_end'] >= simulation_year_start_date) &
        (df['ae_window_end'] <= simulation_year_end_date)
    )
    # Flag those whose AE window closed this year
    df.loc[within_window_closure, 'window_closed_during_year'] = True

    # Target active, eligible, non-participating, non-opted-out employees
    ae_target_mask = (
        df['is_eligible'] &
        (~df['is_participating']) &
        (~df['ae_opted_out']) &
        within_window_closure
    )

    num_targeted = ae_target_mask.sum()
    if num_targeted == 0:
        print("  No employees targeted for Auto Enrollment this year.")
        return df
    # --- END NEW LOGIC ---

    print(f"  Targeting {num_targeted} newly eligible, non-participating employees for AE.")

    # --- Apply AE Outcomes --- # 
    # Ensure distribution sums to 1 (or close enough)
    stay_default_prob = ae_outcome_dist.get('stay_default', 0.0)
    opt_out_prob = ae_outcome_dist.get('opt_out', 0.0)
    # increase_to_match_cap_prob = ae_outcome_dist.get('increase_to_match_cap', 0.0) # Currently unused

    total_prob = stay_default_prob + opt_out_prob # + increase_to_match_cap_prob
    if not np.isclose(total_prob, 1.0) and total_prob > 0:
        print(f"  Warning: AE outcome probabilities sum to {total_prob:.4f}. Normalizing.")
        stay_default_prob /= total_prob
        opt_out_prob /= total_prob
        # increase_to_match_cap_prob /= total_prob

    if total_prob <= 0:
         print("  Warning: AE outcome probabilities are zero or invalid. No changes applied.")
         return df

    outcomes = ['stay_default', 'opt_out']#, 'increase_to_match_cap']
    probabilities = [stay_default_prob, opt_out_prob]#, increase_to_match_cap_prob]

    # Get indices of targeted employees
    target_indices = df.index[ae_target_mask]

    # Assign random outcomes based on distribution
    assigned_outcomes = np.random.choice(outcomes, size=num_targeted, p=probabilities)

    # Apply outcomes using .loc
    for i, outcome in enumerate(assigned_outcomes):
        idx = target_indices[i]
        if outcome == 'stay_default':
            # Auto-enrollment at window closure with variable rate
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
        elif outcome == 'opt_out':
            # Opt-out record
            df.loc[idx, 'ae_opted_out'] = True
            df.loc[idx, 'ae_opt_out_date'] = df.loc[idx, 'ae_window_end']
            df.loc[idx, 'is_participating'] = False # Should already be false, but explicit
        # elif outcome == 'increase_to_match_cap': # Future enhancement?
        #     # Need match cap logic here...
        #     pass

    enrolled_count = df.loc[target_indices, 'is_participating'].sum()
    opted_out_count = df.loc[target_indices, 'ae_opted_out'].sum()
    print(f"  AE Applied: {enrolled_count} enrolled at {ae_default_rate*100:.1f}%, {opted_out_count - enrolled_count} opted out.") # Opt-out counts those enrolled too

    # Retain AE window columns for raw output
    return df


# --- Auto Increase --- #
def apply_auto_increase(df, scenario_config, simulation_year):
    """
    Applies auto-increase rules to participating employees who haven't opted out.
    Updates 'deferral_rate' and 'ai_enrolled' columns inplace.
    """
    plan_rules = scenario_config.get('plan_rules', {})
    ai_config = plan_rules.get('auto_increase', {})
    ai_enabled = ai_config.get('enabled', False)
    ai_increase_rate = ai_config.get('increase_rate', 0.01)
    ai_max_deferral_rate = ai_config.get('cap_rate', ai_config.get('max_deferral_rate', 0.10))
    if not ai_enabled:
        return df
    # Initialize flags if missing
    if 'ai_opted_out' not in df.columns:
        df['ai_opted_out'] = False
    # Preserve prior AI enrollment; initialize flag only if missing
    if 'ai_enrolled' not in df.columns:
        df['ai_enrolled'] = False
    else:
        df['ai_enrolled'] = df['ai_enrolled'].fillna(False)
    # Determine eligible mask
    is_active = (df['status'] == 'Active') if 'status' in df.columns else True
    mask = is_active & df['is_participating'] & (~df['ai_opted_out']) & (df['deferral_rate'] < ai_max_deferral_rate)
    # Optionally restrict to new hires
    if ai_config.get('apply_to_new_hires_only', False):
        year_start = pd.Timestamp(f"{simulation_year}-01-01")
        year_end = pd.Timestamp(f"{simulation_year}-12-31")
        mask &= (df['hire_date'] >= year_start) & (df['hire_date'] <= year_end)
    # Apply increase
    if mask.any():
        new_rates = np.minimum(df.loc[mask, 'deferral_rate'] + ai_increase_rate, ai_max_deferral_rate)
        df.loc[mask, 'deferral_rate'] = new_rates
        df.loc[mask, 'ai_enrolled'] = True
    return df


def calculate_contributions(df, scenario_config, simulation_year, year_start_date, year_end_date):
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

    # --- Filter for Active & Eligible Employees Only ---
    # Align status check with determine_eligibility (handle 'Unknown' from NaN fill)
    active_mask = df['status'].isin(['Active', 'Unknown'])
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
        df.loc[calc_mask, 'plan_year_compensation'] = df.loc[calc_mask, 'gross_compensation'].fillna(0)
        df.loc[calc_mask, 'capped_compensation'] = np.minimum(df.loc[calc_mask, 'plan_year_compensation'], statutory_comp_limit)

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
            # Parse tiered match formula (e.g., "100% up to 3%, 50% up to 5%")
            tiers = []
            try:
                parts = employer_match_formula.split(',')
                for part in parts:
                    match_re = re.match(r"\s*(\d+\.?\d*)\s*%\s+up\s+to\s+(\d+\.?\d*)\s*%", part.strip(), re.IGNORECASE)
                    if match_re:
                        match_percent = float(match_re.group(1)) / 100.0
                        deferral_cap_percent = float(match_re.group(2)) / 100.0
                        tiers.append({'match_pct': match_percent, 'deferral_cap_pct': deferral_cap_percent})
                tiers.sort(key=lambda x: x['deferral_cap_pct']) # Sort by deferral cap
            except Exception as e:
                print(f"  Warning: Could not parse match formula '{employer_match_formula}': {e}. Assuming 0% match.")
                tiers = [] # Reset tiers on parse error

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

def parse_match_formula(formula_str):
    """
    Parses an employer match formula string (e.g., "50% up to 6%") into components.

    Args:
        formula_str (str): The match formula string. Assumes format like 
                           "X% up to Y%" or "X%". Returns (0, 0) if empty or invalid.

    Returns:
        tuple: A tuple containing (match_rate, cap_deferral_percentage). 
               Returns (0, 0) if parsing fails or formula is empty.
               Rates are returned as decimals (e.g., 0.50 for 50%).
    """
    if not formula_str or not isinstance(formula_str, str):
        return 0.0, 0.0 # No formula provided or invalid type

    formula_str = formula_str.strip()
    # Simple case: Flat match rate (e.g., "5%") - Treat as having no cap implicitly
    match_simple = re.match(r"\s*(\d+\.?\d*)\s*%", formula_str)
    if match_simple and "up to" not in formula_str.lower():
        match_rate = float(match_simple.group(1)) / 100.0
        return match_rate, 1.0 # Assume cap is effectively 100% deferral if not specified

    # Standard case: "X% up to Y%"
    match_standard = re.match(r"(\d+\.?\d*)\s*%\s+up\s+to\s+(\d+\.?\d*)\s*%", formula_str, re.IGNORECASE)
    if match_standard:
        match_rate = float(match_standard.group(1)) / 100.0
        cap_deferral_perc = float(match_standard.group(2)) / 100.0
        return match_rate, cap_deferral_perc
    
    # Tiered or complex formulas are not handled by this simple parser
    # You might need a more sophisticated approach for those.
    print(f"  Warning: Could not parse match formula: '{formula_str}'. Returning (0, 0).")
    return 0.0, 0.0

def apply_plan_change_deferral_response(df, current_scenario_config, baseline_scenario_config, simulation_year, start_year):
    """
    Models existing participants' potential deferral rate increase in response to 
    a more generous match formula compared to a baseline scenario.
    
    Updates 'deferral_rate' column inplace.
    """
    print(f"  Applying plan change response logic for {simulation_year}...")
    
    # --- Get Configuration ---
    plan_rules = current_scenario_config.get('plan_rules', {})
    response_config = plan_rules.get('match_change_response', {})
    response_enabled = response_config.get('enabled', False)
    increase_probability = response_config.get('increase_probability', 0.0)
    increase_target = response_config.get('increase_target', 'optimal') #'optimal' is the only supported target currently

    # --- Basic Checks ---
    if not response_enabled:
        print("  Plan change response logic is disabled.")
        return
        
    if not baseline_scenario_config:
        print("  Baseline scenario config not provided. Cannot apply plan change response logic.")
        return
        
    # --- Trigger Condition: Run only in the first year of the simulation ---
    # This prevents the logic from running every year, assuming the plan change happens once.
    # Could be made more sophisticated later to detect changes dynamically if needed.
    if simulation_year != start_year:
        # print(f"  Skipping plan change response logic (not the start year: {start_year}).") # Optional: Reduce verbosity
        return
        
    # --- Ensure required columns exist ---
    if 'is_participating' not in df.columns or 'deferral_rate' not in df.columns:
        print("  Warning: Required columns missing. Cannot apply plan change response logic.")
        return
    if 'status' not in df.columns:
        print("  Warning: 'status' column missing. Assuming all are Active.")
        is_active = True
    else:
        is_active = (df['status'] == 'Active')

    # --- Compare Match Formulas ---
    current_match_formula = plan_rules.get('employer_match_formula', "")
    baseline_plan_rules = baseline_scenario_config.get('plan_rules', {})
    baseline_match_formula = baseline_plan_rules.get('employer_match_formula', "")

    if current_match_formula == baseline_match_formula:
        print("  Match formulas are the same as baseline. No plan change response needed.")
        return

    # Parse formulas
    current_rate, current_cap = parse_match_formula(current_match_formula)
    baseline_rate, baseline_cap = parse_match_formula(baseline_match_formula)

    # --- Determine if Current Match is More Generous ---
    # Simple check: Higher cap is more generous. If caps are equal, higher rate is more generous.
    # More complex generosity definitions could be added later.
    is_more_generous = False
    if current_cap > baseline_cap:
        is_more_generous = True
    elif current_cap == baseline_cap and current_rate > baseline_rate:
         is_more_generous = True
    # Add condition for potentially switching *to* a match from no match
    elif current_match_formula and not baseline_match_formula:
         is_more_generous = True

    if not is_more_generous:
        print(f"  Current match formula ('{current_match_formula}') not considered more generous than baseline ('{baseline_match_formula}').")
        return
        
    print(f"  Detected more generous match formula ('{current_match_formula}' vs baseline '{baseline_match_formula}'). Applying response logic.")

    # --- Identify Target Group ---
    # Participants who are active, participating, and deferring less than the *new* optimal rate.
    optimal_deferral_rate = current_cap # The cap percentage is the rate needed to maximize typical %-up-to-% match
    
    if optimal_deferral_rate <= 0:
        print("  Warning: Cannot determine optimal deferral rate from current match formula. Skipping response.")
        return

    response_target_mask = (
        is_active &
        (df['is_participating'] == True) &
        (df['deferral_rate'] > 0) & # Already participating
        (df['deferral_rate'] < optimal_deferral_rate)
    )
    
    target_indices = df.index[response_target_mask]
    num_targets = len(target_indices)

    if num_targets == 0:
        print("  No participating employees found deferring below the new optimal rate.")
        return

    print(f"  {num_targets} participants identified as potentially increasing deferral to {optimal_deferral_rate:.2%}.")

    # --- Simulate Response ---
    # Select subset based on probability
    increase_decision = np.random.rand(num_targets) < increase_probability
    increase_indices = target_indices[increase_decision]
    num_increasing = len(increase_indices)

    if num_increasing > 0:
        print(f"  Applying deferral increase to {num_increasing} participants (Probability: {increase_probability:.2f}).")
        # Apply Change (currently only 'optimal' target is supported)
        if increase_target == 'optimal':
            df.loc[increase_indices, 'deferral_rate'] = optimal_deferral_rate
        else:
             print(f"  Warning: Unsupported 'increase_target': {increase_target}. No change applied.")
    else:
        print("  No participants selected to increase deferral based on probability.")
