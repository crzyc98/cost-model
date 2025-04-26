import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from scipy.stats import truncnorm
import warnings

# Suppress specific Pandas warnings if needed (optional)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# --- Helper: Generate single employee record ---
def _generate_employee(year, plan_end_date, is_new_hire, existing_ssns,
                      min_working_age, max_working_age, age_mean, age_std_dev,
                      tenure_mean_years, tenure_std_dev_years, max_tenure_years,
                      role_distribution, role_compensation_params,
                      deferral_distribution, unique_id):
    """Generates a dictionary representing a single employee."""
    record = {}
    roles = list(role_distribution.keys())
    role_probs = list(role_distribution.values())
    deferral_rates = list(deferral_distribution.keys())
    deferral_probs = list(deferral_distribution.values())

    # Assign Role
    assigned_role = np.random.choice(roles, p=role_probs)
    record['role'] = assigned_role
    role_params = role_compensation_params[assigned_role]

    # Generate SSN (ensure uniqueness)
    prefix = "NH" if is_new_hire else "EX" # EX for initial population
    ssn = f"DUMMY_{prefix}_{np.random.randint(100000, 999999)}_{unique_id:06d}"
    retry_count = 0
    while ssn in existing_ssns and retry_count < 100:
        ssn = f"DUMMY_{prefix}_{np.random.randint(100000, 999999)}_{unique_id:06d}_{np.random.randint(100)}"
        retry_count += 1
    if ssn in existing_ssns: # Extremely unlikely after retries
        raise ValueError(f"Could not generate unique SSN after {retry_count} retries.")
    existing_ssns.add(ssn)
    record['ssn'] = ssn

    # Generate Age/Birth Date
    safe_age_std_dev = max(age_std_dev, 1e-6)
    age_a, age_b = (min_working_age - age_mean) / safe_age_std_dev, (max_working_age - age_mean) / safe_age_std_dev
    age = truncnorm.rvs(age_a, age_b, loc=age_mean, scale=safe_age_std_dev) if age_std_dev > 0 else age_mean
    age = np.clip(int(round(age)), min_working_age, max_working_age)
    record['age'] = age # Store age calculated as of plan_end_date

    birth_year = plan_end_date.year - age
    birth_month = np.random.randint(1, 13)
    birth_day = np.random.randint(1, 29) # Avoid issues with Feb 30/31 etc.
    record['birth_date'] = datetime(birth_year, birth_month, birth_day)

    # Generate Hire Date and Tenure
    if is_new_hire:
        # Hire date within the current year
        plan_start_date = datetime(year, 1, 1)
        min_possible_hire_date = record['birth_date'] + timedelta(days=365.25 * min_working_age)
        hire_start_bound = max(plan_start_date, min_possible_hire_date)
        if hire_start_bound > plan_end_date: hire_date = plan_end_date # Hired on last day if bounds cross
        else:
            days_in_hire_period = max(0, (plan_end_date - hire_start_bound).days) + 1
            hire_offset = np.random.randint(0, days_in_hire_period)
            hire_date = hire_start_bound + timedelta(days=hire_offset)
        record['hire_date'] = hire_date
        tenure_years = (plan_end_date - hire_date).days / 365.25
    else: # Existing employee (from initial generation or previous year)
        # Generate tenure first, then calculate hire date
        safe_tenure_std_dev = max(tenure_std_dev_years, 1e-6)
        # Ensure tenure doesn't imply hiring before min working age
        max_possible_tenure = age - min_working_age
        effective_max_tenure = min(max_tenure_years, max_possible_tenure)
        if effective_max_tenure <= 0:
            tenure_years = 0
        else:
            tenure_a, tenure_b = (0 - tenure_mean_years) / safe_tenure_std_dev, (effective_max_tenure - tenure_mean_years) / safe_tenure_std_dev
            tenure_years = truncnorm.rvs(tenure_a, tenure_b, loc=tenure_mean_years, scale=safe_tenure_std_dev) if tenure_std_dev_years > 0 else tenure_mean_years
            tenure_years = np.clip(tenure_years, 0, effective_max_tenure)

        # Calculate hire date based on tenure relative to plan_end_date
        hire_date = plan_end_date - timedelta(days=tenure_years * 365.25)
        record['hire_date'] = hire_date

    record['tenure'] = tenure_years # Store tenure calculated as of plan_end_date

    # Calculate Compensation (consistent logic for new/existing based on current age/tenure)
    age_experience_years = record['age'] - min_working_age
    target_comp = (role_params['comp_base_salary'] +
                   (age_experience_years * role_params['comp_increase_per_age_year']) +
                   (record['tenure'] * role_params['comp_increase_per_tenure_year']))

    log_mean = np.log(max(1000, target_comp * role_params['comp_log_mean_factor']))
    comp = np.random.lognormal(mean=log_mean, sigma=role_params['comp_spread_sigma'])
    record['gross_compensation'] = round(max(role_params['comp_min_salary'], comp), 2)

    # Assign deferral rate
    record['pre_tax_deferral_percentage'] = np.random.choice(deferral_rates, p=deferral_probs)
    # Termination date is handled in the main loop
    record['termination_date'] = pd.NaT

    return record

# --- Helper: Calculate derived fields ---
def _calculate_derived_fields(df, year, limits, nec_rate, match_rate, match_cap):
    """Calculates plan year comp, capped comp, and contributions."""
    plan_end_date = datetime(year, 12, 31)
    plan_start_date = datetime(year, 1, 1)
    limits_for_year = limits.get(year, limits.get(max(limits.keys()))) # Use latest if year missing

    comp_limit = limits_for_year['comp_limit']
    deferral_limit = limits_for_year['deferral_limit']
    catch_up_limit = limits_for_year['catch_up']

    df_calc = df.copy()

    # Calculate Plan Year Compensation (Prorated for hire/term date)
    df_calc['plan_start_date'] = plan_start_date
    df_calc['plan_end_date'] = plan_end_date
    df_calc['hire_date'] = pd.to_datetime(df_calc['hire_date'])
    # Use actual termination date if present, otherwise use plan end date
    df_calc['effective_term_date'] = pd.to_datetime(df_calc['termination_date']).fillna(plan_end_date)

    df_calc['service_start'] = df_calc[['hire_date', 'plan_start_date']].max(axis=1)
    df_calc['service_end'] = df_calc[['effective_term_date', 'plan_end_date']].min(axis=1)

    # Calculate days worked in the plan year, handle edge cases
    df_calc['days_in_plan_year'] = (df_calc['service_end'] - df_calc['service_start']).dt.days + 1
    df_calc['days_in_plan_year'] = df_calc['days_in_plan_year'].clip(lower=0) # Ensure non-negative

    total_days_in_year = (plan_end_date - plan_start_date).days + 1
    df_calc['proration_factor'] = df_calc['days_in_plan_year'] / total_days_in_year

    df_calc['plan_year_compensation'] = (df_calc['gross_compensation'] * df_calc['proration_factor']).round(2)

    # Apply Compensation Limit
    df_calc['capped_compensation'] = df_calc['plan_year_compensation'].clip(upper=comp_limit)

    # Calculate Contributions
    # Deferrals
    # Ensure 'birth_date' is datetime before calculating age
    df_calc['birth_date'] = pd.to_datetime(df_calc['birth_date'])
    df_calc['age'] = df_calc.apply(lambda row: (plan_end_date.year - row['birth_date'].year -
                                     ((plan_end_date.month, plan_end_date.day) <
                                      (row['birth_date'].month, row['birth_date'].day))), axis=1)

    eligible_for_catch_up = df_calc['age'] >= 50
    max_deferral = np.where(eligible_for_catch_up, deferral_limit + catch_up_limit, deferral_limit)

    df_calc['calculated_deferral'] = (df_calc['plan_year_compensation'] * (df_calc['pre_tax_deferral_percentage'] / 100.0)).round(2)
    df_calc['pre_tax_contributions'] = df_calc['calculated_deferral'].clip(upper=max_deferral)

    # NEC
    df_calc['employer_non_elective_contribution'] = (df_calc['plan_year_compensation'] * nec_rate).round(2)

    # Match
    deferral_eligible_for_match = (df_calc['plan_year_compensation'] * match_cap).round(2)
    actual_deferral_for_match = df_calc['pre_tax_contributions'].clip(upper=deferral_eligible_for_match)
    df_calc['employer_match_contribution'] = (actual_deferral_for_match * match_rate).round(2)

    # Select and order final columns
    final_cols = ['ssn', 'role', 'birth_date', 'hire_date', 'termination_date',
                  'gross_compensation', 'plan_year_compensation', 'capped_compensation',
                  'pre_tax_deferral_percentage', 'pre_tax_contributions',
                  'employer_non_elective_contribution', 'employer_match_contribution']
    # Ensure all columns exist before returning
    return df_calc[[col for col in final_cols if col in df_calc.columns]]

# --- Main Census Generation Function (Refactored) ---
def create_dummy_census_files(
    # --- File & Population Settings ---
    num_years=5, # Increased default to match previous runs
    base_year=2024,
    total_population=10000, # Increased default
    termination_rate=0.10, # Annual term rate applied to previous year's pop
    # --- Age Distribution ---
    age_mean=42,
    age_std_dev=12,
    min_working_age=18,
    max_working_age=70,
    # --- Tenure Distribution (For Initial Population & Updates) ---
    tenure_mean_years=7,
    tenure_std_dev_years=5,
    #min_tenure_years=0, # Implicitly handled
    max_tenure_years=40,
    # --- Role Configuration ---
    role_distribution={
        'Staff': 0.75, 'Manager': 0.20, 'Executive': 0.05
    },
    role_compensation_params={
        'Staff': {'comp_base_salary': 50000, 'comp_increase_per_age_year': 300, 'comp_increase_per_tenure_year': 500, 'comp_log_mean_factor': 1.0, 'comp_spread_sigma': 0.20, 'comp_min_salary': 28000},
        'Manager': {'comp_base_salary': 150000, 'comp_increase_per_age_year': 600, 'comp_increase_per_tenure_year': 1000, 'comp_log_mean_factor': 1.0, 'comp_spread_sigma': 0.25, 'comp_min_salary': 60000},
        'Executive': {'comp_base_salary': 250000, 'comp_increase_per_age_year': 1500, 'comp_increase_per_tenure_year': 3000, 'comp_log_mean_factor': 1.1, 'comp_spread_sigma': 0.35, 'comp_min_salary': 120000}
    },
    # --- Deferral Distribution ---
    deferral_distribution={
        0: 0.10, 1: 0.05, 2: 0.04, 3: 0.40, 4: 0.10,
        5: 0.10, 6: 0.06, 7: 0.03, 8: 0.01, 9: 0.01, 10: 0.10
    },
    # --- Compensation Increase for Survivors (Optional) ---
    annual_comp_increase_mean=0.03, # Mean % increase
    annual_comp_increase_std=0.015, # Std dev of % increase
    # --- Employer Contributions ---
    employer_nec_rate=0.02,
    employer_match_rate=1.00,
    employer_match_cap_deferral_perc=0.03,
    # --- Output Options ---
    output_dir=".", # Directory to save files
    file_prefix="dummy_census_" # Changed prefix
    ):
    """
    Creates configurable dummy historical census CSV files with realistic population flow.
    - Generates initial population for the first year.
    - For subsequent years, simulates terminations, updates survivors, adds new hires.
    - Ensures SSN persistence for non-terminated employees.
    - Calculates derived fields based on IRS limits and plan rules.
    """
    print(f"\n--- Creating {num_years} Dummy Census Files (Simulating Population Flow) ---")
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists
    all_ssns = set()
    generated_files = []
    current_population_df = None
    employee_counter = 0 # For unique IDs in _generate_employee

    # --- IRS Limits (Ensure coverage for simulation range) ---
    IRS_LIMITS = {
        2020: {'comp_limit': 285000, 'deferral_limit': 19500, 'catch_up': 6500},
        2021: {'comp_limit': 290000, 'deferral_limit': 19500, 'catch_up': 6500},
        2022: {'comp_limit': 305000, 'deferral_limit': 20500, 'catch_up': 6500},
        2023: {'comp_limit': 330000, 'deferral_limit': 22500, 'catch_up': 7500},
        2024: {'comp_limit': 345000, 'deferral_limit': 23000, 'catch_up': 7500}
        # Add future/past years manually if needed beyond this range
    }
    # Add a fallback for future years if needed (using latest known)
    latest_known_year = max(IRS_LIMITS.keys())
    for y in range(base_year + 1, base_year + 5): # Add a few future years fallback
         if y not in IRS_LIMITS:
             IRS_LIMITS[y] = IRS_LIMITS[latest_known_year]

    # --- Normalize Distributions ---
    # Role
    total_role_prob = sum(role_distribution.values())
    if not np.isclose(total_role_prob, 1.0):
        print(f"Warning: Role distribution probabilities sum to {total_role_prob:.4f}. Normalizing.")
        factor = 1.0 / total_role_prob
        role_distribution = {k: v * factor for k, v in role_distribution.items()}
    # Deferral
    total_deferral_prob = sum(deferral_distribution.values())
    if not np.isclose(total_deferral_prob, 1.0):
        print(f"Warning: Deferral distribution probabilities sum to {total_deferral_prob:.4f}. Normalizing.")
        factor = 1.0 / total_deferral_prob
        deferral_distribution = {k: v * factor for k, v in deferral_distribution.items()}

    start_year = base_year - num_years + 1

    # --- Generate Population Year by Year ---
    for year in range(start_year, base_year + 1):
        plan_end_date = datetime(year, 12, 31)
        print(f"\nGenerating data for year ending {plan_end_date.date()}...")

        if year == start_year:
            # --- Generate Initial Population ---
            print(f"  Generating initial population of {total_population}...")
            employee_records = []
            for i in range(total_population):
                employee_counter += 1
                # Generate initial employee - is_new_hire=False means tenure is generated
                record = _generate_employee(
                    year, plan_end_date, is_new_hire=False, existing_ssns=all_ssns,
                    min_working_age=min_working_age, max_working_age=max_working_age,
                    age_mean=age_mean, age_std_dev=age_std_dev,
                    tenure_mean_years=tenure_mean_years, tenure_std_dev_years=tenure_std_dev_years,
                    max_tenure_years=max_tenure_years,
                    role_distribution=role_distribution, role_compensation_params=role_compensation_params,
                    deferral_distribution=deferral_distribution, unique_id=employee_counter
                )
                employee_records.append(record)
            current_population_df = pd.DataFrame(employee_records)
            # Initial population has no termination date set yet
            current_population_df['termination_date'] = pd.NaT
            print(f"  Initial population generated ({len(current_population_df)} rows).")

        else: # Subsequent years: Simulate flow from previous year
            previous_population_df = current_population_df.copy()
            num_previous = len(previous_population_df)
            print(f"  Starting with {num_previous} employees from {year-1}.")

            # --- Simulate Terminations ---
            num_to_terminate = int(round(num_previous * termination_rate))
            terminated_records = pd.DataFrame() # Keep terminated records to add back with term date
            survivor_df = previous_population_df.copy()

            if num_to_terminate > 0 and num_previous > 0:
                if num_to_terminate >= num_previous:
                    print(f"  Warning: Termination rate ({termination_rate:.2%}) results in terminating all {num_previous} employees. Proceeding.")
                    indices_to_terminate = previous_population_df.index
                else:
                    print(f"  Simulating {num_to_terminate} terminations (weighted by low tenure/comp)...")
                    # --- Refined Weighted Termination Logic ---
                    eligible_df = previous_population_df.copy()
                    epsilon = 1e-6

                    # Define risk factors
                    LOW_TENURE_THRESHOLD = 2.0
                    LOW_COMP_PERCENTILE = 0.25
                    HIGH_RISK_WEIGHT = 10.0
                    LOW_RISK_WEIGHT = 1.0

                    # Calculate risk indicators
                    # Tenure Risk
                    if 'tenure' in eligible_df.columns and pd.api.types.is_numeric_dtype(eligible_df['tenure']):
                        eligible_df['low_tenure'] = eligible_df['tenure'].fillna(0) < LOW_TENURE_THRESHOLD
                    else:
                        print("  Warning: 'tenure' column not suitable for risk calc. Assuming not low tenure.")
                        eligible_df['low_tenure'] = False

                    # Compensation Risk (within Role if possible)
                    eligible_df['low_comp'] = False # Default
                    comp_col = 'gross_compensation'
                    role_col = 'role' # Assuming 'role' column exists

                    if comp_col in eligible_df.columns and pd.api.types.is_numeric_dtype(eligible_df[comp_col]):
                        if role_col in eligible_df.columns and eligible_df[role_col].nunique() > 1:
                            try:
                                # Calculate percentile within each role
                                eligible_df['comp_percentile_rank'] = eligible_df.groupby(role_col)[comp_col].rank(pct=True, method='average')
                                eligible_df['low_comp'] = eligible_df['comp_percentile_rank'] < LOW_COMP_PERCENTILE
                                print(f"    Calculated compensation risk based on {LOW_COMP_PERCENTILE:.0%} percentile within roles.")
                            except Exception as e:
                                print(f"  Warning: Error calculating comp percentile by role: {e}. Falling back to global percentile.")
                                eligible_df['comp_percentile_rank'] = eligible_df[comp_col].rank(pct=True, method='average')
                                eligible_df['low_comp'] = eligible_df['comp_percentile_rank'] < LOW_COMP_PERCENTILE
                        else:
                            print("    Calculating compensation risk based on global percentile (role column missing or uniform).")
                            eligible_df['comp_percentile_rank'] = eligible_df[comp_col].rank(pct=True, method='average')
                            eligible_df['low_comp'] = eligible_df['comp_percentile_rank'] < LOW_COMP_PERCENTILE
                    else:
                        print("  Warning: 'gross_compensation' column not suitable for risk calc. Assuming not low comp.")
                        # low_comp remains False

                    # Assign Weights based on combined risk
                    eligible_df['is_at_risk'] = eligible_df['low_tenure'] & eligible_df['low_comp']
                    eligible_df['term_weight'] = np.where(eligible_df['is_at_risk'], HIGH_RISK_WEIGHT, LOW_RISK_WEIGHT)

                    at_risk_count = eligible_df['is_at_risk'].sum()
                    print(f"    Identified {at_risk_count} employees as 'at-risk' (low tenure & low comp).")

                    # Calculate Probabilities
                    total_weight = eligible_df['term_weight'].sum()
                    if total_weight <= 0 or not np.isfinite(total_weight):
                        print("  Warning: Total termination weight invalid. Falling back to uniform probability.")
                        probabilities = None
                    else:
                        probabilities = eligible_df['term_weight'] / total_weight
                        if np.any(~np.isfinite(probabilities)) or np.isclose(probabilities.sum(), 0):
                             print("  Warning: Invalid probabilities calculated (NaN/inf/zero sum). Falling back to uniform probability.")
                             probabilities = None
                        elif not np.isclose(probabilities.sum(), 1.0):
                            print(f"  Warning: Probabilities sum to {probabilities.sum():.4f}. Renormalizing.")
                            probabilities = probabilities / probabilities.sum()

                    # Perform Weighted Choice
                    try:
                        if num_to_terminate > len(eligible_df):
                            print(f"  Warning: Trying to terminate {num_to_terminate} but only {len(eligible_df)} eligible. Terminating all.")
                            indices_to_terminate = eligible_df.index
                        else:
                            indices_to_terminate = np.random.choice(
                                eligible_df.index,
                                size=num_to_terminate,
                                replace=False,
                                p=probabilities # Use calculated probabilities or None for uniform
                            )
                    except Exception as e: # Broader exception catch for choice issues
                        print(f"  Error during weighted choice: {e}. Falling back to uniform selection.")
                        indices_to_terminate = np.random.choice(
                            previous_population_df.index, size=min(num_to_terminate, num_previous), replace=False
                        )
                    # --- End Refined Weighted Termination Logic ---

                # Assign random termination date within the *previous* year
                term_year = year - 1
                terminated_records = previous_population_df.loc[indices_to_terminate].copy()
                terminated_records['termination_date'] = [datetime(term_year, np.random.randint(1, 13), np.random.randint(1, 29)) for _ in range(len(terminated_records))] # Simplified date generation

                # Survivors are those not terminated
                survivor_df = previous_population_df.drop(indices_to_terminate)
                print(f"  {len(survivor_df)} survivors remaining.")
            else:
                print("  No terminations simulated (rate=0 or zero population).")


            # --- Update Survivors for the New Year ---
            print(f"  Updating {len(survivor_df)} survivors for {year}...")
            if not survivor_df.empty:
                # Increment age (implicitly handled by recalculating)
                # Increment tenure
                survivor_df['tenure'] = survivor_df['tenure'] + 1
                # Apply random compensation increase
                increase_factor = np.random.normal(loc=1 + annual_comp_increase_mean, scale=annual_comp_increase_std, size=len(survivor_df))
                survivor_df['gross_compensation'] = (survivor_df['gross_compensation'] * increase_factor).round(2)
                # Update age based on current plan end date
                survivor_df['birth_date'] = pd.to_datetime(survivor_df['birth_date'])
                survivor_df['age'] = survivor_df.apply(lambda row: (plan_end_date.year - row['birth_date'].year -
                                     ((plan_end_date.month, plan_end_date.day) <
                                      (row['birth_date'].month, row['birth_date'].day))), axis=1)

                # Optionally, re-randomize deferral rates for survivors? (Keep simple for now)
                # survivor_df['pre_tax_deferral_percentage'] = np.random.choice(list(deferral_distribution.keys()), size=len(survivor_df), p=list(deferral_distribution.values()))

            # --- Add New Hires ---
            num_survivors = len(survivor_df)
            num_new_hires_needed = max(0, total_population - num_survivors)
            print(f"  Adding {num_new_hires_needed} new hires...")
            new_hire_records = []
            if num_new_hires_needed > 0:
                for i in range(num_new_hires_needed):
                    employee_counter += 1
                    record = _generate_employee(
                        year, plan_end_date, is_new_hire=True, existing_ssns=all_ssns, # is_new_hire = True
                        min_working_age=min_working_age, max_working_age=max_working_age,
                        age_mean=age_mean, age_std_dev=age_std_dev, # Use general age profile
                        # Tenure params not used directly for new hires (hire date set in current year)
                        tenure_mean_years=tenure_mean_years, tenure_std_dev_years=tenure_std_dev_years,
                        max_tenure_years=max_tenure_years,
                        role_distribution=role_distribution, role_compensation_params=role_compensation_params,
                        deferral_distribution=deferral_distribution, unique_id=employee_counter
                    )
                    new_hire_records.append(record)

            new_hires_df = pd.DataFrame(new_hire_records)

            # --- Combine Survivors and New Hires for the current year ---
            current_population_list = []
            if not survivor_df.empty:
                 # Drop the calculated 'age' column before concatenating if it exists
                 if 'age' in survivor_df.columns:
                     current_population_list.append(survivor_df.drop(columns=['age']))
                 else:
                     current_population_list.append(survivor_df)

            if not new_hires_df.empty:
                 # Drop the calculated 'age' and 'tenure' columns used internally by _generate_employee
                 cols_to_drop = [col for col in ['age', 'tenure'] if col in new_hires_df.columns]
                 if cols_to_drop:
                     current_population_list.append(new_hires_df.drop(columns=cols_to_drop))
                 else:
                      current_population_list.append(new_hires_df)


            if current_population_list:
                 current_population_df = pd.concat(current_population_list, ignore_index=True)
                 print(f"  Combined population for {year}: {len(current_population_df)} employees.")
            else:
                 current_population_df = pd.DataFrame() # Handle case of zero population
                 print(f"  Warning: Combined population for {year} is empty.")


            # --- Add back terminated employees from previous year with their term date ---
            # These employees *contributed* in the previous year but are not active in the current census
            # For termination modeling, we usually need the census *before* termination.
            # However, if the goal is just to have historical files, we might include them.
            # Let's keep the current_population_df as ONLY the active employees at year-end.
            # The termination model script will compare year N and N+1 to find who is missing.

        # --- Calculate Derived Fields & Save ---
        if not current_population_df.empty:
             # Ensure required date columns are datetime objects before calculation
             current_population_df['birth_date'] = pd.to_datetime(current_population_df['birth_date'])
             current_population_df['hire_date'] = pd.to_datetime(current_population_df['hire_date'])
             current_population_df['termination_date'] = pd.to_datetime(current_population_df['termination_date'], errors='coerce')

             # Recalculate age/tenure precisely based on plan_end_date for final output (if needed)
             # Or rely on the stored values from _generate_employee / survivor update step
             # Let's recalculate derived fields which depend on year-end status

             df_final_year = _calculate_derived_fields(
                 current_population_df, year, IRS_LIMITS,
                 employer_nec_rate, employer_match_rate, employer_match_cap_deferral_perc
             )

             # Format dates for CSV output
             df_final_year['birth_date'] = df_final_year['birth_date'].dt.strftime('%Y-%m-%d')
             df_final_year['hire_date'] = df_final_year['hire_date'].dt.strftime('%Y-%m-%d')
             # Convert NaT to empty string for termination date
             df_final_year['termination_date'] = df_final_year['termination_date'].dt.strftime('%Y-%m-%d').fillna('')


             # Save to CSV
             output_filename = os.path.join(output_dir, f"{file_prefix}{year}.csv")
             df_final_year.to_csv(output_filename, index=False)
             generated_files.append(output_filename)
             print(f"  Saved: {output_filename} ({len(df_final_year)} rows)")
        else:
            print(f"  Skipping save for {year} due to empty population.")


    print("\n--- Dummy file creation complete ---")
    return generated_files

if __name__ == "__main__":
    # Example call demonstrating role-based compensation
    generated_files_custom = create_dummy_census_files(
        num_years=5,
        base_year=2024,
        total_population=5000, # Smaller population for example
        termination_rate=0.12,
        # --- Age Distribution ---
        age_mean=40,
        age_std_dev=10,
        min_working_age=22,
        max_working_age=68,
        # --- Tenure Distribution ---
        tenure_mean_years=6,
        tenure_std_dev_years=4,
        max_tenure_years=35,
        # --- Role Configuration ---
        role_distribution={'Analyst': 0.6, 'Manager': 0.3, 'Director': 0.1},
        role_compensation_params={
            'Analyst': {'comp_base_salary': 60000, 'comp_increase_per_age_year': 350, 'comp_increase_per_tenure_year': 600, 'comp_log_mean_factor': 1.0, 'comp_spread_sigma': 0.18, 'comp_min_salary': 40000},
            'Manager': {'comp_base_salary': 100000, 'comp_increase_per_age_year': 700, 'comp_increase_per_tenure_year': 1200, 'comp_log_mean_factor': 1.05, 'comp_spread_sigma': 0.22, 'comp_min_salary': 75000},
            'Director': {'comp_base_salary': 180000, 'comp_increase_per_age_year': 1800, 'comp_increase_per_tenure_year': 3500, 'comp_log_mean_factor': 1.15, 'comp_spread_sigma': 0.30, 'comp_min_salary': 130000}
        },
        # --- Deferral Distribution ---
        deferral_distribution={
            0: 0.15, 1: 0.05, 2: 0.05, 3: 0.10, 4: 0.10, 5: 0.15,
            6: 0.10, 7: 0.05, 8: 0.05, 9: 0.05, 10: 0.10, 12: 0.03, 15: 0.02
        },
        # --- Compensation Increase ---
        annual_comp_increase_mean=0.035,
        annual_comp_increase_std=0.01,
        # --- Employer Contributions ---
        employer_nec_rate=0.03,
        employer_match_rate=0.50,
        employer_match_cap_deferral_perc=0.06,
        # --- Output Options ---
        output_dir="./termination_model_example", # Separate dir for example
        file_prefix="example_census_"
    )
    print("\nGenerated example files:")
    for f in generated_files_custom:
        print(f" - {f}")
