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
    output_dir="data/test_census/", # Directory to save files
    file_prefix="dummy_census_" # Changed prefix
):
    print(f"\n--- Creating {num_years} Dummy Census Files (Simulating Population Flow) ---")
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists
    print(f"  Output directory: {os.path.abspath(output_dir)}")
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
            # Save the initial population to CSV
            output_path = os.path.join(output_dir, f"{file_prefix}{year}.csv")
            current_population_df.to_csv(output_path, index=False)
            print(f"  Saved: {output_path}")

        else: # Subsequent years: Simulate flow from previous year
            previous_population_df = current_population_df.copy()
            num_previous = len(previous_population_df)
            print(f"  Starting with {num_previous} employees from {year-1}.")
            # Save the updated population to CSV
            output_path = os.path.join(output_dir, f"{file_prefix}{year}.csv")
            current_population_df.to_csv(output_path, index=False)
            print(f"  Saved: {output_path}")

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

if __name__ == "__main__":
    create_dummy_census_files()
