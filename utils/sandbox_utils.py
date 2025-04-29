"""
Utility functions for data calculations.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import truncnorm
import uuid

def calculate_age(birth_date_series, reference_date):
    """Calculates age as of a reference date, handling Series input vectorially."""
    # Ensure input is datetime, coercing errors to NaT
    birth_date_series = pd.to_datetime(birth_date_series, errors='coerce')
    reference_date = pd.to_datetime(reference_date) # reference_date is a single Timestamp

    # Initialize age series with NaNs
    age = pd.Series(np.nan, index=birth_date_series.index)
    valid_dates = birth_date_series.notna()

    # Calculate initial age difference for valid dates
    age.loc[valid_dates] = reference_date.year - birth_date_series[valid_dates].dt.year

    # Adjust age if birthday hasn't occurred yet this year (vectorized)
    # Check if reference month/day is before birth month/day
    birthday_not_passed = (
        (reference_date.month < birth_date_series[valid_dates].dt.month) |
        ((reference_date.month == birth_date_series[valid_dates].dt.month) & 
         (reference_date.day < birth_date_series[valid_dates].dt.day))
    )
    age.loc[valid_dates & birthday_not_passed] -= 1

    return age

def calculate_tenure(hire_date_series, reference_date, termination_date_series=None):
    """Calculates tenure in years as of a reference date, handling Series input vectorially."""
    # Ensure input is datetime, coercing errors to NaT
    hire_date_series = pd.to_datetime(hire_date_series, errors='coerce')
    reference_date = pd.to_datetime(reference_date) # reference_date is a single Timestamp
    if termination_date_series is not None:
        termination_date_series = pd.to_datetime(termination_date_series, errors='coerce')

    # Initialize tenure series with NaNs
    tenure = pd.Series(np.nan, index=hire_date_series.index)
    valid_dates = hire_date_series.notna()

    # Calculate tenure in days for valid dates, then convert to years
    if valid_dates.any(): # Check if there are any valid dates to calculate
        end_date = reference_date
        if termination_date_series is not None:
            end_date = pd.to_datetime(termination_date_series.where(termination_date_series.notna(), reference_date))
        tenure.loc[valid_dates] = (end_date - hire_date_series[valid_dates]).dt.days / 365.25

    return tenure

def generate_new_ssn(existing_ssns, num_new):
    """Generates unique placeholder SSNs."""
    new_ssns = []
    max_existing_numeric = 0
    try:
        numeric_parts = pd.to_numeric(existing_ssns.astype(str).str.extract(r'(\d+)$', expand=False), errors='coerce')
        if numeric_parts.notna().any(): max_existing_numeric = int(numeric_parts.max())
        else: max_existing_numeric = 900000000 + len(existing_ssns)
    except Exception: max_existing_numeric = 900000000 + len(existing_ssns)

    current_max = max_existing_numeric
    temp_new_ssns = set()
    while len(new_ssns) < num_new:
        current_max += 1
        potential_ssn = f"NEW_{current_max}"
        if potential_ssn not in existing_ssns and potential_ssn not in temp_new_ssns:
             new_ssns.append(potential_ssn)
             temp_new_ssns.add(potential_ssn)
             if current_max > max_existing_numeric + num_new * 100: # Safety break
                  print("Warning: Potential issue generating unique SSNs. Using fallback.")
                  break
    needed = num_new - len(new_ssns)
    if needed > 0:
        print(f"Using fallback SSN generation for {needed} hires.")
        base_ts = pd.Timestamp.now(tz='America/New_York').value
        for i in range(needed):
           fallback_ssn = f"FALLBACK_{base_ts}_{np.random.randint(10000,99999)}_{i}"
           new_ssns.append(fallback_ssn)
    return new_ssns

# --- Constants for New Hire Generation (if not in config) ---
DEFAULT_AGE_MEAN = 35
DEFAULT_AGE_STD_DEV = 10
DEFAULT_MIN_WORKING_AGE = 22
DEFAULT_MAX_WORKING_AGE = 65
DEFAULT_COMP_BASE_SALARY = 50000
DEFAULT_COMP_INCREASE_PER_AGE_YEAR = 500
DEFAULT_COMP_INCREASE_PER_TENURE_YEAR = 1000 # Less relevant for new hires
DEFAULT_COMP_LOG_MEAN_FACTOR = 1.0
DEFAULT_COMP_SPREAD_SIGMA = 0.3
DEFAULT_COMP_MIN_SALARY = 30000

def generate_new_hires(
    num_hires,
    hire_year,
    role_distribution,
    role_compensation_params,
    age_mean,
    age_std_dev,
    min_working_age,
    max_working_age,
    scenario_config # Keep for potential future flexibility, though specific params are passed
    ):
    """Generates a DataFrame of new hire records."""
    if num_hires <= 0:
        return pd.DataFrame() # Return empty DataFrame if no hires needed

    new_hires_list = []
    # Generate random hire dates uniformly throughout the hire year
    year_start = pd.Timestamp(f"{hire_year}-01-01")
    year_end = pd.Timestamp(f"{hire_year}-12-31")
    days_in_year = (year_end - year_start).days + 1
    random_day_offsets = np.random.randint(0, days_in_year, size=num_hires)

    # --- Generate New Hire Details --- 
    roles = list(role_distribution.keys()) if role_distribution else ['Default']
    role_probs = list(role_distribution.values()) if role_distribution else [1.0]
    
    # Normalize probabilities if needed
    if role_distribution:
        total_prob = sum(role_probs)
        if not np.isclose(total_prob, 1.0):
            print(f"Warning: Role distribution probabilities sum to {total_prob:.4f}. Normalizing.")
            role_probs = [p / total_prob for p in role_probs]

    # Pre-generate random data where possible for efficiency
    assigned_roles = np.random.choice(roles, size=num_hires, p=role_probs)
    
    # Generate Ages using truncated normal distribution
    safe_age_std_dev = max(age_std_dev, 1e-6) # Avoid division by zero
    age_a, age_b = (min_working_age - age_mean) / safe_age_std_dev, (max_working_age - age_mean) / safe_age_std_dev
    if age_std_dev <= 0:
        ages = np.full(num_hires, age_mean)
    else:
        ages = truncnorm.rvs(age_a, age_b, loc=age_mean, scale=safe_age_std_dev, size=num_hires)
    ages = np.clip(ages, min_working_age, max_working_age).astype(int)

    # Generate unique SSNs (ensure they don't clash with potential existing ones - tricky without full list)
    # Simple UUID approach for now, replace with safer method if SSN format matters
    new_ssns = [str(uuid.uuid4()) for _ in range(num_hires)]

    for i in range(num_hires):
        record = {}
        record['employee_ssn'] = new_ssns[i]
        record['employee_hire_date'] = year_start + pd.Timedelta(days=int(random_day_offsets[i]))
        record['employee_termination_date'] = pd.NaT
        record['employment_status'] = 'Active'

        # Assign Role
        assigned_role = assigned_roles[i]
        record['employee_role'] = assigned_role

        # Generate Age/Birth Date
        age = ages[i]
        birth_year = record['employee_hire_date'].year - age
        # Simple random birth month/day
        birth_month = np.random.randint(1, 13)
        birth_day = np.random.randint(1, 29) # Avoid Feb 29 issues simply
        try:
            record['employee_birth_date'] = pd.Timestamp(f"{birth_year}-{birth_month}-{birth_day}")
        except ValueError:
            record['employee_birth_date'] = pd.Timestamp(f"{birth_year}-01-01") # Fallback

        # Calculate Compensation
        tenure_years = 0 # New hire
        age_experience_years = age - min_working_age

        # Get role-specific comp params or use defaults
        role_params = {}
        if role_compensation_params and assigned_role in role_compensation_params:
            role_params = role_compensation_params[assigned_role]
        else: # Use defaults if role/params missing
            role_params = {
                'comp_base_salary': scenario_config.get('comp_base_salary', DEFAULT_COMP_BASE_SALARY),
                'comp_increase_per_age_year': scenario_config.get('comp_increase_per_age_year', DEFAULT_COMP_INCREASE_PER_AGE_YEAR),
                'comp_increase_per_tenure_year': scenario_config.get('comp_increase_per_tenure_year', DEFAULT_COMP_INCREASE_PER_TENURE_YEAR),
                'comp_log_mean_factor': scenario_config.get('comp_log_mean_factor', DEFAULT_COMP_LOG_MEAN_FACTOR),
                'comp_spread_sigma': scenario_config.get('comp_spread_sigma', DEFAULT_COMP_SPREAD_SIGMA),
                'comp_min_salary': scenario_config.get('comp_min_salary', DEFAULT_COMP_MIN_SALARY)
            }

        target_comp = (role_params['comp_base_salary'] +
                       (age_experience_years * role_params['comp_increase_per_age_year']) +
                       (tenure_years * role_params['comp_increase_per_tenure_year']))
        log_mean = np.log(max(1000, target_comp * role_params['comp_log_mean_factor'])) # Avoid log(0)
        comp = np.random.lognormal(mean=log_mean, sigma=role_params['comp_spread_sigma'])
        record['employee_gross_compensation'] = round(max(role_params['comp_min_salary'], comp), 2)

        # Initialize plan-related fields (will be updated by rule engine)
        record['is_eligible'] = False
        record['is_participating'] = False
        record['employee_deferral_rate'] = 0.0 # Assume 0% initially
        record['employee_pre_tax_contribution'] = 0.0
        record['employer_non_elective_contribution'] = 0.0
        record['employer_match_contribution'] = 0.0
        record['eligibility_entry_date'] = pd.NaT
        record['employee_plan_year_compensation'] = 0.0 # Will be calculated by plan engine
        record['employee_capped_compensation'] = 0.0 # Will be calculated by plan engine

        # Add any other columns present in the main projection DF, initialized appropriately
        record['gender'] = np.random.choice(['Male', 'Female']) # Simple random assignment
        record['yos'] = 0 # Years of Service starts at 0

        new_hires_list.append(record)

    new_hires_df = pd.DataFrame(new_hires_list)
    return new_hires_df
