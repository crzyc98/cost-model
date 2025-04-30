"""
utils/sandbox_utils.py - Utility functions for data calculations.
"""
import logging
import pandas as pd
import numpy as np
from scipy.stats import truncnorm
import uuid
from utils.date_utils import calculate_age, calculate_tenure
from typing import Sequence, Optional

logger = logging.getLogger(__name__)

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
                  logger.warning("Potential issue generating unique SSNs; using fallback.")
                  break
    needed = num_new - len(new_ssns)
    if needed > 0:
        logger.warning("Using fallback SSN generation for %d hires.", needed)
        base_ts = pd.Timestamp.now(tz='America/New_York').value
        for i in range(needed):
           fallback_ssn = f"FALLBACK_{base_ts}_{np.random.randint(10000,99999)}_{i}"
           new_ssns.append(fallback_ssn)
    return new_ssns

def generate_new_hires(
    num_hires: int,
    hire_year: int,
    role_distribution: dict,
    role_compensation_params: dict,
    age_mean: float,
    age_std_dev: float,
    min_working_age: int,
    max_working_age: int,
    scenario_config: dict,
    existing_ssns: Sequence[str] = (),
    rng: Optional[np.random.Generator] = None
) -> pd.DataFrame:
    # Prepare empty schema if no hires
    cols = [
        'employee_ssn','employee_hire_date','employee_termination_date',
        'employment_status','employee_role','employee_birth_date',
        'employee_gross_compensation','is_eligible','is_participating',
        'employee_deferral_rate','employee_pre_tax_contribution',
        'employer_non_elective_contribution','employer_match_contribution',
        'eligibility_entry_date'
    ]
    if num_hires <= 0:
        return pd.DataFrame(columns=cols)

    # Seedable RNG
    if rng is None:
        rng = np.random.default_rng()
    new_hires_list = []

    # Generate random hire dates uniformly throughout the hire year
    year_start = pd.Timestamp(f"{hire_year}-01-01")
    year_end = pd.Timestamp(f"{hire_year}-12-31")
    days_in_year = (year_end - year_start).days + 1
    random_day_offsets = rng.integers(0, days_in_year, size=num_hires)

    # --- Generate New Hire Details --- 
    roles = list(role_distribution.keys()) if role_distribution else ['Default']
    role_probs = list(role_distribution.values()) if role_distribution else [1.0]
    
    # Normalize probabilities
    if role_distribution:
        total_prob = sum(role_probs)
        if not np.isclose(total_prob, 1.0):
            logger.warning("Role distribution sums to %.4f; normalizing.", total_prob)
            role_probs = [p / total_prob for p in role_probs]

    # Pre-generate random data where possible for efficiency
    assigned_roles = rng.choice(roles, size=num_hires, p=role_probs)
    
    # Generate Ages using truncated normal distribution
    safe_age_std_dev = max(age_std_dev, 1e-6) # Avoid division by zero
    age_a, age_b = (min_working_age - age_mean) / safe_age_std_dev, (max_working_age - age_mean) / safe_age_std_dev
    ages = (
        np.full(num_hires, age_mean)
        if age_std_dev <= 0 else
        truncnorm.rvs(age_a, age_b, loc=age_mean, scale=safe_age_std_dev, size=num_hires, random_state=rng)
    )
    ages = np.clip(ages, min_working_age, max_working_age).astype(int)

    # Generate unique SSNs (ensure they don't clash with potential existing ones - tricky without full list)
    new_ssns = generate_new_ssn(existing_ssns, num_hires)

    # Build records
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
        birth_month = rng.integers(1, 13)
        birth_day = rng.integers(1, 29) # Avoid Feb 29 issues simply
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
                'comp_base_salary': scenario_config.get('comp_base_salary', 50000),
                'comp_increase_per_age_year': scenario_config.get('comp_increase_per_age_year', 500),
                'comp_increase_per_tenure_year': scenario_config.get('comp_increase_per_tenure_year', 1000),
                'comp_log_mean_factor': scenario_config.get('comp_log_mean_factor', 1.0),
                'comp_spread_sigma': scenario_config.get('comp_spread_sigma', 0.3),
                'comp_min_salary': scenario_config.get('comp_min_salary', 30000)
            }

        target_comp = (role_params['comp_base_salary'] +
                       (age_experience_years * role_params['comp_increase_per_age_year']) +
                       (tenure_years * role_params['comp_increase_per_tenure_year']))
        log_mean = np.log(max(1000, target_comp * role_params['comp_log_mean_factor'])) # Avoid log(0)
        comp = rng.lognormal(mean=log_mean, sigma=role_params['comp_spread_sigma'])
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
        record['gender'] = rng.choice(['Male', 'Female']) # Simple random assignment
        record['yos'] = 0 # Years of Service starts at 0

        new_hires_list.append(record)

    return pd.DataFrame(new_hires_list, columns=cols)
