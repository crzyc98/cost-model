# utils/data_generation_utils.py - Utility functions for generating synthetic HR data.

import logging
import pandas as pd
import numpy as np
from scipy.stats import truncnorm
from .date_utils import calculate_age # Assuming date_utils is in the same directory
from typing import Sequence, Optional, List, Dict, Any

logger = logging.getLogger(__name__)

def _generate_sequential_ids(existing_ids: Optional[Sequence[str]], num_new: int) -> List[str]:
    """Generates unique sequential placeholder IDs prefixed with 'NEW_'.

    Handles potential numeric suffixes in existing IDs to continue sequence,
    with fallbacks for non-numeric or complex existing IDs.
    """
    new_ids = []
    temp_new_ids = set() # Use set for efficient checking of newly generated IDs

    current_max = 0
    existing_ids_set = set(existing_ids) if existing_ids else set()

    if existing_ids_set:
        try:
            # Extract numeric parts only if they are strings and match pattern 'prefix_number'
            numeric_parts = pd.to_numeric(
                pd.Series([s for s in existing_ids_set if isinstance(s, str)])
                  .str.extract(r'_(\d+)$', expand=False), # Look for underscore prefix
                errors='coerce'
            )
            if numeric_parts.notna().any():
                current_max = int(numeric_parts.max())
            else:
                # Fallback if no numeric parts found or all NaN
                current_max = 900000000 + len(existing_ids_set)
        except Exception as e:
            logger.warning(f"Error extracting numeric part of existing IDs: {e}. Using fallback sequence start.")
            current_max = 900000000 + len(existing_ids_set)
    else:
        current_max = 900000000 # Starting point if no existing IDs

    loop_count = 0
    max_loops = num_new * 100 + 100 # Safety break limit

    while len(new_ids) < num_new and loop_count < max_loops:
        current_max += 1
        potential_id = f"NEW_{current_max}"
        if potential_id not in existing_ids_set and potential_id not in temp_new_ids:
            new_ids.append(potential_id)
            temp_new_ids.add(potential_id)
        loop_count += 1

    if len(new_ids) < num_new:
        logger.warning("Potential issue generating unique sequential IDs or reached loop limit; using fallback for remaining.")
        needed = num_new - len(new_ids)
        base_ts = pd.Timestamp.now().value
        rng_fallback = np.random.default_rng() # Local RNG for fallback
        for i in range(needed):
            while True:
                fallback_id = f"FALLBACK_{base_ts}_{rng_fallback.integers(10000,99999)}_{i}"
                if fallback_id not in existing_ids_set and fallback_id not in temp_new_ids:
                    new_ids.append(fallback_id)
                    temp_new_ids.add(fallback_id)
                    break
    return new_ids

def _generate_hire_dates(num: int, year: int, rng: np.random.Generator) -> pd.Series:
    """Generates random hire dates within a given year."""
    year_start = pd.Timestamp(f"{year}-01-01")
    year_end = pd.Timestamp(f"{year}-12-31")
    days_in_year = (year_end - year_start).days + 1
    random_day_offsets = rng.integers(0, days_in_year, size=num)
    return year_start + pd.to_timedelta(random_day_offsets, unit='D')

def _assign_roles(num: int, dist: Dict[str, float], rng: np.random.Generator) -> np.ndarray:
    """Assigns roles based on a probability distribution."""
    roles = list(dist.keys()) if dist else ['Default']
    role_probs = list(dist.values()) if dist else [1.0]

    if dist:
        total_prob = sum(role_probs)
        if not np.isclose(total_prob, 1.0) and total_prob > 0:
            logger.warning("Role distribution sums to %.4f; normalizing.", total_prob)
            role_probs = [p / total_prob for p in role_probs]
        elif total_prob <= 0:
            logger.error("Role distribution probabilities sum to zero or less. Using uniform distribution.")
            role_probs = [1.0 / len(roles)] * len(roles)

    return rng.choice(roles, size=num, p=role_probs)

def _generate_ages(num: int, mean: float, std_dev: float, min_age: int, max_age: int, rng: np.random.Generator) -> np.ndarray:
    """Generates ages using a truncated normal distribution."""
    safe_std_dev = max(std_dev, 1e-6) # Avoid division by zero
    a, b = (min_age - mean) / safe_std_dev, (max_age - mean) / safe_std_dev
    ages = truncnorm.rvs(a, b, loc=mean, scale=safe_std_dev, size=num, random_state=rng)
    return np.clip(ages, min_age, max_age).astype(int)

def _calculate_birth_dates(hire_dates: pd.Series, ages: np.ndarray, rng: np.random.Generator) -> pd.Series:
    """Calculates birth dates based on hire date and age."""
    birth_dates = []
    for hire_date, age in zip(hire_dates, ages):
        birth_year = hire_date.year - age
        birth_month = rng.integers(1, 13)
        # Ensure day is valid for month/year (simple approach: cap at 28)
        birth_day = rng.integers(1, 29)
        try:
            birth_dates.append(pd.Timestamp(year=birth_year, month=birth_month, day=birth_day))
        except ValueError:
            logger.warning(f"Could not create birth date {birth_year}-{birth_month}-{birth_day}, using Jan 1st.")
            birth_dates.append(pd.Timestamp(year=birth_year, month=1, day=1))
    # Check length consistency before creating Series
    if len(hire_dates) != len(birth_dates):
        logger.error(f"Length mismatch after loop: hire_dates ({len(hire_dates)}) vs birth_dates ({len(birth_dates)}).")
        return pd.Series(dtype='datetime64[ns]')
    # Create Series with default RangeIndex (0-based) and name
    return pd.Series(birth_dates, name='employee_birth_date')

def _calculate_compensation(roles: np.ndarray, ages: np.ndarray, comp_config: Dict[str, Any], defaults: Dict[str, Any], min_working_age: int, rng: np.random.Generator) -> pd.Series:
    """Calculates gross compensation based on role, age, and config parameters."""
    compensation_list = []
    for role, age in zip(roles, ages):
        role_params = comp_config.get(role, {}).copy() # Get role specific, copy to modify
        # Apply defaults for any missing parameters
        for key, default_val in defaults.items():
            role_params.setdefault(key, default_val)

        age_experience_years = max(0, age - min_working_age)
        tenure_years = 0 # New hire

        target_comp = (role_params['comp_base_salary'] +
                       (age_experience_years * role_params['comp_increase_per_age_year']) +
                       (tenure_years * role_params['comp_increase_per_tenure_year']))

        # Use lognormal distribution for compensation spread
        comp_spread_sigma = max(1e-6, role_params['comp_spread_sigma'])
        base_for_log = max(1000, target_comp * role_params['comp_log_mean_factor'])
        log_mean = np.log(base_for_log)

        comp = rng.lognormal(mean=log_mean, sigma=comp_spread_sigma)
        final_comp = round(max(role_params['comp_min_salary'], comp), 2)
        compensation_list.append(final_comp)

    return pd.Series(compensation_list, index=pd.RangeIndex(len(roles)))

def generate_new_hires(
    num_hires: int,
    hire_year: int,
    scenario_config: Dict[str, Any],
    existing_ids: Optional[Sequence[str]] = None,
    rng: Optional[np.random.Generator] = None
) -> pd.DataFrame:
    """Generates a DataFrame of new hires with core HR data.

    Args:
        num_hires: Number of new hires to generate.
        hire_year: The year in which the hires occur.
        scenario_config: Dictionary containing configuration, expected keys include:
            'role_distribution': Dict[str, float] for role probabilities.
            'role_compensation_params': Dict[str, Dict[str, float]] for role-specific comp.
            'new_hire_average_age': float
            'new_hire_age_std_dev': float
            'min_working_age': int
            'max_working_age': int
            'comp_base_salary': float (default)
            'comp_increase_per_age_year': float (default)
            'comp_increase_per_tenure_year': float (default)
            'comp_log_mean_factor': float (default)
            'comp_spread_sigma': float (default)
            'comp_min_salary': float (default)
            # Optional: 'gender_distribution': Dict[str, float]
        existing_ids: A sequence of existing employee IDs to ensure uniqueness.
        rng: Optional numpy random Generator for reproducibility.

    Returns:
        A pandas DataFrame containing the generated new hire data.
        Columns include: 'employee_id', 'employee_hire_date', 'employee_birth_date',
        'employee_role', 'employee_gross_compensation', 'employment_status',
        'employee_termination_date', 'gender' (if configured).
        Plan-specific columns (eligibility, participation, etc.) are NOT initialized here.
    """
    core_hr_cols = [
        'employee_id', 'employee_hire_date', 'employee_birth_date',
        'employee_role', 'employee_gross_compensation',
        'employment_status', 'employee_termination_date'
    ]

    if num_hires <= 0:
        logger.info("generate_new_hires called with num_hires=0. Returning empty DataFrame.")
        return pd.DataFrame(columns=core_hr_cols)

    logger.debug(f"Generating {num_hires} new hires for year {hire_year}.")

    if rng is None:
        rng = np.random.default_rng()

    # --- Safely Extract Config Parameters ---
    role_dist = scenario_config.get('role_distribution', {})
    role_comp_params = scenario_config.get('role_compensation_params', {})
    age_mean = float(scenario_config.get('new_hire_average_age', 30.0))
    age_std_dev = float(scenario_config.get('new_hire_age_std_dev', 5.0))
    min_age = int(scenario_config.get('min_working_age', 18))
    max_age = int(scenario_config.get('max_working_age', 65))
    gender_dist = scenario_config.get('gender_distribution', None)

    # Default compensation parameters (used if not specified per role)
    default_comp_params = {
        'comp_base_salary': scenario_config.get('comp_base_salary', 50000),
        'comp_increase_per_age_year': scenario_config.get('comp_increase_per_age_year', 500),
        'comp_increase_per_tenure_year': scenario_config.get('comp_increase_per_tenure_year', 1000),
        'comp_log_mean_factor': scenario_config.get('comp_log_mean_factor', 1.0),
        'comp_spread_sigma': scenario_config.get('comp_spread_sigma', 0.3),
        'comp_min_salary': scenario_config.get('comp_min_salary', 30000)
    }

    # --- Generate Data using Helper Functions ---
    ids = _generate_sequential_ids(existing_ids, num_hires)
    hire_dates = _generate_hire_dates(num_hires, hire_year, rng)
    roles = _assign_roles(num_hires, role_dist, rng)
    ages = _generate_ages(num_hires, age_mean, age_std_dev, min_age, max_age, rng)
    birth_dates = _calculate_birth_dates(hire_dates, ages, rng)
    compensation = _calculate_compensation(roles, ages, role_comp_params, default_comp_params, min_age, rng)

    # --- Assemble Core HR DataFrame ---
    nh_df = pd.DataFrame({
        'employee_id': ids,
        'employee_hire_date': hire_dates,
        'employee_birth_date': birth_dates,
        'employee_role': roles,
        'employee_gross_compensation': compensation,
        'employment_status': 'Active',
        'employee_termination_date': pd.NaT,
    })

    # --- Optional: Add Gender ---
    if gender_dist and isinstance(gender_dist, dict):
        genders = list(gender_dist.keys())
        gender_probs = list(gender_dist.values())
        total_gender_prob = sum(gender_probs)
        if not np.isclose(total_gender_prob, 1.0) and total_gender_prob > 0:
             logger.warning("Gender distribution sums to %.4f; normalizing.", total_gender_prob)
             gender_probs = [p / total_gender_prob for p in gender_probs]
        elif total_gender_prob <= 0:
             logger.warning("Gender distribution invalid. Skipping gender assignment.")
        else:
             nh_df['gender'] = rng.choice(genders, size=num_hires, p=gender_probs)
             core_hr_cols.append('gender') # Add to list if generated
    else:
        logger.debug("No valid gender distribution found in config. Skipping gender assignment.")

    # Reorder columns to a consistent standard if needed
    # Ensure all core columns exist, even if gender wasn't added
    final_cols = [col for col in core_hr_cols if col in nh_df.columns]
    nh_df = nh_df[final_cols]

    logger.info(f"Generated DataFrame for {len(nh_df)} new hires with columns: {list(nh_df.columns)}.")
    return nh_df
