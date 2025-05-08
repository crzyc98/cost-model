# cost_model/dynamics/hiring.py
import pandas as pd
import numpy as np
from typing import Sequence, Optional, List, Dict, Any, Union # Added Union
# from cost_model.utils.date_utils import calculate_age, calculate_tenure # Not directly used
from cost_model.utils.id_generation import _generate_sequential_ids
import logging
from scipy.stats import truncnorm
from cost_model.utils.columns import (
    EMP_HIRE_DATE, EMP_BIRTH_DATE, EMP_GROSS_COMP, EMP_TERM_DATE
)
# If CompensationParams is a defined type, you might import it for type hinting, e.g.:
# from ..config.models import CompensationParams # Hypothetical

logger = logging.getLogger(__name__)

# _generate_hire_dates, _assign_roles, _generate_ages, _calculate_birth_dates
# remain unchanged from your version. I'll paste them for completeness if you wish,
# but the key change is in _calculate_compensation and how it's called.

def _generate_hire_dates(num: int, year: int, rng: np.random.Generator) -> pd.Series:
    """Generates random hire dates within a given year."""
    year_start = pd.Timestamp(f"{year}-01-01")
    year_end = pd.Timestamp(f"{year}-12-31")
    days_in_year = int((year_end - year_start).days + 1)
    random_day_offsets = rng.integers(0, days_in_year, size=num)
    return year_start + pd.to_timedelta(random_day_offsets, unit='D')

def _assign_roles(num: int, dist: Dict[str, float], rng: np.random.Generator) -> np.ndarray:
    """Assigns roles based on a probability distribution."""
    roles = list(dist.keys()) if dist else ['DefaultRole']
    role_probs = list(dist.values()) if dist else [1.0]
    if not roles:
        logger.error("No roles available for assignment. Assigning 'UnknownRole'.")
        return np.array(['UnknownRole'] * num)
    if dist:
        total_prob = sum(role_probs)
        if not np.isclose(total_prob, 1.0) and total_prob > 0:
            logger.warning(f"Role distribution sums to {total_prob:.4f}; normalizing.")
            role_probs = [p / total_prob for p in role_probs]
        elif total_prob <= 0:
            logger.error("Role distribution probabilities sum to zero or less. Using uniform distribution.")
            role_probs = [1.0 / len(roles)] * len(roles)
    return rng.choice(roles, size=num, p=role_probs)

def _generate_ages(num: int, mean: float, std_dev: float, min_age: int, max_age: int, rng: np.random.Generator) -> np.ndarray:
    """Generates ages using a truncated normal distribution."""
    safe_std_dev = max(std_dev, 1e-6)
    a, b = (min_age - mean) / safe_std_dev, (max_age - mean) / safe_std_dev
    ages = truncnorm.rvs(a, b, loc=mean, scale=safe_std_dev, size=num, random_state=rng)
    return np.round(np.clip(ages, min_age, max_age)).astype(int)

def _calculate_birth_dates(hire_dates: pd.Series, ages: np.ndarray, rng: np.random.Generator) -> pd.Series:
    """Calculates birth dates based on hire date and age."""
    birth_dates_list = []
    for hire_date, age in zip(hire_dates, ages):
        birth_year = hire_date.year - age
        birth_month = rng.integers(1, 13)
        birth_day = rng.integers(1, 29)
        try:
            birth_dates_list.append(pd.Timestamp(year=birth_year, month=birth_month, day=birth_day))
        except ValueError:
            logger.warning(f"Could not create birth date {birth_year}-{birth_month}-{birth_day}, using Jan 1st.")
            birth_dates_list.append(pd.Timestamp(year=birth_year, month=1, day=1))
    return pd.Series(birth_dates_list, name=EMP_BIRTH_DATE)


# --- MODIFIED FUNCTION ---
def _calculate_compensation(
    roles: Sequence[str],
    ages: np.ndarray,
    role_comp_params: Dict[str, Any], # Values can be CompensationParams objects or dicts
    default_comp_params_as_dict: Dict[str, float], # This is explicitly a dict
    min_age_config: int,
    rng: np.random.Generator
) -> pd.Series:
    """Calculates initial compensation based on role, age, and config parameters."""
    compensation_list = []
    for role, age in zip(roles, ages):
        # current_params can be a CompensationParams object (or other object type)
        # OR it can be the default_comp_params_as_dict.
        current_params = role_comp_params.get(role, default_comp_params_as_dict)

        # Helper to get values correctly based on type of current_params
        def get_param_value(param_source: Union[Dict[str, float], object], key: str, default_val: float) -> float:
            if isinstance(param_source, dict):
                return float(param_source.get(key, default_val))
            else: # Assume it's an object with attributes (like CompensationParams)
                return float(getattr(param_source, key, default_val))

        base = get_param_value(current_params, 'comp_base_salary', 50000.0)
        age_factor = get_param_value(current_params, 'comp_age_factor', 0.01)
        min_salary = get_param_value(current_params, 'comp_min_salary', 40000.0)
        stochastic_std_dev = get_param_value(current_params, 'comp_stochastic_std_dev', 0.0)
        
        age_comp = base * (age - min_age_config) * age_factor
        initial_comp = max(base + age_comp, min_salary)

        if stochastic_std_dev > 0:
            if initial_comp <= 0:
                logger.warning(f"Initial comp for role '{role}' age {age} is {initial_comp:.2f}. Using min_salary ({min_salary}) for log-normal base.")
                initial_comp_for_log = max(min_salary, 1.0) # Ensure positive for np.log
            else:
                initial_comp_for_log = initial_comp
            
            log_mean = np.log(initial_comp_for_log) - (stochastic_std_dev**2) / 2
            stochastic_comp = np.exp(rng.normal(log_mean, stochastic_std_dev))
            compensation_list.append(stochastic_comp)
        else:
            compensation_list.append(initial_comp)

    avg_comp = np.mean(compensation_list) if compensation_list else 0
    logger.debug(f"Calculated initial compensations for {len(roles)} hires. Avg: {avg_comp:.0f}")
    return pd.Series(compensation_list, name=EMP_GROSS_COMP)
# --- END OF MODIFIED FUNCTION ---


def generate_new_hires(
    num_hires: int,
    hire_year: int,
    scenario_config: Dict[str, Any],
    existing_ids: Optional[Sequence[str]] = None,
    rng: Optional[np.random.Generator] = None,
    id_col_name: str = 'employee_id'
) -> pd.DataFrame:
    core_hr_cols_definition = [
        id_col_name, EMP_HIRE_DATE, EMP_BIRTH_DATE,
        'employee_role', EMP_GROSS_COMP,
        'employment_status', EMP_TERM_DATE
    ]

    if num_hires <= 0:
        logger.info("generate_new_hires called with num_hires=0. Returning empty DataFrame.")
        return pd.DataFrame(columns=core_hr_cols_definition)

    logger.debug(f"Generating {num_hires} new hires for year {hire_year}.")

    if rng is None:
        rng = np.random.default_rng()

    def get_config_val(cfg: Union[Dict[str, Any], object], key: str, default: Any) -> Any:
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    role_comp_params = get_config_val(scenario_config, 'role_compensation_params', {})
    age_mean = float(get_config_val(scenario_config, 'new_hire_average_age', 30.0))
    age_std_dev = float(get_config_val(scenario_config, 'new_hire_age_std_dev', 5.0))
    min_age_cfg = int(get_config_val(scenario_config, 'min_working_age', 18))
    max_age_cfg = int(get_config_val(scenario_config, 'max_working_age', 65))
    gender_dist = get_config_val(scenario_config, 'gender_distribution', None)
    role_dist = get_config_val(scenario_config, 'role_distribution', {})


    # --- Definition of default_comp_params_as_dict ---
    # This is the dictionary that will be passed as default to _calculate_compensation
    default_comp_params_as_dict = {
        'comp_base_salary': get_config_val(scenario_config, 'comp_base_salary', 50000.0),
        'comp_age_factor': get_config_val(scenario_config, 'comp_age_factor', 0.01),
        'comp_stochastic_std_dev': get_config_val(scenario_config, 'comp_stochastic_std_dev', 0.0),
        'comp_min_salary': get_config_val(scenario_config, 'comp_min_salary', 30000.0)
        # Add other default comp keys here if _calculate_compensation needs them via get_param_value
    }
    # --- End of definition ---

    ids = _generate_sequential_ids(existing_ids if existing_ids is not None else [], num_hires)
    hire_dates = _generate_hire_dates(num_hires, hire_year, rng)
    roles = _assign_roles(num_hires, role_dist, rng)
    ages = _generate_ages(num_hires, age_mean, age_std_dev, min_age_cfg, max_age_cfg, rng)
    birth_dates = _calculate_birth_dates(hire_dates, ages, rng)
    
    # Call _calculate_compensation with the correctly named default dict
    compensation = _calculate_compensation(
        roles, ages, role_comp_params, default_comp_params_as_dict, min_age_cfg, rng
    )

    nh_df_data = {
        id_col_name: ids,
        EMP_HIRE_DATE: hire_dates,
        EMP_BIRTH_DATE: birth_dates,
        'employee_role': roles,
        EMP_GROSS_COMP: compensation,
        'employment_status': 'Active',
        EMP_TERM_DATE: pd.NaT,
    }

    current_column_order = list(core_hr_cols_definition)
    if gender_dist and isinstance(gender_dist, dict) and gender_dist:
        genders_keys = list(gender_dist.keys())
        gender_probs = list(gender_dist.values())
        if not genders_keys:
            logger.warning("Gender distribution dictionary is empty. Skipping gender assignment.")
        else:
            total_gender_prob = sum(gender_probs)
            if not np.isclose(total_gender_prob, 1.0) and total_gender_prob > 0:
                 logger.warning(f"Gender distribution sums to {total_gender_prob:.4f}; normalizing.")
                 gender_probs = [p / total_gender_prob for p in gender_probs]
            elif total_gender_prob <= 0:
                 logger.warning("Gender distribution invalid (sum <= 0). Skipping gender assignment.")
            else:
                 nh_df_data['gender'] = rng.choice(genders_keys, size=num_hires, p=gender_probs)
                 if 'gender' not in current_column_order:
                     current_column_order.append('gender')
    else:
        logger.debug("No valid gender distribution found in config. Skipping gender assignment.")

    nh_df = pd.DataFrame(nh_df_data)
    for col in current_column_order:
        if col not in nh_df.columns:
            nh_df[col] = pd.NA
    nh_df = nh_df[current_column_order]

    logger.info(f"Generated DataFrame for {len(nh_df)} new hires with columns: {list(nh_df.columns)}.")
    return nh_df