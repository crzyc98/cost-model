# cost_model/dynamics/hiring.py
"""
Functions related to generating new hires during workforce simulations.
QuickStart: see docs/cost_model/dynamics/hiring.md
"""

import logging
from typing import Any, Dict, Optional, Sequence, Union  # Added Union

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from cost_model.utils.columns import (
    EMP_BIRTH_DATE,
    EMP_GROSS_COMP,
    EMP_HIRE_DATE,
    EMP_TERM_DATE,
)

# from cost_model.utils.date_utils import calculate_age, calculate_tenure # Not directly used
from cost_model.utils.id_generation import _generate_sequential_ids

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
    return year_start + pd.to_timedelta(random_day_offsets, unit="D")


def _calculate_compensation_simple(
    ages: np.ndarray, comp_params: Dict[str, Any], min_age_cfg: int, rng: np.random.Generator
) -> np.ndarray:
    """Calculate compensation based on age without role-based logic."""

    def get_param_value(params: Dict[str, Any], key: str, default: float) -> float:
        return float(params.get(key, default))

    base = get_param_value(comp_params, "comp_base_salary", 50000.0)
    age_factor = get_param_value(comp_params, "comp_age_factor", 0.01)
    min_salary = get_param_value(comp_params, "comp_min_salary", 40000.0)
    max_salary = get_param_value(comp_params, "comp_max_salary", base * 1.5)
    stochastic_std_dev = get_param_value(comp_params, "comp_stochastic_std_dev", 0.0)

    compensation_list = []

    for age in ages:
        age_comp = base * (age - min_age_cfg) * age_factor
        initial_comp = max(base + age_comp, min_salary)

        if stochastic_std_dev > 0:
            if initial_comp <= 0:
                logger.warning(
                    f"Initial comp for age {age} is {initial_comp:.2f}. Using min_salary ({min_salary}) for log-normal base."
                )
                initial_comp_for_log = max(min_salary, 1.0)
            else:
                initial_comp_for_log = initial_comp

            log_mean = np.log(initial_comp_for_log) - (stochastic_std_dev**2) / 2
            raw_comp = np.exp(rng.normal(log_mean, stochastic_std_dev))
            clamped_comp = max(min_salary, min(raw_comp, max_salary))
            compensation_list.append(clamped_comp)
        else:
            compensation_list.append(initial_comp)

    return np.array(compensation_list)


def _generate_ages(
    num: int,
    mean: float,
    std_dev: float,
    min_age: int,
    max_age: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate an array of integer ages for new hires using a truncated normal distribution.

    Args:
        num: Number of ages to generate.
        mean: Mean age for the distribution.
        std_dev: Standard deviation for the distribution.
        min_age: Minimum allowed age (inclusive).
        max_age: Maximum allowed age (inclusive).
        rng: Numpy random Generator for reproducibility.

    Returns:
        np.ndarray of shape (num,) with integer ages, clipped to [min_age, max_age].
    """
    safe_std_dev = max(std_dev, 1e-6)
    a, b = (min_age - mean) / safe_std_dev, (max_age - mean) / safe_std_dev
    ages = truncnorm.rvs(a, b, loc=mean, scale=safe_std_dev, size=num, random_state=rng)
    return np.round(np.clip(ages, min_age, max_age)).astype(int)


def _calculate_birth_dates(
    hire_dates: pd.Series, ages: np.ndarray, rng: np.random.Generator
) -> pd.Series:
    """
    Calculate birth dates by subtracting exact years and adding random jitter.
    This ensures unique birth dates and proper date arithmetic.

    Args:
        hire_dates: Series of hire dates
        ages: Array of ages
        rng: Random number generator

    Returns:
        Series of birth dates with datetime64[ns] dtype
    """
    # Generate random month and day for each hire
    months = rng.integers(1, 13, size=len(hire_dates))
    days = rng.integers(1, 29, size=len(hire_dates))

    # Calculate birth dates by subtracting exact years and using random month/day
    birth_dates = []
    for hire_date, age, month, day in zip(hire_dates, ages, months, days):
        try:
            birth_year = hire_date.year - age
            birth_date = pd.Timestamp(year=birth_year, month=month, day=day)
            birth_dates.append(birth_date)
        except ValueError:
            # If the date is invalid (like Feb 30), use the first day of the month
            birth_date = pd.Timestamp(year=birth_year, month=month, day=1)
            birth_dates.append(birth_date)

    # Create Series with explicit datetime64[ns] dtype
    birth_dates_series = pd.Series(
        birth_dates, index=hire_dates.index, name=EMP_BIRTH_DATE, dtype="datetime64[ns]"
    )

    # Debug logging
    if not pd.api.types.is_datetime64_any_dtype(birth_dates_series):
        logger.warning(f"Birth dates not in datetime format: {birth_dates_series.dtype}")
    if birth_dates_series.isna().any():
        logger.warning(f"Found NA values in birth dates: {birth_dates_series.isna().sum()}")

    return birth_dates_series


# --- REMOVED: Role-based compensation function no longer needed ---


def generate_new_hires(
    num_hires: int,
    hire_year: int,
    scenario_config: Dict[str, Any],
    existing_ids: Optional[Sequence[str]] = None,
    rng: Optional[np.random.Generator] = None,
    id_col_name: str = "employee_id",
) -> pd.DataFrame:
    core_hr_cols_definition = [
        id_col_name,
        EMP_HIRE_DATE,
        EMP_BIRTH_DATE,
        EMP_GROSS_COMP,
        "employment_status",
        EMP_TERM_DATE,
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

    # Role-based compensation removed as part of schema refactoring
    age_mean = float(get_config_val(scenario_config, "new_hire_average_age", 30.0))
    age_std_dev = float(get_config_val(scenario_config, "new_hire_age_std_dev", 5.0))
    min_age_cfg = int(get_config_val(scenario_config, "min_working_age", 18))
    max_age_cfg = int(get_config_val(scenario_config, "max_working_age", 65))
    gender_dist = get_config_val(scenario_config, "gender_distribution", None)

    # --- Definition of default_comp_params_as_dict ---
    # This is the dictionary that will be passed as default to _calculate_compensation
    default_comp_params_as_dict = {
        "comp_base_salary": get_config_val(scenario_config, "comp_base_salary", 50000.0),
        "comp_age_factor": get_config_val(scenario_config, "comp_age_factor", 0.01),
        "comp_stochastic_std_dev": get_config_val(scenario_config, "comp_stochastic_std_dev", 0.0),
        "comp_min_salary": get_config_val(scenario_config, "comp_min_salary", 30000.0),
        # Add other default comp keys here if _calculate_compensation needs them via get_param_value
    }
    # --- End of definition ---

    ids = _generate_sequential_ids(existing_ids if existing_ids is not None else [], num_hires)
    hire_dates = _generate_hire_dates(num_hires, hire_year, rng)
    # Role assignment removed as part of schema refactoring
    ages = _generate_ages(num_hires, age_mean, age_std_dev, min_age_cfg, max_age_cfg, rng)

    # Debug logging for age distribution
    logger.debug(
        f"[NEW-HIRE] Age distribution - avg={ages.mean():.1f}, min={ages.min()}, max={ages.max()}, std={ages.std():.1f}"
    )
    logger.debug(
        f"[NEW-HIRE] Age histogram: {pd.Series(ages).value_counts().sort_index().to_dict()}"
    )

    print("Generated Ages:", ages)
    # Defensive check: warn if all ages are identical (could cause identical birth dates)
    if len(set(ages)) == 1:
        logger.warning(
            f"All generated ages are identical ({ages[0]}). This may cause identical birth dates. Check scenario_config or RNG usage."
        )
    birth_dates = _calculate_birth_dates(hire_dates, ages, rng)
    print("Calculated Birth Dates:", birth_dates.tolist())

    # Debug: Verify birth date types and values
    if not pd.api.types.is_datetime64_any_dtype(birth_dates):
        logger.warning(f"Birth dates not in datetime format: {birth_dates.dtype}")
    if birth_dates.isna().any():
        logger.warning(f"Found NA values in birth dates: {birth_dates.isna().sum()}")

    # Ensure birth dates are properly converted to datetime
    birth_dates = pd.to_datetime(birth_dates)

    # Debug: Verify final birth date values
    unique_dates = birth_dates.unique()
    if len(unique_dates) < len(birth_dates):
        logger.warning(
            f"Only {len(unique_dates)} unique birth dates among {len(birth_dates)} hires"
        )
    if (unique_dates == pd.Timestamp("1990-01-01")).any():
        logger.warning("Found default birth date 1990-01-01 in birth dates")

    # Ensure birth dates are explicitly included in the DataFrame
    nh_df_data = {
        id_col_name: ids,
        EMP_HIRE_DATE: hire_dates,
        EMP_BIRTH_DATE: birth_dates,
        EMP_GROSS_COMP: _calculate_compensation_simple(
            ages, default_comp_params_as_dict, min_age_cfg, rng
        ),
        "employment_status": "Active",
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
                logger.warning(
                    "Gender distribution invalid (sum <= 0). Skipping gender assignment."
                )
            else:
                nh_df_data["gender"] = rng.choice(genders_keys, size=num_hires, p=gender_probs)
                if "gender" not in current_column_order:
                    current_column_order.append("gender")
    else:
        logger.debug("No valid gender distribution found in config. Skipping gender assignment.")

    nh_df = pd.DataFrame(nh_df_data)
    for col in current_column_order:
        if col not in nh_df.columns:
            nh_df[col] = pd.NA
    nh_df = nh_df[current_column_order]

    logger.info(
        f"Generated DataFrame for {len(nh_df)} new hires with columns: {list(nh_df.columns)}."
    )
    return nh_df
