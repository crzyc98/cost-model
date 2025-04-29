"""
rules/eligibility.py - Eligibility rule: age/service/hours + entry-date calc
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from utils.date_utils import calculate_age, calculate_tenure
from utils.constants import ACTIVE_STATUSES
from utils.columns import EMP_BIRTH_DATE, EMP_HIRE_DATE, ELIGIBILITY_ENTRY_DATE, IS_ELIGIBLE

# Module-level defaults
DEFAULT_MIN_AGE = 21
DEFAULT_SERVICE_MONTHS = 0

logger = logging.getLogger(__name__)

def apply(df: pd.DataFrame, plan_rules: Dict[str, Any], simulation_year_end_date: pd.Timestamp) -> pd.DataFrame:
    """Eligibility rule: age/service/hours + entry-date calc"""
    # --- Ensure employment status columns are present and up to date ---
    from utils.data_processing import assign_employment_status
    start_year = simulation_year_end_date.year
    df = assign_employment_status(df, start_year)

    logger.info(f"Determining eligibility for {simulation_year_end_date.year}")
    eligibility_config = plan_rules.get('eligibility', {})
    min_age = eligibility_config.get('min_age', DEFAULT_MIN_AGE)
    min_service_months = eligibility_config.get('min_service_months', DEFAULT_SERVICE_MONTHS)

    # Early exit if required columns missing
    if EMP_BIRTH_DATE not in df.columns or EMP_HIRE_DATE not in df.columns:
        logger.warning("'employee_birth_date' or 'employee_hire_date' columns missing. Cannot determine eligibility.")
        if IS_ELIGIBLE not in df.columns:
            df[IS_ELIGIBLE] = False
        if ELIGIBILITY_ENTRY_DATE not in df.columns:
            df[ELIGIBILITY_ENTRY_DATE] = pd.NaT
        return df

    # Ensure datetime types
    df[EMP_BIRTH_DATE] = pd.to_datetime(df[EMP_BIRTH_DATE], errors='coerce')
    df[EMP_HIRE_DATE] = pd.to_datetime(df[EMP_HIRE_DATE], errors='coerce')

    # Warn on parse failures
    if df[EMP_BIRTH_DATE].isnull().any() or df[EMP_HIRE_DATE].isnull().any():
        logger.warning("Some 'employee_birth_date' or 'employee_hire_date' values could not be parsed. Affected rows may not be marked eligible.")

    # Calculate age where possible
    valid_bd = df[EMP_BIRTH_DATE].notna()
    df['current_age'] = pd.NA
    df.loc[valid_bd, 'current_age'] = calculate_age(df.loc[valid_bd, EMP_BIRTH_DATE], simulation_year_end_date)

    # Ensure existing entry-date
    if ELIGIBILITY_ENTRY_DATE not in df.columns:
        df[ELIGIBILITY_ENTRY_DATE] = pd.NaT
    else:
        df[ELIGIBILITY_ENTRY_DATE] = pd.to_datetime(df[ELIGIBILITY_ENTRY_DATE], errors='coerce')

    # Vectorized eligibility entry date calculation
    service_met = df[EMP_HIRE_DATE] + pd.DateOffset(months=min_service_months)
    age_met = df[EMP_BIRTH_DATE] + pd.DateOffset(years=min_age)

    combined = np.maximum(service_met.fillna(pd.Timestamp.min), age_met.fillna(pd.Timestamp.min))
    combined[service_met.isna() | age_met.isna()] = pd.NaT
    combined[combined == pd.Timestamp.min] = pd.NaT
    df[ELIGIBILITY_ENTRY_DATE] = pd.to_datetime(combined, errors='coerce')

    # Determine base eligibility (age/service/status)
    eligible_by_date = (df[ELIGIBILITY_ENTRY_DATE] <= simulation_year_end_date) & df[ELIGIBILITY_ENTRY_DATE].notna()
    active_mask = df['status'].isin(ACTIVE_STATUSES)

    # Hours requirement (optional)
    min_hours = eligibility_config.get('min_hours_worked', None)
    if min_hours is not None:
        meets_hours = df['hours_worked'].ge(min_hours) if 'hours_worked' in df.columns else pd.Series(False, index=df.index)
    else:
        meets_hours = pd.Series(True, index=df.index)

    # Combine all requirements
    df[IS_ELIGIBLE] = eligible_by_date & active_mask & meets_hours

    # Drop intermediate columns
    df.drop(columns=['current_age'], inplace=True)

    eligible_count = df[IS_ELIGIBLE].sum()
    logger.info(f"Eligibility determined: {eligible_count} eligible employees.")

    return df

def agent_is_eligible(birth_date: pd.Timestamp, hire_date: pd.Timestamp, status: Any, hours_worked: Optional[float], eligibility_config: Dict[str, Any], simulation_year_end_date: pd.Timestamp) -> bool:
    """Single-agent eligibility wrapper."""
    min_age = eligibility_config.get('min_age', DEFAULT_MIN_AGE)
    min_service_months = eligibility_config.get('min_service_months', DEFAULT_SERVICE_MONTHS)

    # Age check
    age = calculate_age(birth_date, simulation_year_end_date) if birth_date is not None and pd.notna(birth_date) else 0
    meets_age = age >= min_age

    # Service check
    if hire_date is not None and pd.notna(hire_date):
        service_met_date = hire_date + pd.DateOffset(months=min_service_months)
        meets_service = service_met_date <= simulation_year_end_date
    else:
        meets_service = False

    # Status check
    meets_status = status in ACTIVE_STATUSES

    # Hours check
    if 'min_hours' in eligibility_config:
        min_hours = eligibility_config.get('min_hours', None)
        meets_hours = hours_worked >= min_hours if hours_worked is not None else False
    else:
        meets_hours = True

    return meets_age and meets_service and meets_status and meets_hours

def is_eligible(row: pd.Series, eligibility_config, simulation_year_end_date=None) -> bool:
    """Row-wise eligibility wrapper."""
    if simulation_year_end_date is None:
        simulation_year_end_date = pd.Timestamp.today()
    # If no rules specified or placeholder, assume everyone eligible
    if not eligibility_config or eligibility_config is Ellipsis:
        return True
    return agent_is_eligible(
        row.get(EMP_BIRTH_DATE),
        row.get(EMP_HIRE_DATE),
        row.get('status'),
        row.get('hours_worked'),
        eligibility_config,
        simulation_year_end_date
    )
