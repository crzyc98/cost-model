"""
rules/eligibility.py - Eligibility rule: age/service/hours + entry-date calc
"""
import pandas as pd
import numpy as np
from typing import Optional, Any, Dict
import logging
from utils.date_utils import calculate_age, calculate_tenure
from utils.constants import ACTIVE_STATUSES
from utils.columns import EMP_BIRTH_DATE, EMP_HIRE_DATE, ELIGIBILITY_ENTRY_DATE, IS_ELIGIBLE, STATUS_COL, HOURS_WORKED
from utils.data_processing import assign_employment_status  # unused: respect upstream STATUS_COL
from utils.rules.validators import EligibilityRule

logger = logging.getLogger(__name__)

def apply(df: pd.DataFrame, eligibility_cfg: EligibilityRule, simulation_year_end_date: pd.Timestamp) -> pd.DataFrame:
    """Eligibility rule: age/service/hours + entry-date calc"""
    # Assume STATUS_COL is provided upstream
    # start_year = simulation_year_end_date.year
    # df = assign_employment_status(df, start_year)

    logger.info(f"Determining eligibility for {simulation_year_end_date.year}")
    min_age = eligibility_cfg.min_age
    min_service_months = eligibility_cfg.min_service_months

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

    # Calculate service tenure in months
    df['tenure_months'] = calculate_tenure(df[EMP_HIRE_DATE], simulation_year_end_date) * 12

    # Ensure existing entry-date
    if ELIGIBILITY_ENTRY_DATE not in df.columns:
        df[ELIGIBILITY_ENTRY_DATE] = pd.NaT
    else:
        df[ELIGIBILITY_ENTRY_DATE] = pd.to_datetime(df[ELIGIBILITY_ENTRY_DATE], errors='coerce')

    # Simplify entry-date: take max of service and age dates
    date_service = df[EMP_HIRE_DATE] + pd.DateOffset(months=min_service_months)
    date_age = df[EMP_BIRTH_DATE] + pd.DateOffset(years=min_age)
    df[ELIGIBILITY_ENTRY_DATE] = date_service.combine(date_age, max)

    # Normalize status for a case-insensitive match (and replace en-dashes)
    df_status = df[STATUS_COL].astype(str)
    logger.debug("STATUS unique (raw): %r", df_status.unique())

    df_status = (
        df_status
        .str.replace("–", "-", regex=False)
        .str.strip()
        .str.casefold()
    )
    allowed = { s.replace("–","-").casefold() for s in ACTIVE_STATUSES }
    logger.debug("Allowed statuses (case-folded): %r", allowed)

    active_mask = df_status.isin(allowed)
    logger.debug("active_mask.sum() = %d / %d", int(active_mask.sum()), len(active_mask))

    # Determine base eligibility (age/service/status)
    eligible_by_date = (df[ELIGIBILITY_ENTRY_DATE] <= simulation_year_end_date) & df[ELIGIBILITY_ENTRY_DATE].notna()
    logger.debug("eligible_by_date.sum() = %d / %d", int(eligible_by_date.sum()), len(eligible_by_date))

    # Hours requirement (optional)
    min_hours = eligibility_cfg.min_hours_worked
    if min_hours is not None:
        meets_hours = df[HOURS_WORKED].ge(min_hours) if HOURS_WORKED in df.columns else pd.Series(False, index=df.index)
    else:
        meets_hours = pd.Series(True, index=df.index)

    # Combine all requirements and ensure pure Python bools
    mask = eligible_by_date & active_mask & meets_hours
    # Create object-dtype Series of Python bools for identity-safe comparisons
    df[IS_ELIGIBLE] = pd.Series([bool(v) for v in mask], index=df.index, dtype=object)

    # Drop intermediate columns
    df.drop(columns=['current_age', 'tenure_months'], inplace=True)

    eligible_count = df[IS_ELIGIBLE].sum()
    logger.info(f"Eligibility determined: {eligible_count} eligible employees.")

    return df

def agent_is_eligible(birth_date: pd.Timestamp, hire_date: pd.Timestamp, status: Any, hours_worked: Optional[float], eligibility_config: Dict[str, Any], simulation_year_end_date: pd.Timestamp) -> bool:
    """Single-agent eligibility wrapper."""
    min_age = eligibility_config.get('min_age', 21)
    min_service_months = eligibility_config.get('min_service_months', 0)
    min_hours = eligibility_config.get('min_hours_worked', None)

    # Age check
    age = calculate_age(birth_date, simulation_year_end_date) if birth_date is not None and pd.notna(birth_date) else 0
    meets_age = age >= min_age

    # Service check
    tenure = calculate_tenure(hire_date, simulation_year_end_date) * 12 if hire_date is not None and pd.notna(hire_date) else 0
    meets_service = tenure >= min_service_months

    # Status check
    meets_status = status == ACTIVE_STATUSES[0]

    # Hours check
    if min_hours is not None:
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
        row.get(STATUS_COL),
        row.get(HOURS_WORKED),
        eligibility_config,
        simulation_year_end_date
    )
