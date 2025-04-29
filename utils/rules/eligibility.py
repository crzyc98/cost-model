"""
Eligibility rule: age/service/hours + entry-date calc
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from utils.date_utils import calculate_age, calculate_tenure
from utils.constants import ACTIVE_STATUSES

logger = logging.getLogger(__name__)

def apply(
    df: pd.DataFrame,
    plan_rules: Dict[str, Any],
    simulation_year_end_date: pd.Timestamp
) -> pd.DataFrame:
    """Apply eligibility rules to the DataFrame."""
    logger.info(f"Determining eligibility for {simulation_year_end_date.year}")
    eligibility_config = plan_rules.get('eligibility', {})
    min_age = eligibility_config.get('min_age', 21)
    min_service_months = eligibility_config.get('min_service_months', 12)

    # Early exit if required columns missing
    if 'birth_date' not in df.columns or 'hire_date' not in df.columns:
        logger.warning("'birth_date' or 'hire_date' columns missing. Cannot determine eligibility.")
        if 'is_eligible' not in df.columns:
            df['is_eligible'] = False
        if 'eligibility_entry_date' not in df.columns:
            df['eligibility_entry_date'] = pd.NaT
        return df

    # Ensure datetime types
    df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')
    df['hire_date'] = pd.to_datetime(df['hire_date'], errors='coerce')

    # Warn on parse failures
    if df['birth_date'].isnull().any() or df['hire_date'].isnull().any():
        logger.warning("Some 'birth_date' or 'hire_date' values could not be parsed. Affected rows may not be marked eligible.")

    # Calculate age where possible
    valid_bd = df['birth_date'].notna()
    df['current_age'] = pd.NA
    df.loc[valid_bd, 'current_age'] = calculate_age(df.loc[valid_bd, 'birth_date'], simulation_year_end_date)

    # Prepare eligibility entry date
    if 'eligibility_entry_date' not in df.columns:
        df['eligibility_entry_date'] = pd.NaT
    else:
        df['eligibility_entry_date'] = pd.to_datetime(df['eligibility_entry_date'], errors='coerce')

    # Vectorized eligibility entry date calculation
    service_met = df['hire_date'] + pd.DateOffset(months=min_service_months)
    age_met = df['birth_date'] + pd.DateOffset(years=min_age)

    combined = np.maximum(service_met.fillna(pd.Timestamp.min), age_met.fillna(pd.Timestamp.min))
    combined[service_met.isna() | age_met.isna()] = pd.NaT
    combined[combined == pd.Timestamp.min] = pd.NaT
    df['eligibility_entry_date'] = pd.to_datetime(combined, errors='coerce')

    # Determine base eligibility (age/service/status)
    eligible_by_date = (df['eligibility_entry_date'] <= simulation_year_end_date) & df['eligibility_entry_date'].notna()
    active_mask = df['status'].isin(ACTIVE_STATUSES)

    # Hours requirement (optional)
    min_hours = eligibility_config.get('min_hours_worked', None)
    if min_hours is not None:
        if 'hours_worked' in df.columns:
            meets_hours = df['hours_worked'] >= min_hours
        else:
            meets_hours = pd.Series(False, index=df.index)
    else:
        meets_hours = pd.Series(True, index=df.index)

    # Combine all requirements
    df['is_eligible'] = eligible_by_date & active_mask & meets_hours

    eligible_count = df['is_eligible'].sum()
    logger.info(f"Eligibility determined: {eligible_count} eligible employees.")

    return df

# Single-agent eligibility check using shared logic
def agent_is_eligible(
    birth_date: pd.Timestamp,
    hire_date: pd.Timestamp,
    status: Any,
    hours_worked: Optional[float],
    eligibility_config: Dict[str, Any],
    simulation_year_end_date: pd.Timestamp
) -> bool:
    """Determine eligibility for a single agent."""
    min_age = eligibility_config.get('min_age', 21)
    min_service_months = eligibility_config.get('min_service_months', 12)
    min_hours = eligibility_config.get('min_hours_worked', None)

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
    if min_hours is not None:
        meets_hours = hours_worked >= min_hours if hours_worked is not None else False
    else:
        meets_hours = True

    return meets_age and meets_service and meets_status and meets_hours

def is_eligible(
    row,
    eligibility_config,
    simulation_year_end_date=None
) -> bool:
    """
    Row-wise eligibility wrapper for backward compatibility.
    """
    if simulation_year_end_date is None:
        simulation_year_end_date = pd.Timestamp.today()
    # If no rules specified or placeholder, assume everyone eligible
    if not eligibility_config or eligibility_config is Ellipsis:
        return True
    return agent_is_eligible(
        row.get('birth_date', None),
        row.get('hire_date', None),
        row.get('status', None),
        row.get('hours_worked', None),
        eligibility_config,
        simulation_year_end_date,
    )
