# utils/date_utils.py

"""Date utility functions for the retirement plan simulation."""

import pandas as pd  # type: ignore[import-untyped]
from dateutil.relativedelta import relativedelta


def calculate_age(
    birth_date: pd.Series | pd.Timestamp | None,
    current_date: pd.Timestamp
) -> pd.Series | int:
    """
    Calculate age in years based on birth_date and current_date.
    - If birth_date is a Series, returns a Series of ints.
    - If birth_date is scalar (Timestamp or None), returns an int.
    """
    if isinstance(birth_date, pd.Series):
        return birth_date.apply(
            lambda bd: relativedelta(current_date, bd).years
                       if not pd.isna(bd) else 0
        )
    if pd.isna(birth_date):
        return 0
    return relativedelta(current_date, pd.to_datetime(birth_date)).years


def calculate_tenure(
    hire_date: pd.Series | pd.Timestamp | None,
    current_date: pd.Timestamp
) -> pd.Series | float:
    """
    Calculate tenure in fractional years based on hire_date and current_date.
    - If hire_date is a Series, returns a Series of floats.
    - If hire_date is scalar (Timestamp or None), returns a float.
    """
    if isinstance(hire_date, pd.Series):
        hd = pd.to_datetime(hire_date, errors='coerce')
        delta_days = (current_date - hd).dt.days.fillna(0)
        return delta_days / 365.25
    if pd.isna(hire_date):
        return 0.0
    delta = current_date - pd.to_datetime(hire_date)
    return delta.days / 365.25