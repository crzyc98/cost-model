"""Date utility functions for the retirement plan simulation."""

import pandas as pd
from dateutil.relativedelta import relativedelta


def calculate_age(birth_date, current_date):
    """Calculate age in years based on birth_date and current_date. Supports scalar and pandas Series."""
    if isinstance(birth_date, pd.Series):
        return birth_date.apply(lambda bd: relativedelta(current_date, bd).years if not pd.isnull(bd) else 0)
    if pd.isnull(birth_date):
        return 0
    return relativedelta(current_date, birth_date).years


def calculate_tenure(hire_date, current_date):
    """Calculate tenure in fractional years based on hire_date and current_date. Supports scalar and pandas Series."""
    if isinstance(hire_date, pd.Series):
        hd = pd.to_datetime(hire_date)
        delta_days = (current_date - hd).dt.days.fillna(0)
        return delta_days / 365.25
    if pd.isna(hire_date):
        return 0.0
    delta = current_date - pd.Timestamp(hire_date)
    return delta.days / 365.25
