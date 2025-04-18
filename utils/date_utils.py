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
    """Calculate tenure in months based on hire_date and current_date. Supports scalar and pandas Series."""
    def _tenure(x):
        if pd.isna(x):
            return 0
        x_ts = pd.Timestamp(x)
        dy = current_date.year - x_ts.year
        dm = current_date.month - x_ts.month
        total = dy * 12 + dm
        if current_date.day < x_ts.day:
            total -= 1
        return max(0, total)

    if isinstance(hire_date, pd.Series):
        return hire_date.apply(_tenure)
    return _tenure(hire_date)
