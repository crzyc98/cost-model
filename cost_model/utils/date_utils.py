# utils/date_utils.py

"""Date utility functions for the retirement plan simulation."""

import pandas as pd # type: ignore[import-untyped]
import numpy as np # Added for np.random.Generator
from dateutil.relativedelta import relativedelta
from typing import Union, List # Added List for type hinting
import calendar # Added for checking valid days in month

def calculate_age(
    birth_date: Union[pd.Series, pd.Timestamp, None],
    current_date: pd.Timestamp
) -> Union[pd.Series, int]:
    """
    Calculate age in years based on birth_date and current_date.
    - If birth_date is a Series, returns a Series of ints.
    - If birth_date is scalar (Timestamp or None), returns an int.
    """
    if isinstance(birth_date, pd.Series):
        # Ensure birth_date Series is coerced to datetime
        bd_series = pd.to_datetime(birth_date, errors='coerce')
        return bd_series.apply(
            lambda bd: relativedelta(current_date, bd).years
                       if pd.notna(bd) else 0 # Changed from not pd.isna(bd) for consistency
        )
    if pd.isna(birth_date): # Also pd.isnull(birth_date) is common
        return 0
    # Ensure scalar birth_date is also coerced to datetime
    return relativedelta(current_date, pd.to_datetime(birth_date, errors='coerce')).years


def calculate_tenure(
    hire_date: Union[pd.Series, pd.Timestamp, None],
    current_date: pd.Timestamp
) -> Union[pd.Series, float]:
    """
    Calculate tenure in fractional years based on hire_date and current_date.
    - If hire_date is a Series, returns a Series of floats.
    - If hire_date is scalar (Timestamp or None), returns a float.
    """
    if isinstance(hire_date, pd.Series):
        hd = pd.to_datetime(hire_date, errors='coerce')
        # Calculate delta only for valid (non-NaT) hire dates
        # For NaT dates, dt.days would error or give undesired results
        delta_days = (current_date - hd).dt.days
        # Fill NaNs that arose from NaT hire dates or other issues with 0
        return delta_days.fillna(0) / 365.25
    if pd.isna(hire_date): # Also pd.isnull(hire_date)
        return 0.0
    hd_scalar = pd.to_datetime(hire_date, errors='coerce')
    if pd.isna(hd_scalar): # If conversion results in NaT
        return 0.0
    delta = current_date - hd_scalar
    return delta.days / 365.25

# NEW FUNCTION
def get_random_dates_in_year(
    year: int,
    count: int,
    rng: np.random.Generator,
    day_of_month: int = 15 # Default to mid-month
) -> List[pd.Timestamp]:
    """
    Generates a list of random dates within a given year.

    Args:
        year: The year for which to generate dates.
        count: The number of random dates to generate.
        rng: The numpy random number generator instance.
        day_of_month: The day of the month for each generated date.
                      If the day is invalid for a chosen month (e.g., 31 for Feb),
                      it will be capped at the last valid day of that month.

    Returns:
        A list of pandas Timestamp objects.
    """
    dates: List[pd.Timestamp] = []
    if count == 0:
        return dates

    random_months = rng.integers(1, 13, size=count) # 1 (Jan) to 12 (Dec)

    for month_num in random_months:
        # Determine the last valid day for the generated month and year
        _, last_day_of_month = calendar.monthrange(year, month_num)
        actual_day = min(day_of_month, last_day_of_month)

        try:
            dates.append(pd.Timestamp(f"{year}-{month_num:02d}-{actual_day:02d}"))
        except ValueError as e:
            # This should ideally not happen if actual_day logic is correct
            # Fallback to the 1st of the month if something unexpected occurs
            print(f"Warning: Could not create date {year}-{month_num}-{actual_day}. Error: {e}. Defaulting to 1st.")
            dates.append(pd.Timestamp(f"{year}-{month_num:02d}-01"))
            
    return dates