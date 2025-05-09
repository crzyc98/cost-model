# cost_model/utils/labels.py
"""
Utility functions for creating categorical labels based on simulation state.
"""

import pandas as pd
import numpy as np
import logging

# Attempt to import column constants, provide fallbacks
try:
    from .columns import EMP_HIRE_DATE, EMP_TERM_DATE
except ImportError:
    print(
        "Warning (labels.py): Could not import column constants from .columns. Using string literals."
    )
    EMP_HIRE_DATE = "employee_hire_date"
    EMP_TERM_DATE = "employee_termination_date"

logger = logging.getLogger(__name__)


def label_employment_status(df: pd.DataFrame, sim_year: int) -> pd.DataFrame:
    """
    Assigns an 'employment_status' label to each record based on hire and termination dates
    relative to the simulation year.

    Args:
        df: DataFrame containing employee data (must include hire and termination dates).
        sim_year: The current simulation year.

    Returns:
        The input DataFrame with an added 'employment_status' column.
    """
    if df.empty:
        logger.debug(
            f"Skipping employment status labeling for empty DataFrame in year {sim_year}."
        )
        df["employment_status"] = pd.NA  # Add column even if empty
        return df

    # Ensure date columns are datetime
    if EMP_HIRE_DATE not in df.columns:
        logger.error(f"Missing required column for status labeling: {EMP_HIRE_DATE}")
        df["employment_status"] = "Error: Missing Hire Date"
        return df
    if EMP_TERM_DATE not in df.columns:
        logger.warning(
            f"Column '{EMP_TERM_DATE}' not found. Creating with NaT for status labeling."
        )
        df[EMP_TERM_DATE] = pd.NaT

    df[EMP_HIRE_DATE] = pd.to_datetime(df[EMP_HIRE_DATE], errors="coerce")
    df[EMP_TERM_DATE] = pd.to_datetime(df[EMP_TERM_DATE], errors="coerce")

    plan_end_date = pd.Timestamp(f"{sim_year}-12-31")

    # Define conditions for np.select
    condlist = [
        # Hired and terminated in the same year
        (df[EMP_TERM_DATE].notna())
        & (df[EMP_TERM_DATE].dt.year == sim_year)
        & (df[EMP_HIRE_DATE].dt.year == sim_year),
        # Terminated this year (but hired previously)
        (df[EMP_TERM_DATE].notna()) & (df[EMP_TERM_DATE].dt.year == sim_year),
        # Hired this year and still active
        ((df[EMP_TERM_DATE].isna()) | (df[EMP_TERM_DATE] > plan_end_date))
        & (df[EMP_HIRE_DATE].dt.year == sim_year),
        # Hired in a prior year and still active
        ((df[EMP_TERM_DATE].isna()) | (df[EMP_TERM_DATE] > plan_end_date)),
        # Terminated in a prior year
        (df[EMP_TERM_DATE].notna()) & (df[EMP_TERM_DATE].dt.year < sim_year),
    ]

    # Define choices corresponding to conditions
    choicelist = [
        "New Hire Terminated",
        "Experienced Terminated",
        "New Hire Active",
        "Continuous Active",
        "Previously Terminated",
    ]

    df["employment_status"] = np.select(condlist, choicelist, default="Unknown")
    logger.debug(
        f"Assigned employment status labels for year {sim_year}. Distribution:\n{df['employment_status'].value_counts()}"
    )

    return df


# Example Usage (for testing this module directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_data = {
        EMP_HIRE_DATE: [
            "2020-01-15",
            "2023-05-10",
            "2024-02-20",
            "2024-08-01",
            "2022-11-30",
        ],
        EMP_TERM_DATE: [pd.NaT, "2024-06-15", pd.NaT, "2024-10-01", "2023-03-01"],
    }
    test_df = pd.DataFrame(test_data)
    test_year = 2024
    labeled_df = label_employment_status(test_df.copy(), test_year)
    print(f"\n--- Labeled DataFrame (Year {test_year}) ---")
    print(labeled_df)
