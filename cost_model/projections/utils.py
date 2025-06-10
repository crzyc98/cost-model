import pandas as pd

from cost_model.state.schema import EMP_TERM_DATE


def assign_employment_status(row, sim_year):
    """Assigns employment status based on hire date, term date, and active flag."""
    from ..utils.columns import EMP_HIRE_DATE

    hire_date = pd.to_datetime(row.get(EMP_HIRE_DATE, pd.NaT))
    hire_year = hire_date.year if pd.notna(hire_date) else None
    term_date = pd.to_datetime(row.get(EMP_TERM_DATE, pd.NaT), errors="coerce")
    term_year = term_date.year if pd.notna(term_date) else None
    if row.get("active"):
        return "New Hire Active" if hire_year == sim_year else "Continuous Active"
    else:
        if term_year == sim_year:
            return "New Hire Terminated" if hire_year == sim_year else "Experienced Terminated"
        else:
            return "Prior Terminated"


def filter_prior_terminated(
    snapshot_df: pd.DataFrame, sim_year: int, for_summary: bool = False
) -> pd.DataFrame:
    """
    Excludes employees terminated before the given simulation year.
    Always includes terminated employees for employment status calculations.
    """
    df = snapshot_df.copy()
    df[EMP_TERM_DATE] = pd.to_datetime(df.get(EMP_TERM_DATE, pd.NaT), errors="coerce")
    df["term_year"] = df[EMP_TERM_DATE].dt.year

    # Always include active employees and those who terminated in the current year
    mask = df["active"] | (df["term_year"] == sim_year)

    # For normal processing, also include prior terminations
    if not for_summary:
        mask = mask | (df["term_year"] < sim_year)

    out = df.loc[mask].drop(columns=["term_year"])
    return out
