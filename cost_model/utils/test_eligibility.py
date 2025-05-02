# utils/test_eligibility.py
import pandas as pd
import pytest
from cost_model.utils.columns import EMP_BIRTH_DATE, EMP_HIRE_DATE, STATUS_COL, HOURS_WORKED, ELIGIBILITY_ENTRY_DATE, IS_ELIGIBLE
from cost_model.rules.eligibility import apply, agent_is_eligible, DEFAULT_MIN_AGE, DEFAULT_SERVICE_MONTHS
from cost_model.utils.constants import ACTIVE_STATUSES

# Helper to build simulation end date
SIM_END = pd.Timestamp("2025-12-31")

@pytest.fixture
def base_df():
    # Minimal DF with date and status columns
    return pd.DataFrame({
        EMP_BIRTH_DATE: [],
        EMP_HIRE_DATE: [],
        'employee_termination_date': [],
        STATUS_COL: [],
        HOURS_WORKED: []
    })

def test_missing_dates(base_df):
    df = base_df.copy()
    # simulate missing dates
    df[EMP_BIRTH_DATE] = pd.NaT
    df[EMP_HIRE_DATE] = pd.NaT
    df['employee_termination_date'] = pd.NaT
    out = apply(df, {"eligibility": {}}, SIM_END)
    assert IS_ELIGIBLE in out.columns
    assert (out[IS_ELIGIBLE] == False).all()
    assert ELIGIBILITY_ENTRY_DATE in out.columns

def test_zero_hours(base_df):
    # Enough age/tenure but zero hours
    df = pd.DataFrame({
        EMP_BIRTH_DATE: [pd.Timestamp("2000-01-01")],
        EMP_HIRE_DATE: [pd.Timestamp("2024-01-01")],
        'employee_termination_date': [pd.NaT],
        STATUS_COL: [ACTIVE_STATUSES[0]],
        HOURS_WORKED: [0]
    })
    rules = {"eligibility": {"min_hours_worked": 1}}
    out = apply(df, rules, SIM_END)
    assert out[IS_ELIGIBLE].iloc[0] is False

def test_inactive_status():
    df = pd.DataFrame({
        EMP_BIRTH_DATE: [pd.Timestamp("2000-01-01")],
        EMP_HIRE_DATE: [pd.Timestamp("2010-01-01")],
        'employee_termination_date': [pd.NaT],
        STATUS_COL: ["INACTIVE_X"],
        HOURS_WORKED: [40]
    })
    out = apply(df, {"eligibility": {}}, SIM_END)
    assert out[IS_ELIGIBLE].iloc[0] is False

def test_valid_case():
    df = pd.DataFrame({
        EMP_BIRTH_DATE: [pd.Timestamp("2000-01-01")],
        EMP_HIRE_DATE: [pd.Timestamp("2010-01-01")],
        'employee_termination_date': [pd.NaT],
        STATUS_COL: [ACTIVE_STATUSES[0]],
        HOURS_WORKED: [40]
    })
    # min_age default 21, min_service default 0
    out = apply(df, {"eligibility": {}}, SIM_END)
    # Born 2000, age 25 in 2025 -> eligible; tenure >=0
    assert out[IS_ELIGIBLE].iloc[0] is True

def test_agent_agreement():
    # Compare agent_is_eligible to apply row
    df = pd.DataFrame({
        EMP_BIRTH_DATE: [pd.Timestamp("1990-06-15")],
        EMP_HIRE_DATE: [pd.Timestamp("2020-06-15")],
        'employee_termination_date': [pd.NaT],
        STATUS_COL: [ACTIVE_STATUSES[0]],
        HOURS_WORKED: [20]
    })
    rules = {"eligibility": {"min_age": 30, "min_service_months": 60}}
    out = apply(df, rules, SIM_END)
    row = df.iloc[0]
    assert out[IS_ELIGIBLE].iloc[0] == agent_is_eligible(
        row[EMP_BIRTH_DATE], row[EMP_HIRE_DATE], row[STATUS_COL], row[HOURS_WORKED], rules["eligibility"], SIM_END)

if __name__ == "__main__":
    pytest.main([__file__])
