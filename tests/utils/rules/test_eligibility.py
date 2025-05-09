import pandas as pd
from pandas import Timestamp
from utils.constants import ACTIVE_STATUSES
from utils.rules.eligibility import apply


def make_df(birth_date, hire_date, status=ACTIVE_STATUSES[0], hours_worked=None):
    data = {"birth_date": [birth_date], "hire_date": [hire_date], "status": [status]}
    if hours_worked is not None:
        data["hours_worked"] = [hours_worked]
    return pd.DataFrame(data)


def test_missing_columns():
    df = pd.DataFrame({"foo": [1, 2]})
    out = apply(df.copy(), {}, Timestamp("2025-12-31"))
    assert "is_eligible" in out.columns
    assert not out["is_eligible"].any()
    assert "eligibility_entry_date" in out.columns
    assert out["eligibility_entry_date"].isna().all()


def test_age_and_service_eligibility():
    sim_date = Timestamp("2025-12-31")
    # Exactly at min thresholds
    birth = sim_date - pd.DateOffset(years=21)
    hire = sim_date - pd.DateOffset(months=12)
    df = make_df(birth, hire, status=ACTIVE_STATUSES[0])
    rules = {"eligibility": {"min_age": 21, "min_service_months": 12}}
    out = apply(df.copy(), rules, sim_date)
    assert out["is_eligible"].iloc[0]

    # Under age
    df2 = make_df(sim_date - pd.DateOffset(years=20), hire, status=ACTIVE_STATUSES[0])
    out2 = apply(df2.copy(), rules, sim_date)
    assert not out2["is_eligible"].iloc[0]

    # Under service
    df3 = make_df(birth, sim_date - pd.DateOffset(months=11), status=ACTIVE_STATUSES[0])
    out3 = apply(df3.copy(), rules, sim_date)
    assert not out3["is_eligible"].iloc[0]

    # Terminated employees should not be eligible
    df4 = make_df(birth, hire, status="Terminated")
    out4 = apply(df4.copy(), rules, sim_date)
    assert not out4["is_eligible"].iloc[0]


def test_hours_requirement():
    sim_date = Timestamp("2025-12-31")
    birth = sim_date - pd.DateOffset(years=30)
    hire = sim_date - pd.DateOffset(months=24)
    # Require 40 hours; only 30 hours worked -> ineligible
    df = make_df(birth, hire, status=ACTIVE_STATUSES[0], hours_worked=30)
    rules = {
        "eligibility": {"min_age": 21, "min_service_months": 12, "min_hours_worked": 40}
    }
    out = apply(df.copy(), rules, sim_date)
    assert not out["is_eligible"].iloc[0]

    # 40 hours worked -> eligible
    df2 = make_df(birth, hire, status=ACTIVE_STATUSES[0], hours_worked=40)
    out2 = apply(df2.copy(), rules, sim_date)
    assert out2["is_eligible"].iloc[0]


def test_defaults_when_no_eligibility_config():
    sim_date = Timestamp("2025-12-31")
    birth = sim_date - pd.DateOffset(years=21)
    hire = sim_date - pd.DateOffset(months=12)
    # With empty plan_rules, defaults min_age=21, min_service_months=12
    df = make_df(birth, hire, status=ACTIVE_STATUSES[0])
    out = apply(df.copy(), {}, sim_date)
    assert out["is_eligible"].iloc[0]

    # Below default age
    df2 = make_df(sim_date - pd.DateOffset(years=20), hire, status=ACTIVE_STATUSES[0])
    out2 = apply(df2.copy(), {}, sim_date)
    assert not out2["is_eligible"].iloc[0]
