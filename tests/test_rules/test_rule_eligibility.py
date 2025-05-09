import pandas as pd
from utils.rules.eligibility import apply, agent_is_eligible
from utils.constants import ACTIVE_STATUSES


def test_apply_and_agent_helper_consistency():
    sim_end = pd.Timestamp("2025-12-31")
    eligibility_config = {
        "min_age": 21,
        "min_service_months": 12,
        "min_hours_worked": 40,
    }
    plan_rules = {"eligibility": eligibility_config}

    df = pd.DataFrame(
        [
            {
                "birth_date": sim_end - pd.DateOffset(years=22),
                "hire_date": sim_end - pd.DateOffset(months=13),
                "status": ACTIVE_STATUSES[0],
                "hours_worked": 40,
            },
            {
                "birth_date": sim_end - pd.DateOffset(years=30),
                "hire_date": sim_end - pd.DateOffset(months=24),
                "status": ACTIVE_STATUSES[0],
                "hours_worked": 39,
            },
            {
                "birth_date": sim_end - pd.DateOffset(years=30),
                "hire_date": sim_end - pd.DateOffset(months=24),
                "status": "Terminated",
                "hours_worked": 40,
            },
            {
                "birth_date": sim_end - pd.DateOffset(years=20),
                "hire_date": sim_end - pd.DateOffset(months=24),
                "status": ACTIVE_STATUSES[0],
                "hours_worked": 40,
            },
        ]
    )
    df["birth_date"] = pd.to_datetime(df["birth_date"])
    df["hire_date"] = pd.to_datetime(df["hire_date"])

    df_result = apply(df.copy(), plan_rules, sim_end)
    for idx, row in df.iterrows():
        expected = agent_is_eligible(
            row["birth_date"],
            row["hire_date"],
            row["status"],
            row["hours_worked"],
            eligibility_config,
            sim_end,
        )
        assert df_result.loc[idx, "is_eligible"] == expected
