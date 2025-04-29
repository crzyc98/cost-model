import pandas as pd
from utils.rules import eligibility

def test_year1_eligible_matches_headcount():
    # Load Year-1 census and plan config
    census = pd.read_csv('data/census_data.csv')
    plan_config = ...  # Load your plan rules/config dict

    eligible = census.apply(lambda row: eligibility.is_eligible(row, plan_config['eligibility']), axis=1)
    eligible_count = eligible.sum()
    headcount = len(census)
    # Allow off-by-1 for edge cases (e.g., hire on Jan 1)
    assert abs(eligible_count - headcount) <= 1