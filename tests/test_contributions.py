import pandas as pd
import pytest
from utils.rules.contributions import apply

def test_single_tier_match_and_pre_tax():
    """
    Test basic pre-tax and single-tier employer match contributions.
    """
    # Input DataFrame
    df = pd.DataFrame({
        'gross_compensation': [100000],
        'deferral_rate': [0.06],
        'birth_date': [pd.to_datetime('1990-06-15')],
        'status': ['Active'],
        'hire_date': [pd.to_datetime('2020-01-01')]
    })
    # Plan rules with generous IRS limits and one match tier at 50% up to 6%
    plan_rules = {
        'irs_limits': {
            2025: {
                'comp_limit': 1e9,
                'deferral_limit': 1e9,
                'catch_up': 0,
                'overall_limit': 1e9,
                'catchup_eligibility_age': 50
            }
        },
        'employer_match': {
            'tiers': [
                {'cap_deferral_pct': 0.06, 'match_rate': 0.5}
            ]
        }
    }
    # Run apply
    result = apply(df.copy(), plan_rules, 2025, '2025-01-01', '2025-12-31')
    # Expected contributions
    expected_pre_tax = 100000 * 0.06
    expected_match = expected_pre_tax * 0.5
    assert result['pre_tax_contributions'].iloc[0] == pytest.approx(expected_pre_tax)
    assert result['employer_match_contribution'].iloc[0] == pytest.approx(expected_match)
