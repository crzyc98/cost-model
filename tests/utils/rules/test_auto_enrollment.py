import pandas as pd
import numpy as np
import pytest
from pandas import Timestamp
from utils.rules.auto_enrollment import apply

# Helper to build a minimal eligible DataFrame

def make_df(status='Active', is_participating=False, deferral_rate=0.0, ae_opted_out=False, entry_date=None):
    if entry_date is None:
        entry_date = Timestamp('2025-01-01')
    data = {
        'status': [status],
        'is_eligible': [True],
        'is_participating': [is_participating],
        'deferral_rate': [deferral_rate],
        'ae_opted_out': [ae_opted_out],
        'eligibility_entry_date': [entry_date]
    }
    return pd.DataFrame(data)

YEAR_START = Timestamp('2025-01-01')
YEAR_END = Timestamp('2025-12-31')


def test_ae_disabled():
    df = make_df()
    rules = {'auto_enrollment': {'enabled': False}}
    out = apply(df.copy(), rules, YEAR_START, YEAR_END)
    # Should be unchanged
    pd.testing.assert_frame_equal(out, df)


def test_missing_columns_skips():
    df = pd.DataFrame({'foo': [1]})
    rules = {'auto_enrollment': {'enabled': True}}
    out = apply(df.copy(), rules, YEAR_START, YEAR_END)
    # Missing required cols: should skip but preserve existing data
    assert 'foo' in out.columns
    assert out['foo'].iloc[0] == 1


def test_proactive_enrollment_full_prob():
    df = make_df(is_participating=False)
    rules = {
        'auto_enrollment': {
            'enabled': True,
            'proactive_enrollment_probability': 1.0,
            'proactive_rate_range': [0.05, 0.05],
            'default_rate': 0.02
        }
    }
    np.random.seed(0)
    out = apply(df.copy(), rules, YEAR_START, YEAR_END)
    # Should have proactively enrolled
    assert out['proactive_enrolled'].iloc[0]
    assert out['is_participating'].iloc[0]
    # Rate should be within the range (fixed to 0.05)
    assert out['deferral_rate'].iloc[0] == pytest.approx(0.05)
    # Enrollment and first_contribution dates set to entry_date
    assert out['enrollment_date'].iloc[0] == out['eligibility_entry_date'].iloc[0]
    assert out['first_contribution_date'].iloc[0] == out['eligibility_entry_date'].iloc[0]


def test_re_enroll_existing():
    # Under default rate
    df = make_df(is_participating=True, deferral_rate=0.01)
    rules = {
        'auto_enrollment': {
            'enabled': True,
            'default_rate': 0.04,
            're_enroll_existing': True
        }
    }
    out = apply(df.copy(), rules, YEAR_START, YEAR_END)
    assert out['auto_reenrolled'].iloc[0]
    assert out['deferral_rate'].iloc[0] == pytest.approx(0.04)
    assert out['enrollment_method'].iloc[0] == 'AE'


def test_ae_window_closure_and_outcomes():
    # Test that window_closed_during_year becomes True
    date = Timestamp('2025-06-15')
    df = make_df(entry_date=date)
    rules = {
        'auto_enrollment': {
            'enabled': True,
            'default_rate': 0.03,
            'ae_outcome_distribution': {'stay_default': 1.0}
        }
    }
    np.random.seed(1)
    out = apply(df.copy(), rules, YEAR_START, YEAR_END)
    assert out['window_closed_during_year'].iloc[0]
    # Since no participations before window, should auto_enroll with stay_default
    assert out['auto_enrolled'].iloc[0] or out['proactive_enrolled'].iloc[0]
