import pytest
import pandas as pd
from pandas import Timestamp

from utils.rules.auto_enrollment import apply as apply_ae
from utils.columns import (
    IS_ELIGIBLE,
    IS_PARTICIPATING,
    EMP_DEFERRAL_RATE,
    AE_OPTED_OUT,
    ELIGIBILITY_ENTRY_DATE,
    AE_WINDOW_START,
    AE_WINDOW_END,
    STATUS_COL,
)
from utils.constants import ACTIVE_STATUSES


def test_ae_window_and_flags():
    """
    Smoke test for auto-enrollment window and participation outcome.
    """
    eligibility_date = Timestamp('2025-06-15')
    df = pd.DataFrame({
        IS_ELIGIBLE: [True],
        IS_PARTICIPATING: [False],
        EMP_DEFERRAL_RATE: [0.0],
        AE_OPTED_OUT: [False],
        ELIGIBILITY_ENTRY_DATE: [eligibility_date],
        STATUS_COL: [ACTIVE_STATUSES[0]],  # required by AE logic
    })
    plan_rules = {
        'auto_enrollment': {
            'enabled': True,
            'default_rate': 0.05,
            'window_days': 10,
            'outcome_distribution': {'prob_stay_default': 1.0}
        }
    }
    start_date = Timestamp('2025-01-01')
    end_date = Timestamp('2025-12-31')
    out = apply_ae(df.copy(), plan_rules, start_date, end_date)

    # AE window dates
    assert out[AE_WINDOW_START].iloc[0] == eligibility_date
    assert out[AE_WINDOW_END].iloc[0] == eligibility_date + pd.Timedelta(days=10)

    # Ensure no opted out
    assert out[AE_OPTED_OUT].iloc[0] == False

    # With stay_default=1, participant should be enrolled
    assert out[IS_PARTICIPATING].iloc[0] == True
