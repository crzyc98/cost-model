import pandas as pd
import pytest
from datetime import datetime
from cost_model.plan_rules.auto_increase import run as ai_run
from cost_model.config.plan_rules import AutoIncreaseConfig

EVT_ENROLL       = 'EVT_ENROLL'
EVT_AUTO_ENROLL  = 'EVT_AUTO_ENROLL'
EVT_OPT_OUT      = 'EVT_OPT_OUT'
EVT_INCREASE     = 'EVT_INCREASE'

@pytest.fixture
def snap():
    return pd.DataFrame({
        'employee_id': ['A','B','C','D'],
        'employee_deferral_rate': [0.03, 0.05, 0.07, 0.10]
    }).set_index('employee_id')

@pytest.fixture
def evts():
    return pd.DataFrame([
        {'event_time': datetime(2025,1,1),'employee_id':'A','event_type':EVT_ENROLL},
        {'event_time': datetime(2025,1,1),'employee_id':'B','event_type':EVT_AUTO_ENROLL},
        {'event_time': datetime(2025,1,1),'employee_id':'C','event_type':EVT_ENROLL},
        {'event_time': datetime(2025,1,1),'employee_id':'C','event_type':EVT_OPT_OUT},
        {'event_time': datetime(2025,1,1),'employee_id':'D','event_type':EVT_ENROLL},
        {'event_time': datetime(2025,1,1),'employee_id':'D','event_type':EVT_INCREASE},
    ])

def test_increase_applied_and_capped(snap, evts):
    cfg = AutoIncreaseConfig(increase_pct=0.02, cap=0.08)
    out = ai_run(snap, evts, pd.Timestamp('2025-01-01'), cfg)
    df = pd.concat(out, ignore_index=True)
    # A: .03→.05, B: .05→.07, C opted out, D already increased
    assert set(df.employee_id) == {'A','B'}
    assert all(df.value_json.map(lambda j: j['new_rate'] <= 0.08))
