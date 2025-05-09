# tests/plan_rules/test_plan_rules_integration.py

import pandas as pd
import pytest
from datetime import datetime
from cost_model.plan_rules.eligibility import run as eligibility_run
from cost_model.plan_rules.enrollment  import run as enrollment_run
from cost_model.plan_rules.contributions import run as contributions_run
from cost_model.plan_rules.auto_increase import run as auto_increase_run
from cost_model.config.plan_rules import (
    EligibilityConfig, EnrollmentConfig,
    ContributionConfig, AutoIncreaseConfig
)

@pytest.fixture
def tiny_snapshot():
    # Two employees: one young (ineligible), one old (eligible)
    return pd.DataFrame({
        'employee_id': ['X','Y'],
        'employee_birth_date': [pd.Timestamp('2005-01-01'), pd.Timestamp('1970-01-01')],
        'employee_hire_date' : [pd.Timestamp('2024-01-01'), pd.Timestamp('2020-01-01')],
        'employee_gross_compensation': [50_000, 100_000],
        'employee_deferral_rate':        [0.00,   0.05],
    })

@pytest.fixture
def base_events():
    # Everyone gets eligibility on Jan 1
    return pd.DataFrame([
      {'event_time': datetime(2025,1,1),'employee_id':'X','event_type':'EVT_ELIGIBLE'},
      {'event_time': datetime(2025,1,1),'employee_id':'Y','event_type':'EVT_ELIGIBLE'},
    ])

def test_full_plan_rules_pipeline(tiny_snapshot, base_events):
    as_of = pd.Timestamp('2025-01-01')
    # config stubs
    elig_cfg = EligibilityConfig(min_age=20, min_service_months=0)
    from cost_model.config.plan_rules import AutoEnrollmentConfig
    enrol_cfg = EnrollmentConfig(
        auto_enrollment=AutoEnrollmentConfig(enabled=False, default_rate=0.02, window_days=0),
        voluntary_enrollment_rate=1.0,
        default_rate=0.01
    )
    contr_cfg = ContributionConfig(
        default_rate=0.0,
        employer_match_tiers=[{'match_rate':1.0,'cap_pct':0.01}]
    )
    ai_cfg = AutoIncreaseConfig(enabled=True, increase_pct=0.01, cap_pct=0.10)

    # Start with no events, only eligibility engine output
    events = pd.DataFrame(columns=[
        "event_id", "event_time", "employee_id",
        "event_type", "value_num", "value_json", "meta"
    ])
    snap = tiny_snapshot.copy()
    # ensure we index by employee_id
    if snap.index.name != "employee_id":
        snap = snap.set_index("employee_id", drop=False)

    # 1. Eligibility
    evs = eligibility_run(snap, as_of, elig_cfg)
    events = pd.concat(evs, ignore_index=True)
    print('After eligibility:', set(events.event_type))
    print('Events before enrollment_run:')
    print(events)

    # 2. Enrollment
    evs = enrollment_run(snap, events, as_of, enrol_cfg)
    print('Enrollment events returned:', [df.to_dict('records') for df in evs])
    events = pd.concat([events, *evs], ignore_index=True)
    print('After enrollment:', set(events.event_type))

    # 3. Contributions
    evs = contributions_run(snap, events, as_of, contr_cfg)
    events = pd.concat([events, evs], ignore_index=True)
    print('After contributions:', set(events.event_type))

    # 4. Auto-Increase
    evs = auto_increase_run(snap, events, as_of, ai_cfg)
    events = pd.concat([events, *evs], ignore_index=True)
    print('After auto-increase:', set(events.event_type))

    # Assertions:
    types = set(events.event_type)
    # must have at least one of each
    assert {'EVT_ELIGIBLE','EVT_ENROLL','EVT_CONTRIB','EVT_AUTO_INCREASE'}.issubset(types)
    # no duplicate events for same emp/type/time
    dup = events.duplicated(subset=['employee_id','event_type','event_time']).any()
    assert not dup
    # headcount unchanged
    assert set(events.employee_id.unique()) == set(snap.index)
