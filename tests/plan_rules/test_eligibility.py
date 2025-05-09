import pandas as pd
from datetime import date, timedelta
import pytest
from cost_model.plan_rules.eligibility import run as eligibility_run
from cost_model.config.plan_rules import EligibilityConfig

def make_snapshot(records):
    return pd.DataFrame(records)

def test_under_min_age_no_event():
    today = date(2025, 1, 1)
    snap = make_snapshot([
        {'employee_id': 1, 'employee_birth_date': date(2010, 1, 1), 'employee_hire_date': date(2020, 1, 1)},
    ])
    cfg = EligibilityConfig(min_age=21, min_service_months=12)
    events = eligibility_run(snap, today, cfg)
    assert len(events) == 0

def test_under_min_service_no_event():
    today = date(2025, 1, 1)
    snap = make_snapshot([
        {'employee_id': 2, 'employee_birth_date': date(1990, 1, 1), 'employee_hire_date': date(2024, 7, 1)},
    ])
    cfg = EligibilityConfig(min_age=21, min_service_months=12)
    events = eligibility_run(snap, today, cfg)
    assert len(events) == 0

def test_exactly_at_threshold_event():
    today = date(2025, 1, 1)
    snap = make_snapshot([
        {'employee_id': 3, 'employee_birth_date': date(2004, 1, 1), 'employee_hire_date': date(2024, 1, 1)},
    ])
    cfg = EligibilityConfig(min_age=21, min_service_months=12)
    events = eligibility_run(snap, today, cfg)
    assert len(events) == 1
    evt = events[0]
    assert evt['employee_id'].iloc[0] == 3
    assert evt['event_type'].iloc[0] == 'EVT_ELIGIBLE'

def test_mixed_eligibility():
    today = date(2025, 1, 1)
    snap = make_snapshot([
        # Eligible
        {'employee_id': 4, 'employee_birth_date': date(1990, 1, 1), 'employee_hire_date': date(2020, 1, 1)},
        # Too young
        {'employee_id': 5, 'employee_birth_date': date(2010, 1, 1), 'employee_hire_date': date(2020, 1, 1)},
        # Too new
        {'employee_id': 6, 'employee_birth_date': date(1990, 1, 1), 'employee_hire_date': date(2024, 6, 1)},
    ])
    cfg = EligibilityConfig(min_age=21, min_service_months=12)
    events = eligibility_run(snap, today, cfg)
    assert len(events) == 1
    assert events[0]['employee_id'].iloc[0] == 4
    assert events[0]['event_type'].iloc[0] == 'EVT_ELIGIBLE'
