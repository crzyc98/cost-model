import pandas as pd
import pytest
from cost_model.plan_rules.eligibility_events import run
from cost_model.config.plan_rules import EligibilityEventsConfig

def test_single_milestone_crossing():
    # employee hired 13 months ago; prev run was at 11 months
    snap = pd.DataFrame({
        'employee_id': ['E'],
        'employee_hire_date': [pd.Timestamp('2023-02-01')],
    }).set_index('employee_id')
    cfg = EligibilityEventsConfig(
        milestone_months=[12],
        milestone_years=[],
        event_type_map={12: 'EVT_1YR'}
    )
    as_of = pd.Timestamp('2024-03-01')
    prev = pd.Timestamp('2024-01-01')
    out = run(snap, pd.DataFrame(), as_of, prev, cfg)
    df = pd.concat(out, ignore_index=True)
    assert len(df) == 1
    assert df.event_type.iloc[0] == 'EVT_1YR'
    assert df.employee_id.iloc[0] == 'E'

def test_no_duplicate_if_not_crossed():
    # same employee, but both prev and as_of after milestone
    snap = pd.DataFrame({
        'employee_id': ['E'],
        'employee_hire_date': [pd.Timestamp('2023-01-01')],
    }).set_index('employee_id')
    cfg = EligibilityEventsConfig(
        milestone_months=[12],
        milestone_years=[],
        event_type_map={12: 'EVT_1YR'}
    )
    prev = pd.Timestamp('2024-03-01')  # 14 months
    as_of = pd.Timestamp('2024-04-01') # 15 months
    out = run(snap, pd.DataFrame(), as_of, prev, cfg)
    assert out == []

def test_multiple_milestones_and_employees():
    snap = pd.DataFrame({
        'employee_id': ['A', 'B'],
        'employee_hire_date': [pd.Timestamp('2022-01-01'), pd.Timestamp('2023-01-01')],
    }).set_index('employee_id')
    cfg = EligibilityEventsConfig(
        milestone_months=[12, 24],
        milestone_years=[3],
        event_type_map={12: 'EVT_1YR', 24: 'EVT_2YR', 36: 'EVT_3YR'}
    )
    prev = pd.Timestamp('2023-12-01')
    as_of = pd.Timestamp('2024-02-01')
    out = run(snap, pd.DataFrame(), as_of, prev, cfg)
    df = pd.concat(out, ignore_index=True)
    # A: 23 at prev, 25 at as_of (should fire 24)
    # B: 11 at prev, 13 at as_of (should fire 12)
    assert set(df.employee_id) == {'A', 'B'}
    assert set(df.event_type) == {'EVT_1YR', 'EVT_2YR'}

def test_milestone_years():
    snap = pd.DataFrame({
        'employee_id': ['C'],
        'employee_hire_date': [pd.Timestamp('2020-01-01')],
    }).set_index('employee_id')
    cfg = EligibilityEventsConfig(
        milestone_months=[],
        milestone_years=[3, 5],
        event_type_map={36: 'EVT_3YR', 60: 'EVT_5YR'}
    )
    prev = pd.Timestamp('2022-12-01')  # 35 months
    as_of = pd.Timestamp('2023-02-01') # 37 months
    out = run(snap, pd.DataFrame(), as_of, prev, cfg)
    df = pd.concat(out, ignore_index=True)
    # Should only fire 36 (3yr) for C
    assert set(df.event_type) == {'EVT_3YR'}
    assert set(df.employee_id) == {'C'}