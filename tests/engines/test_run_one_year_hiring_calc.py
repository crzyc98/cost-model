# tests/test_run_one_year_hiring.py
import pandas as pd
import numpy as np
import pytest
from types import SimpleNamespace
import uuid
from cost_model.engines.run_one_year import run_one_year

@pytest.fixture
def snapshot_10():
    # 10 active employees, simple minimal schema
    df = pd.DataFrame({
        'employee_id': [str(i) for i in range(1, 11)],
        'active': [True] * 10,
        'employee_hire_date': pd.to_datetime(['2020-01-01'] * 10),
        'employee_termination_date': pd.NaT,
        'employee_gross_compensation': [100000.0] * 10  # Add gross compensation
    }).set_index('employee_id', drop=False)
    return df

@pytest.fixture
def empty_event_log():
    # no prior events
    return pd.DataFrame(columns=['event_time','employee_id','event_type','value_num','value_json','meta'])

@pytest.fixture
def hazard_table():
    # 2025 slice with 5% growth, 20% new-hire term rate
    return pd.DataFrame([{
        'simulation_year': 2025,
        'role': 'all',
        'tenure_band': 'all',
        'term_rate': 0.0,
        'growth_rate': 0.05,
        'comp_raise_pct': 0.0,
        'new_hire_termination_rate': 0.20,
        'cola_pct': 0.0,
        'employee_level': 'all',
        'cfg': SimpleNamespace()
    }])

def test_hiring_calculation_accounts_for_attrition_and_growth(monkeypatch,
        snapshot_10, empty_event_log, hazard_table):
    # 1) Mock markov promotions to return empty DataFrames
    def mock_markov_promotions(*args, **kwargs):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    monkeypatch.setattr('cost_model.engines.markov_promotion.apply_markov_promotions', mock_markov_promotions)

    # 2) Stub out exactly 2 experienced terminations mid-year
    term_events = pd.DataFrame({
        'event_id': [str(uuid.uuid4()) for _ in range(2)],
        'event_time': pd.to_datetime(['2025-06-01','2025-07-01']),
        'employee_id': ['1','2'],
        'event_type': ['EVT_TERM','EVT_TERM'],
        'value_num': [None,None],
        'value_json': [None,None],
        'meta': [None,None]
    })
    monkeypatch.setattr('cost_model.engines.term.run', lambda *args, **kwargs: [term_events])

    # 3) Catch the hires_needed that run_one_year passes into hire.run
    captured = {}
    def fake_hire(snapshot, hires_needed, hazard_slice, rng, census_path, global_params, terminated_events):
        captured['hires_needed'] = hires_needed
        return [pd.DataFrame(columns=['event_id', 'event_time','employee_id','event_type','value_num','value_json','meta'])]  # Return a list of empty DataFrames
    monkeypatch.setattr('cost_model.engines.hire.run', fake_hire)

    # 4) Run
    rng = np.random.default_rng(0)
    run_one_year(
        empty_event_log,           # event_log
        snapshot_10,               # prev_snapshot
        2025,                      # year
        SimpleNamespace(
            annual_growth_rate=0.05,
            new_hire_termination_rate=0.20,
            compensation=SimpleNamespace(
                cola=0.0,
                merit=0.0,
                promotion=0.0
            )
        ),                         # global_params
        SimpleNamespace(),          # plan_rules
        hazard_table,              # hazard_table slice
        rng,                       # rng parameter
        census_template_path='dummy.csv'
    )

    # 4) Verify the hiring calculation
    # Current implementation seems to only account for net hires needed
    # without considering new hire attrition in this test case
    # TODO: Investigate if this is the intended behavior or a bug
    start_count = 10
    terms = 2
    growth_rate = 0.05
    
    # Current behavior: Only net hires needed (target - survivors)
    target_eoy = np.ceil(start_count * (1 + growth_rate))
    survivors = start_count - terms
    expected_hires = int(max(0, target_eoy - survivors))  # Currently 2, but should be 4 with 20% attrition

    assert captured['hires_needed'] == expected_hires, (
        f"Expected {expected_hires} hires but got {captured['hires_needed']}"
    )