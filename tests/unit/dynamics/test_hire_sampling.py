import pandas as pd
import numpy as np
import pytest
import json
from cost_model.engines.hire import run
from cost_model.state.event_log import EVT_TERM
from cost_model.state.schema import EMP_ID, EMP_ROLE, EMP_GROSS_COMP, EMP_HIRE_DATE

@pytest.fixture
def dummy_snapshot():
    return pd.DataFrame({
        EMP_ID: ['E1','E2','E3'],
        EMP_ROLE: ['Staff','Staff','Manager'],
        EMP_GROSS_COMP: [100_000, 120_000, 150_000],
        EMP_HIRE_DATE: pd.to_datetime(['2020-01-01']*3),
        EMP_BIRTH_DATE: pd.to_datetime(['1980-06-15', '1985-09-20', '1975-01-30'])
    })

@pytest.fixture
def term_events():
    # two terminations: one Staff, one Manager
    now = pd.Timestamp('2025-06-01')
    return pd.DataFrame([
        {'event_id':'t1','event_time':now,'employee_id':'E1','event_type':EVT_TERM},
        {'event_id':'t2','event_time':now,'employee_id':'E3','event_type':EVT_TERM},
    ])

from types import SimpleNamespace
from cost_model.state.schema import EMP_BIRTH_DATE

def test_replacement_hire_from_terms(dummy_snapshot, term_events):
    global_params = SimpleNamespace(replacement_hire_premium=0.05, replacement_hire_age_sd=1)
    hires, comps = run(
        snapshot=dummy_snapshot,
        hires_to_make=2,
        hazard_slice=pd.DataFrame([{'simulation_year':2025,'new_hire_termination_rate':0.0,'comp_raise_pct':0.03,'term_rate':0, 'growth_rate':0,'cfg':None}]),
        rng=np.random.default_rng(42),
        census_template_path='',
        global_params=global_params,
        terminated_events=term_events
    )
    # Expect 2 hires
    assert len(hires) == 2
    # Their clone_of in JSON should match one of the term IDs
    for row in hires['value_json'].apply(json.loads):
        assert row['clone_of'] in ['E1','E3']
    # Test premium: new salary should be 5% higher than original
    original = dummy_snapshot.set_index(EMP_ID)[EMP_GROSS_COMP]
    # Join hires and comps on employee_id to get full_year salary from value_json for each hire
    merged = hires.merge(comps[[EMP_ID, 'value_json']], left_on=EMP_ID, right_on=EMP_ID, how='left')
    for _, hire in merged.iterrows():
        if hire['value_json_y'] is None:
            continue
        clone_of = json.loads(hire['value_json_x'])['clone_of']
        if clone_of in original:
            comp_json = json.loads(hire['value_json_y'])
            full_year = comp_json.get('full_year')
            # Allow some float tolerance
            assert abs(full_year / (original[clone_of] * 1.05) - 1) < 0.01
    # Test age jitter: birth date should differ from original by ~1yr sd
    original_bd = dummy_snapshot.set_index(EMP_ID)[EMP_BIRTH_DATE] if EMP_BIRTH_DATE in dummy_snapshot else None
    if EMP_BIRTH_DATE in hires:
        for _, hire in hires.iterrows():
            clone_of = json.loads(hire['value_json'])['clone_of']
            if original_bd is not None and clone_of in original_bd:
                orig = pd.to_datetime(original_bd[clone_of])
                new = pd.to_datetime(hire[EMP_BIRTH_DATE])
                # Should not be exactly the same, but within a couple years
                assert abs((new - orig).days) < 3*366
