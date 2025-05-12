"""
Test that hazard table regeneration in the projection engine covers all (role, tenure_band) combos present in the snapshot for each year.
"""
import pytest
import pandas as pd
from types import SimpleNamespace
from cost_model.projections.runner import run_projection_engine
from cost_model.utils.columns import EMP_ROLE

def make_snapshot_with_roles_tenures(year, combos):
    # combos: list of (role, tenure_band)
    data = {
        'employee_id': [f"EMP{i}" for i in range(len(combos))],
        EMP_ROLE: [c[0] for c in combos],
        'tenure_band': [c[1] for c in combos],
        'active': [True] * len(combos),
    }
    return pd.DataFrame(data)

def test_hazard_table_covers_all_combos(tmp_path, caplog):
    # Simulate a scenario where new (role, tenure_band) combos appear in year 2
    combos_year1 = [('Engineer', '0-1'), ('Manager', '1-3')]
    combos_year2 = [('Engineer', '0-1'), ('Manager', '1-3'), ('Analyst', '3-5')]
    
    # Initial snapshot only has year 1 combos
    snapshot = make_snapshot_with_roles_tenures(2025, combos_year1)
    
    # Minimal config
    config = SimpleNamespace(
        global_parameters=SimpleNamespace(
            start_year=2025,
            projection_years=2,
            random_seed=1,
            census_template_path=None,
        ),
        plan_rules=SimpleNamespace()
    )
    # Empty event log
    event_log = pd.DataFrame([])
    
    # Patch: monkeypatch run_one_year to inject new combo in year 2
    from cost_model.projections import runner
    orig_run_one_year = runner.run_one_year
    def fake_run_one_year(event_log, prev_snapshot, year, *a, **kw):
        if year == 2026:
            # Add a new Analyst row for year 2
            prev_snapshot = pd.concat([
                prev_snapshot,
                make_snapshot_with_roles_tenures(year, [('Analyst', '3-5')])
            ], ignore_index=True)
        # Return empty event log, unchanged snapshot
        return pd.DataFrame([]), prev_snapshot
    runner.run_one_year = fake_run_one_year
    try:
        with caplog.at_level('WARNING'):
            run_projection_engine(config, snapshot, event_log)
        # Check for warning about missing combo in hazard table for year 2
        found = any('Missing hazard table entries' in r.message and "('Analyst', '3-5')" in r.message for r in caplog.records)
        assert found, "Expected warning for missing hazard table entry for ('Analyst', '3-5') in year 2"
    finally:
        runner.run_one_year = orig_run_one_year
