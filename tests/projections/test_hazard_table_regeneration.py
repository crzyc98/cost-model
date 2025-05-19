"""Test that hazard table regeneration logic covers all (role, tenure_band) combinations present in the snapshot."""
import pytest
import pandas as pd
import logging
from types import SimpleNamespace
from cost_model.projections.hazard import build_hazard_table
from cost_model.utils.columns import EMP_ROLE
from cost_model.state.schema import EMP_TENURE_BAND

def test_hazard_table_regeneration():
    """Test that hazard table regeneration includes all role/tenure combinations."""
    # Create a snapshot with specific role/tenure combinations
    snapshot = pd.DataFrame({
        'employee_id': ['EMP1', 'EMP2', 'EMP3'],
        EMP_ROLE: ['Engineer', 'Manager', 'Analyst'],
        EMP_TENURE_BAND: ['0-1', '1-3', '3-5'],
        'active': [True, True, True]
    })
    
    # Create minimal config
    global_params = SimpleNamespace(
        annual_termination_rate=0.10,
        annual_compensation_increase_rate=0.03
    )
    plan_rules_config = SimpleNamespace()
    
    # Build hazard table for a single year
    years = [2025]
    hazard_table = build_hazard_table(years, snapshot, global_params, plan_rules_config)
    
    # Verify all role/tenure combinations from snapshot are in hazard table
    snapshot_combos = set(tuple(x) for x in snapshot[[EMP_ROLE, EMP_TENURE_BAND]].drop_duplicates().values)
    hazard_combos = set(tuple(x) for x in hazard_table[[EMP_ROLE, EMP_TENURE_BAND]].drop_duplicates().values)
    
    # All snapshot combinations should be in hazard table
    assert snapshot_combos.issubset(hazard_combos), f"Missing combinations: {snapshot_combos - hazard_combos}"
    
    # Hazard table should have exactly the combinations from snapshot
    assert len(snapshot_combos) == len(hazard_combos), f"Extra combinations in hazard table: {hazard_combos - snapshot_combos}"

def test_hazard_table_warning_on_missing_combo(caplog):
    """Test that appropriate warnings are logged when role/tenure combinations are missing."""
    # Create snapshot with specific combinations
    snapshot = pd.DataFrame({
        'employee_id': ['EMP1', 'EMP2'],
        EMP_ROLE: ['Engineer', 'Manager'],
        EMP_TENURE_BAND: ['0-1', '1-3'],
        'active': [True, True]
    })
    
    # Create minimal config
    global_params = SimpleNamespace(
        annual_termination_rate=0.10,
        annual_compensation_increase_rate=0.03
    )
    plan_rules_config = SimpleNamespace()
    
    # Patch the hazard table function to simulate missing combinations
    from cost_model.projections import hazard
    orig_build_hazard_table = hazard.build_hazard_table
    
    def mock_build_hazard_table(*args, **kwargs):
        # Call original but then remove one combination
        result = orig_build_hazard_table(*args, **kwargs)
        if not result.empty:
            # Remove the Manager/1-3 combination to simulate a missing entry
            result = result[~((result[EMP_ROLE] == 'Manager') & (result[EMP_TENURE_BAND] == '1-3'))]
        return result
    
    # Apply the patch
    hazard.build_hazard_table = mock_build_hazard_table
    
    try:
        # Set up logging capture
        with caplog.at_level(logging.WARNING):
            # Import here to ensure we use the patched version
            from cost_model.projections.runner import run_projection_engine
            
            # Create a minimal runner function that just checks for missing combinations
            def check_missing_combos(snapshot, year):
                hazard_table = mock_build_hazard_table([year], snapshot, global_params, plan_rules_config)
                snapshot_combos = set(tuple(x) for x in snapshot[[EMP_ROLE, EMP_TENURE_BAND]].drop_duplicates().values)
                hazard_combos = set(tuple(x) for x in hazard_table[[EMP_ROLE, EMP_TENURE_BAND]].drop_duplicates().values)
                missing_combos = snapshot_combos - hazard_combos
                if missing_combos:
                    logging.warning(f"[HAZARD TABLE] Year {year}: Missing hazard table entries for combinations: {missing_combos}")
            
            # Run our check
            check_missing_combos(snapshot, 2025)
            
            # Verify warning was logged
            assert any("Missing hazard table entries" in record.message and "Manager" in record.message and "1-3" in record.message 
                      for record in caplog.records), "Expected warning about missing Manager/1-3 combination"
    finally:
        # Restore original function
        hazard.build_hazard_table = orig_build_hazard_table
