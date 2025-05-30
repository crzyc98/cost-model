#!/usr/bin/env python3
"""
Test script to verify our diagnostic logging is working correctly.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_diagnostic_functions():
    """Test the diagnostic helper functions we added."""
    print("Testing diagnostic helper functions...")
    
    # Import our diagnostic functions
    from cost_model.engines.run_one_year.orchestrator import n_active, check_duplicates, log_headcount_stage
    from cost_model.state.schema import EMP_ID, EMP_ACTIVE
    
    logger = logging.getLogger(__name__)
    
    # Create test data
    test_data = {
        EMP_ID: ['emp_001', 'emp_002', 'emp_003', 'emp_004', 'emp_005'],
        EMP_ACTIVE: [True, True, False, True, True],
        'other_col': [1, 2, 3, 4, 5]
    }
    test_df = pd.DataFrame(test_data)
    
    print(f"Test DataFrame:\n{test_df}")
    
    # Test n_active function
    active_count = n_active(test_df)
    print(f"Active count: {active_count} (expected: 4)")
    assert active_count == 4, f"Expected 4 active, got {active_count}"
    
    # Test log_headcount_stage function
    print("\nTesting log_headcount_stage...")
    log_headcount_stage(test_df, "TEST_STAGE", 2025, logger)
    
    # Test duplicate detection (should pass)
    print("\nTesting duplicate detection (should pass)...")
    try:
        check_duplicates(test_df, "TEST_NO_DUPES", logger)
        print("✅ No duplicates detected correctly")
    except AssertionError as e:
        print(f"❌ Unexpected duplicate detection: {e}")
        return False
    
    # Test duplicate detection (should fail)
    print("\nTesting duplicate detection (should fail)...")
    duplicate_data = test_data.copy()
    duplicate_data[EMP_ID].append('emp_001')  # Add duplicate
    duplicate_data[EMP_ACTIVE].append(True)
    duplicate_data['other_col'].append(6)
    duplicate_df = pd.DataFrame(duplicate_data)
    
    try:
        check_duplicates(duplicate_df, "TEST_WITH_DUPES", logger)
        print("❌ Duplicates not detected when they should have been")
        return False
    except AssertionError as e:
        print(f"✅ Duplicates correctly detected: {e}")
    
    print("\n✅ All diagnostic function tests passed!")
    return True


def test_simple_simulation():
    """Test a simple one-year simulation with diagnostic logging."""
    print("\nTesting simple simulation with diagnostic logging...")
    
    try:
        # Import required modules
        from cost_model.engines.run_one_year import run_one_year
        from cost_model.state.snapshot import build_full
        from cost_model.utils.columns import EMP_ID
        
        # Create minimal test census
        census_data = {
            EMP_ID: [f'emp_{i:03d}' for i in range(1, 21)],  # 20 employees
            'employee_birth_date': ['1990-01-01'] * 20,
            'employee_hire_date': ['2020-01-01'] * 20,
            'employee_gross_compensation': [50000.0] * 20,
            'employee_level': [1] * 20,
        }
        census_df = pd.DataFrame(census_data)
        
        # Build initial snapshot
        snapshot = build_full(census_df, 2025)
        
        # Create minimal hazard table
        hazard_data = {
            'simulation_year': [2025],
            'employee_level': [1],
            'employee_tenure_band': ['<1'],
            'term_rate': [0.15],
            'merit_raise_pct': [0.03],
            'cola_pct': [0.02],
            'new_hire_termination_rate': [0.25],
            'promotion_rate': [0.05]
        }
        hazard_df = pd.DataFrame(hazard_data)
        
        # Create minimal global params
        class MockGlobalParams:
            target_growth = 0.03
            min_eligibility_age = 21
            min_service_months = 12
        
        global_params = MockGlobalParams()
        
        # Create minimal plan rules
        plan_rules = {
            'contributions': {'enabled': True},
            'match': {'tiers': [{'match_rate': 0.5, 'cap_deferral_pct': 0.06}]},
            'non_elective': {'rate': 0.0},
            'irs_limits': {}
        }
        
        # Setup RNG
        rng = np.random.default_rng(42)
        
        # Create empty event log
        event_log = pd.DataFrame()
        
        print(f"Starting simulation with {len(snapshot)} employees...")
        
        # Run one year simulation
        new_events, final_snapshot = run_one_year(
            event_log=event_log,
            prev_snapshot=snapshot,
            year=2025,
            global_params=global_params,
            plan_rules=plan_rules,
            hazard_table=hazard_df,
            rng=rng,
            deterministic_term=True
        )
        
        print(f"Simulation completed!")
        print(f"Final snapshot size: {len(final_snapshot)}")
        print(f"New events generated: {len(new_events)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=== Testing Diagnostic Logging Implementation ===")
    
    # Test 1: Diagnostic functions
    if not test_diagnostic_functions():
        print("❌ Diagnostic function tests failed")
        exit(1)
    
    # Test 2: Simple simulation
    if not test_simple_simulation():
        print("❌ Simulation test failed")
        exit(1)
    
    print("\n🎯 All tests passed! Diagnostic logging is working correctly.")
