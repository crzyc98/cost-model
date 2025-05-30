#!/usr/bin/env python3
"""
Test script to run a full simulation and capture our diagnostic logging output.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Setup logging to capture our diagnostic output
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('diagnostic_test.log')
    ]
)

def run_diagnostic_simulation():
    """Run a simulation with our diagnostic logging enabled."""
    print("=== Running Diagnostic Simulation Test ===")
    
    try:
        # Import required modules
        from cost_model.engines.run_one_year import run_one_year
        from cost_model.state.snapshot_build import build_full
        from cost_model.state.event_log import EVENT_COLS
        from cost_model.state.schema import EMP_ID, EMP_ACTIVE, EMP_HIRE_DATE, EMP_BIRTH_DATE, EMP_GROSS_COMP
        
        # Create test census data (100 employees to match the targeting test)
        census_data = []
        for i in range(1, 101):  # 100 employees
            census_data.append({
                'event_id': f'hire_{i:03d}',
                'event_time': '2024-01-01',
                'employee_id': f'emp_{i:03d}',
                'event_type': 'EVT_HIRE',
                'value_num': 50000.0 + (i * 100),  # Varying salaries
                'value_json': f'{{"level": {(i % 5) + 1}, "birth_date": "1990-01-01"}}',
                'meta': f'Initial hire for emp_{i:03d}',
                'job_level_source': 'hire'
            })
        
        events_df = pd.DataFrame(census_data, columns=EVENT_COLS)
        
        # Build initial snapshot
        print("Building initial snapshot...")
        snapshot = build_full(events_df, 2025)
        print(f"Initial snapshot built with {len(snapshot)} employees")
        
        # Create hazard table with exact targeting parameters
        hazard_data = {
            'simulation_year': [2025] * 5,
            'employee_level': [1, 2, 3, 4, 5],
            'employee_tenure_band': ['1-3'] * 5,
            'term_rate': [0.18] * 5,  # 18% experienced termination rate
            'merit_raise_pct': [0.03] * 5,
            'cola_pct': [0.02] * 5,
            'new_hire_termination_rate': [0.25] * 5,  # 25% new hire termination rate
            'promotion_rate': [0.05] * 5
        }
        hazard_df = pd.DataFrame(hazard_data)
        
        # Create global params with exact targeting
        class MockGlobalParams:
            target_growth = 0.03  # 3% growth target
            min_eligibility_age = 21
            min_service_months = 12
        
        global_params = MockGlobalParams()
        
        # Create plan rules
        plan_rules = {
            'contributions': {'enabled': True},
            'match': {'tiers': [{'match_rate': 0.5, 'cap_deferral_pct': 0.06}]},
            'non_elective': {'rate': 0.0},
            'irs_limits': {}
        }
        
        # Setup RNG
        rng = np.random.default_rng(42)
        
        # Create empty event log
        event_log = pd.DataFrame(columns=EVENT_COLS)
        
        print(f"\nStarting simulation with diagnostic logging...")
        print(f"Initial employees: {len(snapshot)}")
        print(f"Target growth: {global_params.target_growth:.1%}")
        print(f"Expected target headcount: {round(len(snapshot) * (1 + global_params.target_growth))}")
        
        # Run one year simulation with our diagnostic logging
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
        
        # Analyze results
        final_active = final_snapshot[EMP_ACTIVE].sum() if EMP_ACTIVE in final_snapshot.columns else len(final_snapshot)
        expected_target = round(len(snapshot) * (1 + global_params.target_growth))
        
        print(f"\n=== SIMULATION RESULTS ===")
        print(f"Initial employees: {len(snapshot)}")
        print(f"Final total rows: {len(final_snapshot)}")
        print(f"Final active employees: {final_active}")
        print(f"Expected target: {expected_target}")
        print(f"Difference from target: {final_active - expected_target}")
        print(f"New events generated: {len(new_events)}")
        
        # Check if we hit our target
        if abs(final_active - expected_target) <= 1:  # Allow 1 employee tolerance
            print("✅ TARGET ACHIEVED! Exact targeting is working correctly.")
            return True
        else:
            print(f"❌ TARGET MISSED! Off by {final_active - expected_target} employees.")
            return False
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_diagnostic_simulation()
    
    print(f"\n=== DIAGNOSTIC LOG ANALYSIS ===")
    print("Check 'diagnostic_test.log' for detailed stage-by-stage logging.")
    print("Look for lines containing:")
    print("  - [2025] SOY_INITIAL: total_rows=X, actives=Y")
    print("  - [2025] AFTER_EXPERIENCED_TERMS: total_rows=X, actives=Y")
    print("  - [2025] AFTER_HIRING: total_rows=X, actives=Y")
    print("  - [2025] AFTER_NH_TERMS: total_rows=X, actives=Y")
    print("  - [2025] FINAL_EOY: total_rows=X, actives=Y")
    print("  - [EOY SUCCESS] or [EOY ASSERTION] messages")
    
    if success:
        print("\n🎯 Diagnostic simulation completed successfully!")
    else:
        print("\n❌ Diagnostic simulation revealed issues that need investigation.")
