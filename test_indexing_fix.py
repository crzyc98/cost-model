#!/usr/bin/env python3
"""
Test the indexing fix for compensation events.
"""

import pandas as pd
import numpy as np
import json
from cost_model.state.snapshot_update import update
from cost_model.state.schema import EMP_ID, EMP_GROSS_COMP, EMP_ACTIVE

def test_indexing_fix():
    """Test that compensation events work with integer-indexed snapshots."""
    
    print("=== TESTING INDEXING FIX FOR COMPENSATION EVENTS ===")
    
    # Create a snapshot with INTEGER INDEX (like what the orchestrator produces)
    test_snapshot = pd.DataFrame({
        EMP_ID: ['NEW_900000001', 'NEW_900000002', 'NEW_900000003'],
        EMP_GROSS_COMP: [100000.0, 150000.0, 200000.0],
        EMP_ACTIVE: [True, True, True],
        'employee_level': [1, 2, 3],
        'employee_tenure_band': ['1-3', '3-5', '5-10'],
        'employee_hire_date': [pd.Timestamp('2020-01-01')] * 3,
        'employee_birth_date': [pd.Timestamp('1990-01-01')] * 3,
        'employee_termination_date': [pd.NaT] * 3,
        'employee_deferral_rate': [0.0] * 3,
        'active': [True] * 3,
        'exited': [False] * 3,
        'simulation_year': [2026] * 3
    })
    # CRITICAL: Use integer index instead of employee ID index
    test_snapshot.index = range(len(test_snapshot))  # 0, 1, 2
    
    print("Created test snapshot with integer index:")
    print(f"  Index: {test_snapshot.index.tolist()}")
    print(f"  Employee IDs: {test_snapshot[EMP_ID].tolist()}")
    print(f"  Initial compensations: {test_snapshot[EMP_GROSS_COMP].tolist()}")
    
    # Create compensation events
    comp_events = pd.DataFrame({
        'employee_id': ['NEW_900000001', 'NEW_900000002', 'NEW_900000003'],
        'event_id': ['evt_comp_001', 'evt_comp_002', 'evt_comp_003'],
        'event_time': [pd.Timestamp('2026-01-01 00:01:00')] * 3,
        'event_type': ['EVT_COMP'] * 3,
        'value_num': [104000.0, 156000.0, 208000.0],  # 4% merit raises
        'value_json': ['{}'] * 3,
        'meta': ['Merit raise'] * 3,
        'simulation_year': [2026] * 3
    })
    
    cola_events = pd.DataFrame({
        'employee_id': ['NEW_900000001', 'NEW_900000002', 'NEW_900000003'],
        'event_id': ['evt_cola_001', 'evt_cola_002', 'evt_cola_003'],
        'event_time': [pd.Timestamp('2026-01-01 00:02:00')] * 3,
        'event_type': ['EVT_COLA'] * 3,
        'value_num': [1872.0, 2808.0, 3744.0],  # 1.8% COLA on merit-adjusted compensation
        'value_json': ['{}'] * 3,
        'meta': ['COLA 1.8%'] * 3,
        'simulation_year': [2026] * 3
    })
    
    all_events = pd.concat([comp_events, cola_events], ignore_index=True)
    
    print("\nCreated compensation events:")
    for i, row in all_events.iterrows():
        print(f"  {row['event_type']} for {row['employee_id']}: ${row['value_num']:,.2f}")
    
    # Apply events using the fixed update function
    print("\nApplying events with fixed snapshot_update.update()...")
    
    try:
        updated_snapshot = update(
            prev_snapshot=test_snapshot,
            new_events=all_events,
            snapshot_year=2026
        )
        
        print("\nResults:")
        expected_results = [105872.0, 158808.0, 211744.0]  # merit + cola
        
        for i, emp_id in enumerate(['NEW_900000001', 'NEW_900000002', 'NEW_900000003']):
            emp_row = updated_snapshot[updated_snapshot[EMP_ID] == emp_id]
            if not emp_row.empty:
                actual_comp = emp_row[EMP_GROSS_COMP].iloc[0]
                expected_comp = expected_results[i]
                
                print(f"  {emp_id}:")
                print(f"    Expected: ${expected_comp:,.2f}")
                print(f"    Actual:   ${actual_comp:,.2f}")
                print(f"    Diff:     ${actual_comp - expected_comp:,.2f}")
                
                if abs(actual_comp - expected_comp) < 0.01:
                    print(f"    ✅ CORRECT")
                else:
                    print(f"    ❌ WRONG")
                    return False
            else:
                print(f"  {emp_id}: ❌ NOT FOUND")
                return False
        
        print("\n🎉 SUCCESS: All compensation events applied correctly with integer index!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_indexing_fix()
    
    if success:
        print("\n✅ INDEXING FIX VERIFIED!")
        print("   Run a new simulation to see correct compensation growth.")
    else:
        print("\n❌ INDEXING FIX FAILED!")
        print("   Additional debugging needed.")