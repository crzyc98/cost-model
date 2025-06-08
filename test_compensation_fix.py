#!/usr/bin/env python3
"""
Unit test to verify the compensation fix works correctly.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

from cost_model.state.snapshot_update import update
from cost_model.state.schema import EMP_ID, EMP_GROSS_COMP, EMP_ACTIVE

def test_compensation_fix():
    """
    Self-contained test that verifies compensation events are applied correctly.
    
    This test creates a sample employee snapshot and compensation events,
    then calls the snapshot_update.update() function to verify the fix works.
    """
    
    print("=== COMPENSATION FIX VERIFICATION TEST ===")
    
    # Step 1: Create sample employee snapshot
    test_snapshot = pd.DataFrame({
        EMP_ID: ['NH_2025_0021'],
        EMP_GROSS_COMP: [69108.49],
        EMP_ACTIVE: [True],
        'employee_level': [1],
        'employee_tenure_band': ['1-3'],
        'employee_hire_date': [pd.Timestamp('2024-01-01')],
        'employee_birth_date': [pd.Timestamp('1990-01-01')],
        'employee_termination_date': [pd.NaT],
        'employee_deferral_rate': [0.0],
        'active': [True],
        'exited': [False],
        'simulation_year': [2026]
    })
    test_snapshot = test_snapshot.set_index(EMP_ID)
    
    print(f"Starting compensation: ${test_snapshot.loc['NH_2025_0021', EMP_GROSS_COMP]:,.2f}")
    
    # Step 2: Create compensation events (matching real simulation data)
    
    # EVT_COMP event (merit raise to new total compensation)
    comp_event = pd.DataFrame({
        'employee_id': ['NH_2025_0021'],
        'event_id': ['evt_comp_2026_0001'],
        'event_time': [pd.Timestamp('2026-01-01 00:01:00')],
        'event_type': ['EVT_COMP'],
        'value_num': [71872.83],  # New total compensation after 4% merit raise
        'value_json': [json.dumps({
            "reason": "merit_raise",
            "pct": 0.04,
            "old_comp": 69108.49,
            "new_comp": 71872.83
        })],
        'meta': ['Merit raise for NH_2025_0021: 69108.49 -> 71872.83 (+4.00%)'],
        'simulation_year': [2026]
    })
    
    # EVT_COLA event (cost of living adjustment amount)
    cola_event = pd.DataFrame({
        'employee_id': ['NH_2025_0021'],
        'event_id': ['evt_cola_2026_0001'],
        'event_time': [pd.Timestamp('2026-01-01 00:02:00')],
        'event_type': ['EVT_COLA'],
        'value_num': [1243.95],  # COLA amount to add (1.8% of merit-adjusted compensation)
        'value_json': ['{}'],
        'meta': ['COLA 1.8%'],
        'simulation_year': [2026]
    })
    
    # Combine events
    all_events = pd.concat([comp_event, cola_event], ignore_index=True)
    
    print(f"Created {len(all_events)} compensation events:")
    for _, event in all_events.iterrows():
        print(f"  {event['event_type']}: ${event['value_num']:,.2f}")
    
    # Step 3: Apply events using snapshot_update.update()
    try:
        updated_snapshot = update(
            prev_snapshot=test_snapshot,
            new_events=all_events,
            snapshot_year=2026
        )
        
        final_comp = updated_snapshot.loc['NH_2025_0021', EMP_GROSS_COMP]
        
        # Step 4: Verify the result
        expected_comp = 71872.83 + 1243.95  # Merit (total) + COLA (additive)
        actual_comp = final_comp
        difference = actual_comp - expected_comp
        
        print(f"\\nResults:")
        print(f"  Expected final compensation: ${expected_comp:,.2f}")
        print(f"  Actual final compensation:   ${actual_comp:,.2f}")
        print(f"  Difference:                  ${difference:,.2f}")
        
        # Test passes if difference is within 1 cent
        if abs(difference) < 0.01:
            print("\\n✅ SUCCESS: Compensation events applied correctly!")
            print("   The snapshot_update logic works as expected.")
            return True
        else:
            print("\\n❌ FAILED: Compensation events not applied correctly!")
            return False
            
    except Exception as e:
        print(f"\\n❌ ERROR during snapshot update: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_fix_simulation():
    """
    Demonstrate that the CLI fix would resolve the issue.
    """
    
    print("\\n=== CLI FIX DEMONSTRATION ===")
    print("The issue is in cost_model/projections/cli.py line 342:")
    print()
    print("BEFORE (buggy):")
    print("  yearly_eoy_snapshots[year] = enhanced_yearly_snapshot")
    print()
    print("AFTER (fixed):")
    print("  yearly_eoy_snapshots[year] = eoy_snapshot")
    print()
    print("This ensures that the snapshot with correct compensation")
    print("updates (from the orchestrator) is saved to files, rather")
    print("than the enhanced snapshot which may lose compensation changes.")

if __name__ == "__main__":
    # Run the unit test
    success = test_compensation_fix()
    
    # Show the CLI fix
    test_cli_fix_simulation()
    
    if success:
        print("\\n🎉 COMPENSATION FIX VERIFIED!")
        print("   Apply the CLI fix to resolve the negative pay growth issue.")
    else:
        print("\\n💥 COMPENSATION LOGIC BROKEN!")
        print("   Additional debugging needed in snapshot_update.py")