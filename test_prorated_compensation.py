#!/usr/bin/env python3
"""
Test script to verify that prorated compensation events are applied correctly for terminated employees.
"""

import pandas as pd
import numpy as np
from cost_model.state.snapshot_update import update
from cost_model.state.schema import EMP_ID, EMP_GROSS_COMP, EMP_TERM_DATE, EVT_COMP, EVT_TERM

def test_prorated_compensation():
    """Test that prorated compensation events are applied to terminated employees."""
    print("Testing prorated compensation for terminated employees...")
    
    # Create test snapshot with one employee
    snapshot = pd.DataFrame({
        EMP_ID: ["TEST_TERM"],
        EMP_GROSS_COMP: [60000.0],  # Full annual compensation
        EMP_TERM_DATE: [pd.NaT],    # Not terminated yet
        "active": [True],
        "employee_hire_date": pd.to_datetime(["2020-01-01"]),
        "employee_birth_date": pd.to_datetime(["1990-01-01"]),
        "employee_role": ["Staff"],
        "employee_deferral_rate": [0.05],
        "employee_tenure": [5.0],
        "employee_tenure_band": ["5-10"],
        "employee_level": [1],
        "job_level_source": ["hire"],
        "exited": [False],
        "employee_status_eoy": ["Active"],
        "simulation_year": [2025],
    }).set_index(EMP_ID)
    
    print(f"Initial compensation: {snapshot[EMP_GROSS_COMP].iloc[0]}")
    
    # Create events: termination on March 31 (90 days) with prorated compensation
    termination_date = pd.Timestamp("2025-03-31")
    days_worked = 90  # Jan 1 to March 31
    prorated_comp = 60000.0 * (days_worked / 365)  # ~14,795
    
    events = pd.DataFrame({
        "event_time": [termination_date, termination_date],
        EMP_ID: ["TEST_TERM", "TEST_TERM"],
        "event_type": [EVT_TERM, EVT_COMP],
        "value_num": [None, prorated_comp],
        "value_json": ['{"reason": "termination"}', None],
        "meta": ["Termination", f"Prorated comp ({days_worked} days)"],
        "simulation_year": [2025, 2025]
    })
    
    print(f"Expected prorated compensation: {prorated_comp:.2f}")
    print(f"Events:")
    print(events[["event_type", "value_num", "meta"]])
    
    # Apply events to snapshot
    updated_snapshot = update(
        prev_snapshot=snapshot,
        new_events=events,
        snapshot_year=2025
    )
    
    final_comp = updated_snapshot[EMP_GROSS_COMP].iloc[0]
    final_term_date = updated_snapshot[EMP_TERM_DATE].iloc[0]
    final_active = updated_snapshot["active"].iloc[0]
    
    print(f"\nResults:")
    print(f"Final compensation: {final_comp:.2f}")
    print(f"Termination date: {final_term_date}")
    print(f"Active status: {final_active}")
    
    # Check if prorated compensation was applied
    if abs(final_comp - prorated_comp) < 0.01:
        print("✅ SUCCESS: Prorated compensation was applied correctly!")
        return True
    else:
        print(f"❌ FAILURE: Expected {prorated_comp:.2f}, got {final_comp:.2f}")
        print(f"   Difference: {abs(final_comp - prorated_comp):.2f}")
        return False

if __name__ == "__main__":
    success = test_prorated_compensation()
    if success:
        print("\n🎉 Prorated compensation test passed!")
    else:
        print("\n💥 Prorated compensation test failed.")
