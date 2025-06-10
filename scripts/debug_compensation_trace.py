#!/usr/bin/env python3
"""
Debug script to trace exactly what happens when compensation events are applied.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from logging_config import get_logger

# Import the necessary modules
from cost_model.config.loaders import load_config_to_namespace
from cost_model.engines.comp import bump
from cost_model.engines.run_one_year.validation import validate_and_extract_hazard_slice
from cost_model.projections.hazard import load_and_expand_hazard_table
from cost_model.projections.snapshot import create_initial_snapshot
from cost_model.state.schema import EMP_ACTIVE, EMP_GROSS_COMP, EMP_ID
from cost_model.state.snapshot_update import update

logger = get_logger(__name__)


def test_compensation_application():
    """Test compensation event application step by step."""

    logger.info("=== TESTING COMPENSATION EVENT APPLICATION ===")

    # 1. Create a minimal test case
    print("Step 1: Create minimal test snapshot")
    test_snapshot = pd.DataFrame(
        {
            EMP_ID: ["NH_2025_0021"],
            EMP_GROSS_COMP: [69108.49],
            EMP_ACTIVE: [True],
            "employee_level": [1],
            "employee_tenure_band": ["1-3"],
            "employee_hire_date": [pd.Timestamp("2024-01-01")],
            "employee_birth_date": [pd.Timestamp("1990-01-01")],
            "employee_termination_date": [pd.NaT],
            "employee_deferral_rate": [0.0],
            "active": [True],
            "exited": [False],
            "simulation_year": [2026],
        }
    )
    test_snapshot = test_snapshot.set_index(EMP_ID)

    print(f"Initial compensation: ${test_snapshot.loc['NH_2025_0021', EMP_GROSS_COMP]:,.2f}")

    # 2. Create test events matching what we saw in the actual simulation
    print("\nStep 2: Create test events")

    # EVT_COMP event (merit raise)
    comp_event = pd.DataFrame(
        {
            "employee_id": ["NH_2025_0021"],
            "event_id": ["evt_comp_2026_0001"],
            "event_time": [pd.Timestamp("2026-01-01 00:01:00")],
            "event_type": ["EVT_COMP"],
            "value_num": [71872.83],  # New total compensation
            "value_json": [
                json.dumps(
                    {
                        "reason": "merit_raise",
                        "pct": 0.04,
                        "old_comp": 69108.49,
                        "new_comp": 71872.83,
                    }
                )
            ],
            "meta": ["Merit raise for NH_2025_0021: 69108.49 -> 71872.83 (+4.00%)"],
            "simulation_year": [2026],
        }
    )

    # EVT_COLA event (cost of living adjustment)
    cola_event = pd.DataFrame(
        {
            "employee_id": ["NH_2025_0021"],
            "event_id": ["evt_cola_2026_0001"],
            "event_time": [pd.Timestamp("2026-01-01 00:02:00")],
            "event_type": ["EVT_COLA"],
            "value_num": [1243.95],  # COLA amount to add
            "value_json": ["{}"],
            "meta": ["COLA 1.8%"],
            "simulation_year": [2026],
        }
    )

    # Combine events
    all_events = pd.concat([comp_event, cola_event], ignore_index=True)

    print(f"Generated {len(all_events)} events:")
    for _, event in all_events.iterrows():
        print(f"  {event['event_type']}: ${event['value_num']:,.2f} at {event['event_time']}")

    # 3. Apply events using the same update function used by the orchestrator
    print("\nStep 3: Apply events using snapshot_update.update()")

    try:
        updated_snapshot = update(
            prev_snapshot=test_snapshot, new_events=all_events, snapshot_year=2026
        )

        final_comp = updated_snapshot.loc["NH_2025_0021", EMP_GROSS_COMP]
        print(f"Final compensation after update(): ${final_comp:,.2f}")

        # Calculate expected final compensation
        expected_comp = 71872.83 + 1243.95  # Merit + COLA
        print(f"Expected compensation: ${expected_comp:,.2f}")
        print(f"Difference: ${final_comp - expected_comp:,.2f}")

        # Check if update worked correctly
        if abs(final_comp - expected_comp) < 0.01:
            print("✅ SUCCESS: Compensation events applied correctly!")
            return True
        else:
            print("❌ FAILED: Compensation events not applied correctly!")
            return False

    except Exception as e:
        print(f"❌ ERROR during update(): {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = test_compensation_application()

    if success:
        print("\n🎉 The compensation application logic works correctly!")
        print("   The issue must be elsewhere in the orchestrator flow.")
    else:
        print("\n💥 Found the bug in compensation application!")
        print("   The snapshot_update.update() function is not working correctly.")
