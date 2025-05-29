#!/usr/bin/env python3
"""
End-to-end test for COLA integration.

This test verifies that:
1. COLA events are generated correctly
2. COLA events are integrated into the orchestrator
3. COLA events are processed by snapshot updates
4. Employee compensation is updated correctly
"""

import pandas as pd
import numpy as np
from types import SimpleNamespace
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

from cost_model.engines.comp import generate_cola_events
from cost_model.engines.run_one_year.orchestrator import _generate_cola_events
from cost_model.engines.run_one_year.orchestrator.base import YearContext
from cost_model.state.snapshot_update import update as update_snapshot
from cost_model.state.schema import (
    EMP_ID, EMP_GROSS_COMP, EMP_HIRE_DATE, EMP_BIRTH_DATE,
    EMP_TERM_DATE, EMP_ACTIVE, EMP_DEFERRAL_RATE, EMP_TENURE,
    EMP_TENURE_BAND, EMP_LEVEL, EMP_LEVEL_SOURCE, EMP_EXITED,
    EMP_STATUS_EOY, SIMULATION_YEAR, EMP_CONTR, EMPLOYER_CORE,
    EMPLOYER_MATCH, IS_ELIGIBLE, EVT_COLA
)
from cost_model.state.event_log import EVENT_COLS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_snapshot():
    """Create a test snapshot with 3 employees."""
    from cost_model.state.schema import EMP_ROLE
    return pd.DataFrame({
        EMP_ID: ["E001", "E002", "E003"],
        EMP_HIRE_DATE: [pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01"), pd.Timestamp("2022-01-01")],
        EMP_BIRTH_DATE: [pd.Timestamp("1990-01-01"), pd.Timestamp("1985-01-01"), pd.Timestamp("1995-01-01")],
        EMP_ROLE: ["Staff", "Staff", "Staff"],  # Add missing role column
        EMP_GROSS_COMP: [50000.0, 75000.0, 60000.0],
        EMP_TERM_DATE: [pd.NaT, pd.NaT, pd.NaT],  # All active
        EMP_ACTIVE: [True, True, True],
        EMP_DEFERRAL_RATE: [0.05, 0.10, 0.03],
        EMP_TENURE: [5.0, 4.0, 3.0],
        EMP_TENURE_BAND: ["5-10", "3-5", "3-5"],
        EMP_LEVEL: [1, 2, 1],
        EMP_LEVEL_SOURCE: ["hire", "hire", "hire"],
        EMP_EXITED: [False, False, False],
        EMP_STATUS_EOY: ["Active", "Active", "Active"],
        SIMULATION_YEAR: [2025, 2025, 2025],
        EMP_CONTR: [0.0, 0.0, 0.0],
        EMPLOYER_CORE: [0.0, 0.0, 0.0],
        EMPLOYER_MATCH: [0.0, 0.0, 0.0],
        IS_ELIGIBLE: [True, True, True]
    }).set_index(EMP_ID)


def create_test_hazard_slice(cola_pct=0.03):
    """Create a test hazard slice with specified COLA percentage."""
    return pd.DataFrame({
        "simulation_year": [2025],
        "cola_pct": [cola_pct]
    })


def test_end_to_end_cola_integration():
    """Test the complete COLA integration from generation to snapshot update."""
    print("Testing end-to-end COLA integration...")

    # Step 1: Create test data
    snapshot = create_test_snapshot()
    hazard_slice = create_test_hazard_slice(cola_pct=0.04)  # 4% COLA
    as_of = pd.Timestamp("2025-01-01")
    rng = np.random.default_rng(42)

    print(f"Initial compensation: {snapshot[EMP_GROSS_COMP].tolist()}")

    # Step 2: Generate COLA events using the compensation engine
    cola_events = generate_cola_events(
        snapshot=snapshot.reset_index(),  # Reset index for the function
        hazard_slice=hazard_slice,
        as_of=as_of,
        days_into_year=0,
        jitter_days=0,
        rng=rng
    )

    assert len(cola_events) == 1, "Should return one DataFrame"
    events_df = cola_events[0]
    assert len(events_df) == 3, f"Expected 3 COLA events, got {len(events_df)}"

    print(f"Generated {len(events_df)} COLA events:")
    for _, event in events_df.iterrows():
        print(f"  Employee {event[EMP_ID]}: ${event['value_num']:.2f}")

    # Step 3: Test orchestrator integration
    global_params = SimpleNamespace(
        days_into_year_for_cola=0,
        cola_jitter_days=0
    )

    year_context = YearContext(
        year=2025,
        as_of=as_of,
        end_of_year=pd.Timestamp('2025-12-31'),
        year_rng=rng,
        hazard_slice=hazard_slice,
        global_params=global_params,
        plan_rules={}
    )

    orchestrator_result = _generate_cola_events(
        snapshot.reset_index(),
        year_context,
        logger
    )

    assert len(orchestrator_result) == 1, "Orchestrator should return one list"
    assert len(orchestrator_result[0]) == 1, "Should contain one DataFrame"
    orchestrator_events = orchestrator_result[0][0]

    print(f"Orchestrator generated {len(orchestrator_events)} COLA events")

    # Step 4: Test snapshot update with COLA events
    updated_snapshot = update_snapshot(
        prev_snapshot=snapshot,
        new_events=events_df,
        snapshot_year=2025
    )

    print(f"Updated compensation: {updated_snapshot[EMP_GROSS_COMP].tolist()}")

    # Step 5: Verify compensation updates
    expected_increases = [2000.0, 3000.0, 2400.0]  # 4% of [50000, 75000, 60000]
    expected_new_comp = [52000.0, 78000.0, 62400.0]

    for i, emp_id in enumerate(["E001", "E002", "E003"]):
        actual_comp = updated_snapshot.loc[emp_id, EMP_GROSS_COMP]
        expected_comp = expected_new_comp[i]

        assert abs(actual_comp - expected_comp) < 0.01, \
            f"Employee {emp_id}: Expected compensation {expected_comp}, got {actual_comp}"

    print("✓ End-to-end COLA integration test passed!")
    print(f"  - All employees received 4% COLA increase")
    print(f"  - Compensation updated correctly in snapshot")

    return True


def test_cola_with_timing():
    """Test COLA events with different timing parameters."""
    print("Testing COLA with timing parameters...")

    snapshot = create_test_snapshot()
    hazard_slice = create_test_hazard_slice(cola_pct=0.025)  # 2.5% COLA
    as_of = pd.Timestamp("2025-01-01")
    rng = np.random.default_rng(42)

    # Test with mid-year timing
    cola_events = generate_cola_events(
        snapshot=snapshot.reset_index(),
        hazard_slice=hazard_slice,
        as_of=as_of,
        days_into_year=182,  # Mid-year (approximately July 1st)
        jitter_days=0,
        rng=rng
    )

    events_df = cola_events[0]

    # Verify timing
    expected_date = pd.Timestamp("2025-07-02")  # 182 days from Jan 1
    for _, event in events_df.iterrows():
        event_date = pd.to_datetime(event["event_time"]).date()
        assert event_date == expected_date.date(), \
            f"Expected event date {expected_date.date()}, got {event_date}"

    print("✓ COLA timing test passed!")
    print(f"  - Events scheduled for {expected_date.date()}")

    return True


def main():
    """Run all tests."""
    print("Running end-to-end COLA integration tests...\n")

    try:
        test_end_to_end_cola_integration()
        print()
        test_cola_with_timing()

        print("\n🎉 All end-to-end COLA integration tests passed!")
        print("\nSummary:")
        print("✓ COLA events are generated correctly")
        print("✓ COLA events are integrated into the orchestrator")
        print("✓ COLA events are processed by snapshot updates")
        print("✓ Employee compensation is updated correctly")
        print("✓ COLA timing parameters work correctly")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
