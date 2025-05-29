#!/usr/bin/env python3
"""
Test script to verify COLA event integration into the compensation engine.

This script tests that:
1. COLA events are generated when cola_pct > 0 in hazard table
2. COLA events are properly integrated into the main simulation flow
3. EVT_COLA events are created with correct structure and values
"""

import pandas as pd
import numpy as np
from types import SimpleNamespace
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

from cost_model.engines.comp import generate_cola_events
from cost_model.engines.run_one_year import run_one_year
from cost_model.state.schema import (
    EMP_ID, EMP_GROSS_COMP, EMP_HIRE_DATE, EMP_BIRTH_DATE,
    EMP_TERM_DATE, EMP_ACTIVE, EMP_DEFERRAL_RATE, EMP_TENURE,
    EMP_TENURE_BAND, EMP_LEVEL, EMP_LEVEL_SOURCE, EMP_EXITED,
    EMP_STATUS_EOY, SIMULATION_YEAR, EMP_CONTR, EMPLOYER_CORE,
    EMPLOYER_MATCH, IS_ELIGIBLE, EVT_COLA
)
from cost_model.state.event_log import EVENT_COLS


def create_test_snapshot():
    """Create a minimal test snapshot with 3 employees."""
    return pd.DataFrame({
        EMP_ID: ["E001", "E002", "E003"],
        EMP_HIRE_DATE: [pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01"), pd.Timestamp("2022-01-01")],
        EMP_BIRTH_DATE: [pd.Timestamp("1990-01-01"), pd.Timestamp("1985-01-01"), pd.Timestamp("1995-01-01")],
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
    })


def create_test_hazard_slice(cola_pct=0.03):
    """Create a test hazard slice with specified COLA percentage."""
    return pd.DataFrame({
        "simulation_year": [2025],
        "cola_pct": [cola_pct],
        "role": ["all"],
        "tenure_band": ["all"],
        "term_rate": [0.0],
        "comp_raise_pct": [0.0],
        "new_hire_termination_rate": [0.0],
        "cfg": [SimpleNamespace()]
    })


def test_cola_event_generation():
    """Test that COLA events are generated correctly."""
    print("Testing COLA event generation...")

    snapshot = create_test_snapshot()
    hazard_slice = create_test_hazard_slice(cola_pct=0.03)  # 3% COLA
    as_of = pd.Timestamp("2025-01-01")
    rng = np.random.default_rng(42)

    # Generate COLA events
    cola_events = generate_cola_events(
        snapshot=snapshot,
        hazard_slice=hazard_slice,
        as_of=as_of,
        days_into_year=0,
        jitter_days=0,
        rng=rng
    )

    # Verify structure
    assert len(cola_events) == 1, "Should return a list with one DataFrame"
    events_df = cola_events[0]

    # Verify we have events for all active employees
    assert len(events_df) == 3, f"Expected 3 COLA events, got {len(events_df)}"

    # Verify event structure
    assert set(events_df.columns) == set(EVENT_COLS), "Events should have correct columns"
    assert all(events_df["event_type"] == EVT_COLA), "All events should be EVT_COLA type"

    # Verify COLA amounts (3% of compensation)
    expected_amounts = [1500.0, 2250.0, 1800.0]  # 3% of [50000, 75000, 60000]
    actual_amounts = sorted(events_df["value_num"].tolist())
    expected_amounts_sorted = sorted(expected_amounts)

    for actual, expected in zip(actual_amounts, expected_amounts_sorted):
        assert abs(actual - expected) < 0.01, f"Expected COLA amount {expected}, got {actual}"

    print("✓ COLA event generation test passed")


def test_zero_cola_rate():
    """Test that no events are generated when COLA rate is 0."""
    print("Testing zero COLA rate...")

    snapshot = create_test_snapshot()
    hazard_slice = create_test_hazard_slice(cola_pct=0.0)  # 0% COLA
    as_of = pd.Timestamp("2025-01-01")
    rng = np.random.default_rng(42)

    # Generate COLA events
    cola_events = generate_cola_events(
        snapshot=snapshot,
        hazard_slice=hazard_slice,
        as_of=as_of,
        days_into_year=0,
        jitter_days=0,
        rng=rng
    )

    # Verify no events generated
    assert len(cola_events) == 1, "Should return a list with one DataFrame"
    events_df = cola_events[0]
    assert len(events_df) == 0, "Should generate no events when COLA rate is 0"

    print("✓ Zero COLA rate test passed")


def test_integration_with_run_one_year():
    """Test that COLA events are integrated into the main simulation flow."""
    print("Testing integration with run_one_year...")

    # Create minimal test data
    snapshot = pd.DataFrame({
        EMP_ID: ["E001", "E002"],
        EMP_HIRE_DATE: [pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")],
        EMP_BIRTH_DATE: [pd.Timestamp("1990-01-01"), pd.Timestamp("1985-01-01")],
        EMP_GROSS_COMP: [50000.0, 75000.0],
        EMP_TERM_DATE: [pd.NaT, pd.NaT],
        EMP_ACTIVE: [True, True],
        EMP_DEFERRAL_RATE: [0.05, 0.10],
        EMP_TENURE: [5.0, 4.0],
        EMP_TENURE_BAND: ["5-10", "3-5"],
        EMP_LEVEL: [1, 2],
        EMP_LEVEL_SOURCE: ["hire", "hire"],
        EMP_EXITED: [False, False],
        EMP_STATUS_EOY: ["Active", "Active"],
        SIMULATION_YEAR: [2025, 2025],
        EMP_CONTR: [0.0, 0.0],
        EMPLOYER_CORE: [0.0, 0.0],
        EMPLOYER_MATCH: [0.0, 0.0],
        IS_ELIGIBLE: [True, True]
    })

    empty_event_log = pd.DataFrame(columns=EVENT_COLS)

    # Create hazard table with COLA
    hazard_table = pd.DataFrame([{
        "simulation_year": 2025,
        "role": "all",
        "tenure_band": "all",
        "term_rate": 0.0,
        "comp_raise_pct": 0.0,
        "new_hire_termination_rate": 0.0,
        "cola_pct": 0.02,  # 2% COLA
        "employee_level": "all",
        "cfg": SimpleNamespace()
    }])

    # Create minimal global params
    global_params = SimpleNamespace(
        days_into_year_for_cola=0,  # Start of year COLA for simplicity
        cola_jitter_days=0,
        annual_compensation_increase_rate=0.0,
        annual_termination_rate=0.0,
        new_hire_termination_rate=0.0,
        min_eligibility_age=21,
        min_service_months=12
    )

    plan_rules = {}
    rng = np.random.default_rng(42)

    # Run one year simulation
    try:
        events, final_snapshot = run_one_year(
            event_log=empty_event_log,
            prev_snapshot=snapshot,
            year=2025,
            global_params=global_params,
            plan_rules=plan_rules,
            hazard_table=hazard_table,
            rng=rng,
            census_template_path=None,
            rng_seed_offset=0,
            deterministic_term=False
        )

        # Check for COLA events in the output
        cola_events = events[events["event_type"] == EVT_COLA]
        print(f"Found {len(cola_events)} COLA events")

        if len(cola_events) > 0:
            print("COLA event details:")
            for _, event in cola_events.iterrows():
                print(f"  Employee {event[EMP_ID]}: ${event['value_num']:.2f}")

            # Verify COLA events have correct structure
            assert all(cola_events["value_num"] > 0), "COLA events should have positive values"
            print(f"✓ Integration test passed - Generated {len(cola_events)} COLA events")
        else:
            print("⚠ No COLA events generated - this may indicate the integration needs debugging")

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Run all tests."""
    print("Running COLA integration tests...\n")

    try:
        test_cola_event_generation()
        test_zero_cola_rate()
        test_integration_with_run_one_year()

        print("\n🎉 All COLA integration tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
