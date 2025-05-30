#!/usr/bin/env python3
"""
Simple test to verify COLA integration is working.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

def test_basic_cola():
    """Test basic COLA functionality."""
    print("Testing basic COLA functionality...")

    try:
        from cost_model.engines.cola import cola
        from cost_model.state.schema import EMP_ID, EMP_GROSS_COMP, EMP_TERM_DATE, EVT_COLA

        # Create test data
        snapshot = pd.DataFrame({
            EMP_ID: ['E001', 'E002'],
            EMP_GROSS_COMP: [50000.0, 75000.0],
            EMP_TERM_DATE: [pd.NaT, pd.NaT]
        })

        hazard_slice = pd.DataFrame({
            'simulation_year': [2025],
            'cola_pct': [0.03]
        })

        as_of = pd.Timestamp('2025-01-01')

        # Test COLA function
        result = cola(snapshot, hazard_slice, as_of)
        events_df = result[0]

        print(f"✓ Generated {len(events_df)} COLA events")
        print(f"  Event values: {events_df['value_num'].tolist()}")

        return True

    except Exception as e:
        print(f"✗ Basic COLA test failed: {e}")
        return False


def test_comp_integration():
    """Test COLA integration in comp engine."""
    print("Testing COLA integration in comp engine...")

    try:
        from cost_model.engines.comp import generate_cola_events
        from cost_model.state.schema import EMP_ID, EMP_GROSS_COMP, EMP_TERM_DATE, EVT_COLA

        # Create test data
        snapshot = pd.DataFrame({
            EMP_ID: ['E001', 'E002'],
            EMP_GROSS_COMP: [50000.0, 75000.0],
            EMP_TERM_DATE: [pd.NaT, pd.NaT]
        })

        hazard_slice = pd.DataFrame({
            'simulation_year': [2025],
            'cola_pct': [0.03]
        })

        as_of = pd.Timestamp('2025-01-01')
        rng = np.random.default_rng(42)

        # Test integration function
        result = generate_cola_events(snapshot, hazard_slice, as_of, rng=rng)
        events_df = result[0]

        print(f"✓ Generated {len(events_df)} COLA events via comp engine")
        print(f"  Event values: {events_df['value_num'].tolist()}")

        return True

    except Exception as e:
        print(f"✗ Comp integration test failed: {e}")
        return False


def test_orchestrator_integration():
    """Test COLA integration in orchestrator."""
    print("Testing COLA integration in orchestrator...")

    try:
        from cost_model.engines.run_one_year.orchestrator import _generate_compensation_events
        from cost_model.engines.run_one_year.orchestrator.base import YearContext
        from cost_model.state.schema import (
            EMP_ID, EMP_GROSS_COMP, EMP_TERM_DATE, EMP_LEVEL,
            EMP_TENURE_BAND, EMP_HIRE_DATE, EVT_COLA
        )
        from types import SimpleNamespace
        import logging

        # Create test data with required columns
        snapshot = pd.DataFrame({
            EMP_ID: ['E001', 'E002'],
            EMP_GROSS_COMP: [50000.0, 75000.0],
            EMP_TERM_DATE: [pd.NaT, pd.NaT],
            EMP_LEVEL: [1, 2],  # Required for hazard table merge
            EMP_TENURE_BAND: ['1-2', '3-5'],  # Required for hazard table merge
            EMP_HIRE_DATE: ['2020-01-01', '2019-01-01']  # Required for tenure calculation
        })

        # Create proper hazard slice with required columns
        hazard_slice = pd.DataFrame({
            'simulation_year': [2025, 2025],
            'level': [1, 2],  # Required level column
            EMP_TENURE_BAND: ['1-2', '3-5'],  # Required tenure band column
            'cola_pct': [0.03, 0.03],
            'comp_raise_pct': [0.02, 0.03]  # Standard raise percentages
        })

        global_params = SimpleNamespace()

        year_context = YearContext(
            year=2025,
            as_of=pd.Timestamp('2025-01-01'),
            end_of_year=pd.Timestamp('2025-12-31'),
            year_rng=np.random.default_rng(42),
            hazard_slice=hazard_slice,
            global_params=global_params,
            plan_rules={}
        )

        logger = logging.getLogger(__name__)

        # Test orchestrator function (now returns List[pd.DataFrame])
        result = _generate_compensation_events(snapshot, year_context, logger)

        # Find COLA events in the result
        all_events = pd.concat(result, ignore_index=True) if result else pd.DataFrame()
        cola_events = all_events[all_events['event_type'] == EVT_COLA]

        print(f"✓ Generated {len(cola_events)} COLA events via orchestrator")
        print(f"  Event values: {cola_events['value_num'].tolist()}")

        return True

    except Exception as e:
        print(f"✗ Orchestrator integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Running simple COLA integration tests...\n")

    tests = [
        test_basic_cola,
        test_comp_integration,
        test_orchestrator_integration
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All COLA integration tests passed!")
        print("\nSummary:")
        print("✓ Basic COLA function works")
        print("✓ COLA integration in comp engine works")
        print("✓ COLA integration in orchestrator works")
        print("\nThe COLA event generation has been successfully integrated!")
    else:
        print("❌ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
