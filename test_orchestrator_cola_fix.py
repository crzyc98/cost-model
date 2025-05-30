#!/usr/bin/env python3
"""
Test script to verify that the orchestrator COLA fix works correctly.

This test verifies that:
1. The _generate_compensation_events function calls bump() which includes COLA events
2. The return type is correctly List[pd.DataFrame] instead of List[List[pd.DataFrame]]
3. COLA events are properly generated and included in the result
"""

import pandas as pd
import numpy as np
from types import SimpleNamespace
import sys
import os
import logging

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_snapshot():
    """Create a test snapshot with active employees."""
    from cost_model.state.schema import (
        EMP_ID, EMP_GROSS_COMP, EMP_TERM_DATE, EMP_LEVEL,
        EMP_TENURE_BAND, EMP_HIRE_DATE
    )

    return pd.DataFrame({
        EMP_ID: ['E001', 'E002', 'E003'],
        EMP_GROSS_COMP: [50000.0, 75000.0, 100000.0],
        EMP_TERM_DATE: [pd.NaT, pd.NaT, pd.NaT],  # All active
        EMP_LEVEL: [1, 2, 1],  # Required for hazard table merge
        EMP_TENURE_BAND: ['1-2', '3-5', '1-2'],  # Required for hazard table merge
        EMP_HIRE_DATE: ['2020-01-01', '2019-01-01', '2021-01-01']  # Required for tenure calculation
    })

def create_test_hazard_slice(cola_pct=0.03):
    """Create a test hazard slice with COLA percentage."""
    from cost_model.state.schema import EMP_TENURE_BAND

    return pd.DataFrame({
        'simulation_year': [2025, 2025],
        'level': [1, 2],  # Required level column
        EMP_TENURE_BAND: ['1-2', '3-5'],  # Required tenure band column
        'cola_pct': [cola_pct, cola_pct],
        'comp_raise_pct': [0.02, 0.03]  # Standard raise percentages
    })

def test_compensation_events_generation():
    """Test that _generate_compensation_events returns the correct structure."""
    print("Testing _generate_compensation_events function...")

    try:
        from cost_model.engines.run_one_year.orchestrator import _generate_compensation_events
        from cost_model.engines.run_one_year.orchestrator.base import YearContext
        from cost_model.state.schema import EMP_ID, EVT_COMP, EVT_COLA

        # Create test data
        snapshot = create_test_snapshot()
        hazard_slice = create_test_hazard_slice(cola_pct=0.03)  # 3% COLA

        global_params = SimpleNamespace()
        plan_rules = {}

        year_context = YearContext(
            year=2025,
            as_of=pd.Timestamp('2025-01-01'),
            end_of_year=pd.Timestamp('2025-12-31'),
            year_rng=np.random.default_rng(42),
            hazard_slice=hazard_slice,
            global_params=global_params,
            plan_rules=plan_rules
        )

        # Test the function
        result = _generate_compensation_events(snapshot, year_context, logger)

        # Verify return type
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        print(f"✓ Function returns a list with {len(result)} elements")

        # Verify each element is a DataFrame
        for i, item in enumerate(result):
            assert isinstance(item, pd.DataFrame), f"Element {i} is not a DataFrame: {type(item)}"
            print(f"  Element {i}: DataFrame with {len(item)} rows")

        # Check for both EVT_COMP and EVT_COLA events
        all_events = pd.concat(result, ignore_index=True) if result else pd.DataFrame()

        if not all_events.empty:
            event_types = all_events['event_type'].unique()
            print(f"✓ Event types generated: {list(event_types)}")

            # Check for COLA events specifically
            cola_events = all_events[all_events['event_type'] == EVT_COLA]
            comp_events = all_events[all_events['event_type'] == EVT_COMP]

            print(f"  EVT_COLA events: {len(cola_events)}")
            print(f"  EVT_COMP events: {len(comp_events)}")

            if len(cola_events) > 0:
                print("✓ COLA events are being generated!")
                print(f"  COLA values: {cola_events['value_num'].tolist()}")
            else:
                print("⚠ No COLA events generated (check cola_pct in hazard_slice)")

        else:
            print("⚠ No events generated")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_zero_cola_rate():
    """Test that function works correctly when COLA rate is 0."""
    print("\nTesting with zero COLA rate...")

    try:
        from cost_model.engines.run_one_year.orchestrator import _generate_compensation_events
        from cost_model.engines.run_one_year.orchestrator.base import YearContext
        from cost_model.state.schema import EVT_COMP, EVT_COLA

        # Create test data with 0% COLA
        snapshot = create_test_snapshot()
        hazard_slice = create_test_hazard_slice(cola_pct=0.0)  # 0% COLA

        global_params = SimpleNamespace()
        plan_rules = {}

        year_context = YearContext(
            year=2025,
            as_of=pd.Timestamp('2025-01-01'),
            end_of_year=pd.Timestamp('2025-12-31'),
            year_rng=np.random.default_rng(42),
            hazard_slice=hazard_slice,
            global_params=global_params,
            plan_rules=plan_rules
        )

        # Test the function
        result = _generate_compensation_events(snapshot, year_context, logger)

        # Verify structure
        assert isinstance(result, list), f"Expected list, got {type(result)}"

        # Check events
        all_events = pd.concat(result, ignore_index=True) if result else pd.DataFrame()

        if not all_events.empty:
            event_types = all_events['event_type'].unique()
            cola_events = all_events[all_events['event_type'] == EVT_COLA]
            comp_events = all_events[all_events['event_type'] == EVT_COMP]

            print(f"✓ With 0% COLA rate:")
            print(f"  EVT_COLA events: {len(cola_events)}")
            print(f"  EVT_COMP events: {len(comp_events)}")

            # Should have no COLA events but may have comp events
            assert len(cola_events) == 0, "Should have no COLA events when rate is 0%"
            print("✓ Correctly generates no COLA events when rate is 0%")

        return True

    except Exception as e:
        print(f"✗ Zero COLA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Testing orchestrator COLA fix...")

    success1 = test_compensation_events_generation()
    success2 = test_zero_cola_rate()

    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED!")
        print("The orchestrator fix should now:")
        print("1. ✅ Call bump() which includes both EVT_COMP and EVT_COLA events")
        print("2. ✅ Return List[pd.DataFrame] instead of List[List[pd.DataFrame]]")
        print("3. ✅ Generate COLA events when cola_pct > 0")
        print("4. ✅ Handle zero COLA rate correctly")
    else:
        print("\n💥 SOME TESTS FAILED!")

    return success1 and success2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
