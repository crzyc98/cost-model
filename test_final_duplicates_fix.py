#!/usr/bin/env python3
"""
Final test for the complete hazard table duplicates fix.

This test verifies that:
1. Conflicting rates generate appropriate warnings (once per simulation)
2. Harmless duplicates are handled quietly
3. Warning cache is properly managed
4. Events are still generated correctly
"""

import pandas as pd
import numpy as np
from types import SimpleNamespace
import sys
import os
import logging
from io import StringIO

# Add the project root to the path
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def capture_logs():
    """Set up log capture to verify warning behavior."""
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.WARNING)
    
    # Get the comp logger specifically
    comp_logger = logging.getLogger('cost_model.engines.comp')
    comp_logger.addHandler(handler)
    comp_logger.setLevel(logging.DEBUG)
    
    return log_capture, handler, comp_logger

def test_final_duplicates_fix():
    """Test the complete duplicates fix with log verification."""
    print("Testing final hazard table duplicates fix...")
    
    try:
        from cost_model.engines.comp import bump, clear_warning_cache
        
        # Set up log capture
        log_capture, handler, comp_logger = capture_logs()
        
        # Create test snapshot
        snapshot = pd.DataFrame({
            "employee_id": ["E1", "E2", "E3"],
            "active": [True, True, True],
            "employee_hire_date": pd.to_datetime(["2021-06-01", "2018-03-15", "2020-01-01"]),
            "employee_birth_date": pd.to_datetime(["1990-01-01", "1985-01-01", "1992-01-01"]),
            "employee_gross_compensation": [80000, 95000, 75000],
            "employee_termination_date": pd.NaT,
            "employee_deferral_rate": [0.05, 0.06, 0.04],
            "employee_tenure": [4, 7, 5],
            "employee_tenure_band": ["3-5", "5-10", "3-5"],
            "employee_level": [0, 1, 0],
            "job_level_source": ["initial", "initial", "initial"],
            "exited": [False, False, False],
            "employee_status_eoy": ["Active", "Active", "Active"],
            "simulation_year": [2025, 2025, 2025],
        }).set_index("employee_id", drop=False)
        
        print("=== Test 1: Conflicting rates (should warn once) ===")
        clear_warning_cache()
        log_capture.truncate(0)
        log_capture.seek(0)
        
        # Create hazard table with conflicting rates
        hazard_conflicts = pd.DataFrame({
            "simulation_year": [2025, 2025, 2025, 2025],
            "employee_level": [0, 0, 1, 1],
            "employee_tenure_band": ["3-5", "3-5", "5-10", "5-10"],
            "comp_raise_pct": [0.01, 0.0175, 0.015, 0.02],  # Conflicts!
            "cola_pct": [0.02, 0.02, 0.02, 0.02],
            "term_rate": [0.1, 0.1, 0.1, 0.1],
            "new_hire_termination_rate": [0.25, 0.25, 0.25, 0.25],
            "cfg": [SimpleNamespace()] * 4,
        })
        
        # First call - should generate warnings
        events1 = bump(
            snapshot=snapshot,
            hazard_slice=hazard_conflicts,
            as_of=pd.Timestamp("2025-01-01"),
            rng=np.random.default_rng(0)
        )
        
        first_warnings = log_capture.getvalue()
        print(f"First call warnings: {len(first_warnings)} characters")
        print("Sample warning:", first_warnings[:200] + "..." if len(first_warnings) > 200 else first_warnings)
        
        # Second call - should NOT repeat warnings
        log_capture.truncate(0)
        log_capture.seek(0)
        
        events2 = bump(
            snapshot=snapshot,
            hazard_slice=hazard_conflicts,
            as_of=pd.Timestamp("2025-01-01"),
            rng=np.random.default_rng(0)
        )
        
        second_warnings = log_capture.getvalue()
        print(f"Second call warnings: {len(second_warnings)} characters")
        
        # Verify warning suppression
        if len(first_warnings) > 100 and len(second_warnings) < 50:
            print("✅ SUCCESS: Warnings properly suppressed on repeated calls")
        else:
            print(f"❌ FAILURE: Warning suppression not working. First: {len(first_warnings)}, Second: {len(second_warnings)}")
            return False
        
        print("\n=== Test 2: Clear cache and call again (should warn again) ===")
        clear_warning_cache()
        log_capture.truncate(0)
        log_capture.seek(0)
        
        events3 = bump(
            snapshot=snapshot,
            hazard_slice=hazard_conflicts,
            as_of=pd.Timestamp("2025-01-01"),
            rng=np.random.default_rng(0)
        )
        
        third_warnings = log_capture.getvalue()
        print(f"Third call warnings (after cache clear): {len(third_warnings)} characters")
        
        if len(third_warnings) > 100:
            print("✅ SUCCESS: Warnings reappear after cache clear")
        else:
            print("❌ FAILURE: Warnings not reappearing after cache clear")
            return False
        
        print("\n=== Test 3: Harmless duplicates (should be quiet) ===")
        clear_warning_cache()
        log_capture.truncate(0)
        log_capture.seek(0)
        
        # Create hazard table with harmless duplicates (same rates)
        hazard_harmless = pd.DataFrame({
            "simulation_year": [2025, 2025, 2025, 2025],
            "employee_level": [0, 0, 1, 1],
            "employee_tenure_band": ["3-5", "3-5", "5-10", "5-10"],
            "comp_raise_pct": [0.03, 0.03, 0.015, 0.015],  # Same rates
            "cola_pct": [0.02, 0.02, 0.02, 0.02],
            "term_rate": [0.1, 0.1, 0.1, 0.1],
            "new_hire_termination_rate": [0.25, 0.25, 0.25, 0.25],
            "cfg": [SimpleNamespace()] * 4,
        })
        
        events4 = bump(
            snapshot=snapshot,
            hazard_slice=hazard_harmless,
            as_of=pd.Timestamp("2025-01-01"),
            rng=np.random.default_rng(0)
        )
        
        harmless_warnings = log_capture.getvalue()
        print(f"Harmless duplicates warnings: {len(harmless_warnings)} characters")
        
        if len(harmless_warnings) < 50:  # Should be minimal or no warnings
            print("✅ SUCCESS: Harmless duplicates handled quietly")
        else:
            print("❌ FAILURE: Harmless duplicates generating excessive warnings")
            print("Warnings:", harmless_warnings)
            return False
        
        print("\n=== Test 4: Event generation still works ===")
        # Verify that events are still generated correctly despite the fixes
        total_events = sum(len(df) for df in events1 if not df.empty)
        print(f"Total events generated: {total_events}")
        
        if total_events > 0:
            print("✅ SUCCESS: Events still generated correctly")
        else:
            print("❌ FAILURE: No events generated")
            return False
        
        # Clean up
        comp_logger.removeHandler(handler)
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Conflicting rates generate warnings (once per simulation)")
        print("✅ Repeated warnings are suppressed")
        print("✅ Warning cache can be cleared")
        print("✅ Harmless duplicates are handled quietly")
        print("✅ Event generation still works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_final_duplicates_fix()
    if success:
        print("\n🎉 Final duplicates fix is working perfectly!")
        sys.exit(0)
    else:
        print("\n💥 Final duplicates fix needs more work.")
        sys.exit(1)
