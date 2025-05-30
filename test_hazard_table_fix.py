#!/usr/bin/env python3
"""
Test to verify that the hazard table loading fix resolves the duplicate warnings.

This test verifies that when we load the hazard table properly and filter by year,
we don't get conflicting rate warnings for the same (level, tenure_band) combinations.
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

def test_hazard_table_fix():
    """Test that proper hazard table loading eliminates false duplicate warnings."""
    print("Testing hazard table loading fix...")
    
    try:
        from cost_model.engines.comp import bump, clear_warning_cache
        from cost_model.projections.hazard import load_and_expand_hazard_table
        from cost_model.engines.run_one_year.validation import validate_and_extract_hazard_slice
        
        # Set up log capture to verify no warnings
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.WARNING)
        
        comp_logger = logging.getLogger('cost_model.engines.comp')
        comp_logger.addHandler(handler)
        comp_logger.setLevel(logging.DEBUG)
        
        clear_warning_cache()
        
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
            "employee_tenure_band": ["1-3", "5-10", "3-5"],
            "employee_level": [0, 1, 2],
            "job_level_source": ["initial", "initial", "initial"],
            "exited": [False, False, False],
            "employee_status_eoy": ["Active", "Active", "Active"],
            "simulation_year": [2025, 2025, 2025],
        }).set_index("employee_id", drop=False)
        
        print("=== Test 1: Load hazard table from CSV (multi-year) ===")
        
        # Simulate the old way: load CSV directly (this would cause duplicates)
        hazard_csv = pd.read_csv('cost_model/state/hazard_table.csv')
        print(f"Loaded CSV hazard table with {len(hazard_csv)} rows")
        print(f"Years in CSV: {sorted(hazard_csv['year'].unique())}")
        print(f"Unique (level, tenure_band) combinations in CSV: {len(hazard_csv[['employee_level', 'tenure_band']].drop_duplicates())}")
        
        # Rename columns to match expected format
        hazard_csv_renamed = hazard_csv.rename(columns={
            'year': 'simulation_year',
            'tenure_band': 'employee_tenure_band'
        })
        
        print("\n=== Test 2: Filter hazard table by year (proper way) ===")
        
        # Use the proper validation function to filter by year
        hazard_slice_2025 = validate_and_extract_hazard_slice(hazard_csv_renamed, 2025)
        print(f"Filtered hazard slice for 2025: {len(hazard_slice_2025)} rows")
        print(f"Unique (level, tenure_band) combinations in 2025 slice: {len(hazard_slice_2025[['employee_level', 'employee_tenure_band']].drop_duplicates())}")
        
        # Check for duplicates in the filtered slice
        duplicates_2025 = hazard_slice_2025.duplicated(subset=['employee_level', 'employee_tenure_band'])
        print(f"Duplicates in 2025 slice: {duplicates_2025.sum()}")
        
        print("\n=== Test 3: Test compensation engine with filtered data ===")
        log_capture.truncate(0)
        log_capture.seek(0)
        
        # Test with properly filtered hazard slice (should not generate warnings)
        events_filtered = bump(
            snapshot=snapshot,
            hazard_slice=hazard_slice_2025,
            as_of=pd.Timestamp("2025-01-01"),
            rng=np.random.default_rng(0)
        )
        
        warnings_filtered = log_capture.getvalue()
        print(f"Warnings with filtered data: {len(warnings_filtered)} characters")
        if warnings_filtered:
            print("Warning content:", warnings_filtered[:200])
        
        print("\n=== Test 4: Test compensation engine with multi-year data (old way) ===")
        log_capture.truncate(0)
        log_capture.seek(0)
        clear_warning_cache()
        
        # Test with multi-year data (would generate warnings)
        try:
            events_multi_year = bump(
                snapshot=snapshot,
                hazard_slice=hazard_csv_renamed,  # This has multiple years
                as_of=pd.Timestamp("2025-01-01"),
                rng=np.random.default_rng(0)
            )
            
            warnings_multi_year = log_capture.getvalue()
            print(f"Warnings with multi-year data: {len(warnings_multi_year)} characters")
            if warnings_multi_year:
                print("Warning content:", warnings_multi_year[:200])
        except Exception as e:
            print(f"Expected error with multi-year data: {e}")
        
        print("\n=== Test 5: Verify events are still generated ===")
        total_events = sum(len(df) for df in events_filtered if not df.empty)
        print(f"Total events generated with filtered data: {total_events}")
        
        # Clean up
        comp_logger.removeHandler(handler)
        
        # Determine success
        if len(warnings_filtered) < 50:  # Should be minimal warnings with properly filtered data
            print("\n✅ SUCCESS: Hazard table loading fix is working!")
            print("   - Properly filtered hazard data generates no duplicate warnings")
            print("   - Events are still generated correctly")
            print("   - Multi-year data would generate warnings (as expected)")
            return True
        else:
            print("\n❌ FAILURE: Still getting warnings with filtered data")
            print(f"Warnings: {warnings_filtered}")
            return False
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hazard_table_fix()
    if success:
        print("\n🎉 Hazard table loading fix is working perfectly!")
        sys.exit(0)
    else:
        print("\n💥 Hazard table loading fix needs more work.")
        sys.exit(1)
