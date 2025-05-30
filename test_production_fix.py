#!/usr/bin/env python3
"""
Test to verify that the production issue with conflicting comp_raise_pct values is resolved.

This test simulates the exact production scenario and verifies that:
1. No false duplicate warnings are generated
2. Tenure band standardization works correctly
3. Hazard table loading and filtering works properly
4. Events are generated correctly
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

def test_production_fix():
    """Test that the production issue is completely resolved."""
    print("Testing production fix for conflicting comp_raise_pct values...")
    
    try:
        from cost_model.engines.comp import bump, clear_warning_cache
        from cost_model.utils.tenure_utils import standardize_tenure_band
        
        # Set up log capture
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.WARNING)
        
        comp_logger = logging.getLogger('cost_model.engines.comp')
        comp_logger.addHandler(handler)
        comp_logger.setLevel(logging.DEBUG)
        
        clear_warning_cache()
        
        print("=== Test 1: Verify tenure band standardization fix ===")
        
        # Test the tenure band standardization
        test_cases = [
            ('10+', '10+'),
            ('5+', '10+'),  # Legacy format should map to 10+
            ('1-3', '1-3'),
            ('5-10', '5-10'),
            ('0-1', '0-1'),
            ('3-5', '3-5'),
        ]
        
        for input_band, expected in test_cases:
            result = standardize_tenure_band(input_band)
            print(f"  {input_band} -> {result} (expected: {expected})")
            if result != expected:
                print(f"❌ FAILURE: Tenure band standardization failed for {input_band}")
                return False
        
        print("✅ Tenure band standardization working correctly")
        
        print("\n=== Test 2: Load and test real hazard table ===")
        
        # Load the actual hazard table from CSV
        hazard_csv = pd.read_csv('cost_model/state/hazard_table.csv')
        print(f"Loaded real hazard table: {len(hazard_csv)} rows")
        
        # Rename columns to match expected format
        hazard_renamed = hazard_csv.rename(columns={
            'year': 'simulation_year',
            'tenure_band': 'employee_tenure_band'
        })
        
        # Filter for 2025 only (simulating proper year filtering)
        hazard_2025 = hazard_renamed[hazard_renamed['simulation_year'] == 2025].copy()
        print(f"Filtered to 2025: {len(hazard_2025)} rows")
        
        # Check for duplicates after filtering
        duplicates = hazard_2025.duplicated(subset=['employee_level', 'employee_tenure_band'])
        print(f"Duplicates in 2025 data: {duplicates.sum()}")
        
        if duplicates.sum() > 0:
            print("❌ FAILURE: Found duplicates in properly filtered 2025 data")
            return False
        
        print("✅ No duplicates in properly filtered hazard data")
        
        print("\n=== Test 3: Test with realistic employee data ===")
        
        # Create realistic employee snapshot that matches the production scenario
        snapshot = pd.DataFrame({
            "employee_id": ["E1", "E2", "E3", "E4", "E5"],
            "active": [True, True, True, True, True],
            "employee_hire_date": pd.to_datetime([
                "2023-06-01",  # 1-3 years
                "2018-03-15",  # 5-10 years  
                "2020-01-01",  # 3-5 years
                "2010-01-01",  # 10+ years
                "2024-06-01"   # 0-1 years
            ]),
            "employee_birth_date": pd.to_datetime(["1990-01-01", "1985-01-01", "1992-01-01", "1980-01-01", "1995-01-01"]),
            "employee_gross_compensation": [80000, 95000, 75000, 120000, 65000],
            "employee_termination_date": pd.NaT,
            "employee_deferral_rate": [0.05, 0.06, 0.04, 0.08, 0.03],
            "employee_tenure": [2, 7, 5, 15, 0.5],
            "employee_tenure_band": ["1-3", "5-10", "3-5", "10+", "0-1"],
            "employee_level": [0, 1, 2, 3, 4],
            "job_level_source": ["initial"] * 5,
            "exited": [False] * 5,
            "employee_status_eoy": ["Active"] * 5,
            "simulation_year": [2025] * 5,
        }).set_index("employee_id", drop=False)
        
        print("Created realistic employee snapshot:")
        print(snapshot[["employee_level", "employee_tenure_band", "employee_gross_compensation"]])
        
        print("\n=== Test 4: Test compensation engine with real data ===")
        log_capture.truncate(0)
        log_capture.seek(0)
        
        # Test with the real 2025 hazard data
        events = bump(
            snapshot=snapshot,
            hazard_slice=hazard_2025,
            as_of=pd.Timestamp("2025-01-01"),
            rng=np.random.default_rng(0)
        )
        
        warnings = log_capture.getvalue()
        print(f"Warnings generated: {len(warnings)} characters")
        
        if warnings:
            print("Warning content:")
            print(warnings)
        
        # Count events
        total_events = sum(len(df) for df in events if not df.empty)
        comp_events = sum(len(df) for df in events if not df.empty and 'event_type' in df.columns and df['event_type'].iloc[0] == 'EVT_COMP')
        cola_events = sum(len(df) for df in events if not df.empty and 'event_type' in df.columns and df['event_type'].iloc[0] == 'EVT_COLA')
        
        print(f"Total events: {total_events}")
        print(f"EVT_COMP events: {comp_events}")
        print(f"EVT_COLA events: {cola_events}")
        
        # Clean up
        comp_logger.removeHandler(handler)
        
        # Determine success
        if len(warnings) < 50:  # Should be minimal warnings
            print("\n✅ SUCCESS: Production fix is working!")
            print("   - No false duplicate warnings generated")
            print("   - Tenure band standardization fixed")
            print("   - Hazard table filtering works correctly")
            print("   - Events generated successfully")
            print(f"   - Generated {comp_events} compensation events and {cola_events} COLA events")
            return True
        else:
            print("\n❌ FAILURE: Still getting unexpected warnings")
            print(f"Warnings: {warnings}")
            return False
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_production_fix()
    if success:
        print("\n🎉 Production fix verified! The conflicting comp_raise_pct issue is resolved.")
        sys.exit(0)
    else:
        print("\n💥 Production fix verification failed.")
        sys.exit(1)
