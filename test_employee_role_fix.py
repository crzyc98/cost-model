#!/usr/bin/env python3
"""
Test script to verify the fix for the employee_role PyArrow conversion error.

This script reproduces the original error condition and verifies that the fix
in cost_model/projections/reporting.py correctly handles mixed data types
in the employee_role column.
"""

import pandas as pd
import numpy as np
import tempfile
import logging
from pathlib import Path
from types import SimpleNamespace

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_problematic_snapshot():
    """Create a snapshot DataFrame with mixed data types in employee_role column."""
    # Create a DataFrame that reproduces the original error condition
    data = {
        'employee_id': ['emp_001', 'emp_002', 'emp_003', 'emp_004'],
        'employee_hire_date': pd.to_datetime(['2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01']),
        'employee_birth_date': pd.to_datetime(['1990-01-01', '1991-01-01', '1992-01-01', '1993-01-01']),
        'employee_gross_compensation': [50000.0, 60000.0, 70000.0, 80000.0],
        'employee_termination_date': [pd.NaT, pd.NaT, pd.NaT, pd.NaT],
        'active': [True, True, True, True],
        'employee_deferral_rate': [0.05, 0.06, 0.07, 0.08],
        'employee_tenure': [4.0, 3.0, 2.0, 1.0],
        'employee_tenure_band': ['3-5', '1-3', '1-3', '0-1'],
        'employee_level': [1, 2, 1, 1],
        'job_level_source': ['hire', 'promotion', 'hire', 'hire'],
        'exited': [False, False, False, False],
        'simulation_year': [2024, 2024, 2024, 2024],
        # This is the problematic column - mixed integers and strings
        'employee_role': ['Manager', 1, 'Staff', 2]  # Mixed types!
    }
    
    return pd.DataFrame(data)

def test_original_error():
    """Test that the original error would occur without the fix."""
    logger.info("Testing original error condition...")
    
    snapshot = create_problematic_snapshot()
    
    # Check that we indeed have mixed types
    role_types = [type(x).__name__ for x in snapshot['employee_role']]
    logger.info(f"employee_role column types: {role_types}")
    
    # Try to save to parquet without the fix - this should fail
    try:
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            snapshot.to_parquet(tmp.name, index=False)
            logger.error("ERROR: Expected PyArrow conversion error did not occur!")
            return False
    except Exception as e:
        if "Expected bytes, got a 'int' object" in str(e):
            logger.info(f"✓ Successfully reproduced original error: {e}")
            return True
        else:
            logger.error(f"Unexpected error: {e}")
            return False

def test_fix_works():
    """Test that the fix in save_detailed_results works correctly."""
    logger.info("Testing that the fix works...")
    
    # Import the fixed function
    from cost_model.projections.reporting import save_detailed_results
    
    # Create problematic data
    snapshot = create_problematic_snapshot()
    yearly_snapshots = {2024: snapshot}
    
    # Create minimal required data for save_detailed_results
    final_snapshot = snapshot.copy()
    full_event_log = pd.DataFrame({
        'event_id': ['e1', 'e2'],
        'event_time': pd.to_datetime(['2024-01-01', '2024-06-01']),
        'employee_id': ['emp_001', 'emp_002'],
        'event_type': ['EVT_HIRE', 'EVT_COMP'],
        'value_num': [None, 5000],
        'value_json': [None, None],
        'meta': [None, None],
        'simulation_year': [2024, 2024]
    })
    
    summary_statistics = pd.DataFrame({
        'Year': [2024],
        'Active Headcount': [4],
        'Total Compensation': [260000]
    })
    
    employment_status_summary = pd.DataFrame({
        'Year': [2024],
        'Active': [4],
        'Terminated': [0]
    })
    
    # Test the fix
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir)
            
            save_detailed_results(
                output_path=output_path,
                scenario_name="test_scenario",
                final_snapshot=final_snapshot,
                full_event_log=full_event_log,
                summary_statistics=summary_statistics,
                employment_status_summary_df=employment_status_summary,
                yearly_snapshots=yearly_snapshots,
                config_to_save=None
            )
            
            # Verify files were created successfully
            yearly_dir = output_path / "yearly_snapshots"
            snapshot_file = yearly_dir / "test_scenario_snapshot_2024.parquet"
            final_file = output_path / "test_scenario_final_eoy_snapshot.parquet"
            
            if snapshot_file.exists() and final_file.exists():
                logger.info("✓ Files created successfully!")
                
                # Read back and verify the employee_role column is properly typed
                saved_snapshot = pd.read_parquet(snapshot_file)
                saved_final = pd.read_parquet(final_file)
                
                logger.info(f"Saved snapshot employee_role dtype: {saved_snapshot['employee_role'].dtype}")
                logger.info(f"Saved final employee_role dtype: {saved_final['employee_role'].dtype}")
                logger.info(f"Saved snapshot employee_role values: {saved_snapshot['employee_role'].tolist()}")
                
                # Verify all values are strings now
                all_strings = all(isinstance(x, str) or pd.isna(x) for x in saved_snapshot['employee_role'])
                if all_strings:
                    logger.info("✓ All employee_role values are properly converted to strings!")
                    return True
                else:
                    logger.error("✗ Some employee_role values are still not strings")
                    return False
            else:
                logger.error("✗ Expected files were not created")
                return False
                
    except Exception as e:
        logger.error(f"✗ Fix failed with error: {e}")
        return False

def main():
    """Run the test suite."""
    logger.info("=" * 60)
    logger.info("Testing employee_role PyArrow conversion fix")
    logger.info("=" * 60)
    
    # Test 1: Verify we can reproduce the original error
    original_error_reproduced = test_original_error()
    
    # Test 2: Verify the fix works
    fix_works = test_fix_works()
    
    logger.info("=" * 60)
    logger.info("Test Results:")
    logger.info(f"Original error reproduced: {'✓' if original_error_reproduced else '✗'}")
    logger.info(f"Fix works correctly: {'✓' if fix_works else '✗'}")
    
    if original_error_reproduced and fix_works:
        logger.info("🎉 All tests passed! The fix is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())
