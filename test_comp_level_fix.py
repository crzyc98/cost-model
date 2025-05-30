#!/usr/bin/env python3
"""
Test script to verify the comp.py level column fix works correctly.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_dynamic_level_column_selection():
    """Test that the bump function works with different level column names."""
    print("Testing dynamic level column selection...")
    
    try:
        from cost_model.engines.comp import bump
        from cost_model.state.schema import (
            EMP_ID, EMP_TERM_DATE, EMP_LEVEL, EMP_GROSS_COMP, 
            EMP_HIRE_DATE, EMP_TENURE_BAND, SIMULATION_YEAR
        )
        
        # Create test snapshot with EMP_LEVEL
        snapshot = pd.DataFrame({
            EMP_ID: ['E001', 'E002', 'E003'],
            EMP_LEVEL: [1, 2, 1],
            EMP_GROSS_COMP: [50000.0, 75000.0, 60000.0],
            EMP_HIRE_DATE: ['2020-01-01', '2019-01-01', '2021-01-01'],
            EMP_TERM_DATE: [pd.NaT, pd.NaT, pd.NaT],
            EMP_TENURE_BAND: ['1-2', '3-5', '1-2']
        })
        
        # Test 1: hazard_slice with 'level' column
        print("  Test 1: hazard_slice with 'level' column")
        hazard_slice_1 = pd.DataFrame({
            'level': [1, 2],
            EMP_TENURE_BAND: ['1-2', '3-5'],
            'comp_raise_pct': [0.03, 0.05],
            'simulation_year': [2025, 2025],
            'cola_pct': [0.02, 0.02]
        })
        
        as_of = pd.Timestamp('2025-01-01')
        rng = np.random.default_rng(42)
        
        result1 = bump(snapshot, hazard_slice_1, as_of, rng)
        print(f"    ✓ Success with 'level' column: {len(result1)} event DataFrames")
        
        # Test 2: hazard_slice with 'employee_level' column
        print("  Test 2: hazard_slice with 'employee_level' column")
        hazard_slice_2 = pd.DataFrame({
            'employee_level': [1, 2],
            EMP_TENURE_BAND: ['1-2', '3-5'],
            'comp_raise_pct': [0.03, 0.05],
            'simulation_year': [2025, 2025],
            'cola_pct': [0.02, 0.02]
        })
        
        result2 = bump(snapshot, hazard_slice_2, as_of, rng)
        print(f"    ✓ Success with 'employee_level' column: {len(result2)} event DataFrames")
        
        # Test 3: hazard_slice with 'job_level' column (should be found by fallback)
        print("  Test 3: hazard_slice with 'job_level' column")
        hazard_slice_3 = pd.DataFrame({
            'job_level': [1, 2],
            EMP_TENURE_BAND: ['1-2', '3-5'],
            'comp_raise_pct': [0.03, 0.05],
            'simulation_year': [2025, 2025],
            'cola_pct': [0.02, 0.02]
        })
        
        result3 = bump(snapshot, hazard_slice_3, as_of, rng)
        print(f"    ✓ Success with 'job_level' column: {len(result3)} event DataFrames")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test that the error handling works when no level column is found."""
    print("Testing error handling for missing level column...")
    
    try:
        from cost_model.engines.comp import bump
        from cost_model.state.schema import (
            EMP_ID, EMP_TERM_DATE, EMP_LEVEL, EMP_GROSS_COMP, 
            EMP_HIRE_DATE, EMP_TENURE_BAND, SIMULATION_YEAR
        )
        
        # Create test snapshot
        snapshot = pd.DataFrame({
            EMP_ID: ['E001', 'E002'],
            EMP_LEVEL: [1, 2],
            EMP_GROSS_COMP: [50000.0, 75000.0],
            EMP_HIRE_DATE: ['2020-01-01', '2019-01-01'],
            EMP_TERM_DATE: [pd.NaT, pd.NaT],
            EMP_TENURE_BAND: ['1-2', '3-5']
        })
        
        # Create hazard_slice with NO level column
        hazard_slice_bad = pd.DataFrame({
            'role': ['A', 'B'],  # Wrong column name
            EMP_TENURE_BAND: ['1-2', '3-5'],
            'comp_raise_pct': [0.03, 0.05],
            'simulation_year': [2025, 2025],
            'cola_pct': [0.02, 0.02]
        })
        
        as_of = pd.Timestamp('2025-01-01')
        rng = np.random.default_rng(42)
        
        try:
            result = bump(snapshot, hazard_slice_bad, as_of, rng)
            print(f"    ✗ Expected error but got result: {result}")
            return False
        except KeyError as e:
            if "No level column found" in str(e):
                print(f"    ✓ Correctly caught error: {e}")
                return True
            else:
                print(f"    ✗ Unexpected KeyError: {e}")
                return False
        
    except Exception as e:
        print(f"    ✗ Unexpected error: {e}")
        return False

def test_orchestrator_error_handling():
    """Test that the orchestrator error handling works."""
    print("Testing orchestrator error handling...")
    
    try:
        from cost_model.engines.run_one_year.orchestrator import _generate_compensation_events
        from cost_model.engines.run_one_year.orchestrator.base import YearContext
        from cost_model.state.schema import (
            EMP_ID, EMP_TERM_DATE, EMP_LEVEL, EMP_GROSS_COMP, 
            EMP_HIRE_DATE, EMP_TENURE_BAND, SIMULATION_YEAR
        )
        import logging
        
        # Create test snapshot
        snapshot = pd.DataFrame({
            EMP_ID: ['E001', 'E002'],
            EMP_LEVEL: [1, 2],
            EMP_GROSS_COMP: [50000.0, 75000.0],
            EMP_HIRE_DATE: ['2020-01-01', '2019-01-01'],
            EMP_TERM_DATE: [pd.NaT, pd.NaT],
            EMP_TENURE_BAND: ['1-2', '3-5']
        })
        
        # Create hazard_slice with NO level column
        hazard_slice_bad = pd.DataFrame({
            'role': ['A', 'B'],  # Wrong column name
            EMP_TENURE_BAND: ['1-2', '3-5'],
            'comp_raise_pct': [0.03, 0.05],
            'simulation_year': [2025, 2025],
            'cola_pct': [0.02, 0.02]
        })
        
        # Create year context
        year_context = YearContext(
            year=2025,
            as_of=pd.Timestamp('2025-01-01'),
            hazard_slice=hazard_slice_bad,
            year_rng=np.random.default_rng(42)
        )
        
        # Create logger
        logger = logging.getLogger(__name__)
        
        # Test the orchestrator function
        result = _generate_compensation_events(snapshot, year_context, logger)
        
        # Should return empty events instead of crashing
        print(f"    ✓ Orchestrator handled error gracefully: {len(result)} event lists")
        return True
        
    except Exception as e:
        print(f"    ✗ Orchestrator error handling failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing comp.py level column fix")
    print("=" * 60)
    
    tests = [
        test_dynamic_level_column_selection,
        test_error_handling,
        test_orchestrator_error_handling,
    ]
    
    results = []
    for test in tests:
        print()
        result = test()
        results.append(result)
    
    print()
    print("=" * 60)
    print("Test Results:")
    print(f"  Passed: {sum(results)}/{len(results)}")
    print(f"  Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("✓ All tests passed! Level column fix is working.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
