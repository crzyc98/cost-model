#!/usr/bin/env python3
"""
Test script to verify the comp.py refactoring from EMP_ROLE to EMP_LEVEL works correctly.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_comp_bump_with_emp_level():
    """Test that the bump function works with EMP_LEVEL instead of EMP_ROLE."""
    print("Testing comp.bump function with EMP_LEVEL...")
    
    try:
        from cost_model.engines.comp import bump
        from cost_model.state.schema import (
            EMP_ID, EMP_TERM_DATE, EMP_LEVEL, EMP_GROSS_COMP, 
            EMP_HIRE_DATE, EMP_TENURE_BAND, SIMULATION_YEAR
        )
        
        # Create test snapshot with EMP_LEVEL instead of EMP_ROLE
        snapshot = pd.DataFrame({
            EMP_ID: ['E001', 'E002', 'E003'],
            EMP_LEVEL: [1, 2, 1],  # Using EMP_LEVEL instead of EMP_ROLE
            EMP_GROSS_COMP: [50000.0, 75000.0, 60000.0],
            EMP_HIRE_DATE: ['2020-01-01', '2019-01-01', '2021-01-01'],
            EMP_TERM_DATE: [pd.NaT, pd.NaT, pd.NaT],
            EMP_TENURE_BAND: ['1-2', '3-5', '1-2']
        })
        
        # Create test hazard_slice with 'level' column instead of 'role'
        hazard_slice = pd.DataFrame({
            'level': [1, 2],  # Using 'level' instead of 'role'
            EMP_TENURE_BAND: ['1-2', '3-5'],
            'comp_raise_pct': [0.03, 0.05],
            'simulation_year': [2025, 2025],
            'cola_pct': [0.02, 0.02]
        })
        
        as_of = pd.Timestamp('2025-01-01')
        rng = np.random.default_rng(42)
        
        # Test the bump function
        result = bump(snapshot, hazard_slice, as_of, rng)
        
        print(f"✓ bump function executed successfully")
        print(f"  Returned {len(result)} event DataFrames")
        
        # Check the first DataFrame (comp events)
        if result and len(result) > 0 and not result[0].empty:
            comp_events = result[0]
            print(f"  Generated {len(comp_events)} compensation events")
            print(f"  Event types: {comp_events['event_type'].unique().tolist()}")
        
        # Check for COLA events (should be in additional DataFrames)
        total_events = sum(len(df) for df in result if not df.empty)
        print(f"  Total events across all DataFrames: {total_events}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing comp.bump: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_imports():
    """Test that the imports work correctly after refactoring."""
    print("Testing imports...")
    
    try:
        from cost_model.engines.comp import bump, generate_cola_events
        from cost_model.engines.cola import cola
        from cost_model.state.schema import EMP_LEVEL, EMP_TENURE_BAND
        
        print("✓ All imports successful")
        print(f"  EMP_LEVEL constant: '{EMP_LEVEL}'")
        print(f"  EMP_TENURE_BAND constant: '{EMP_TENURE_BAND}'")
        
        return True
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing comp.py refactoring from EMP_ROLE to EMP_LEVEL")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_comp_bump_with_emp_level,
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
        print("✓ All tests passed! Refactoring appears successful.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
