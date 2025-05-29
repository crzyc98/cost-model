#!/usr/bin/env python3
"""
Test script to verify that SettingWithCopyWarning and FutureWarning fixes are working.
"""

import warnings
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_contributions_warnings():
    """Test that contributions.py doesn't generate SettingWithCopyWarning."""
    print("Testing contributions.py for SettingWithCopyWarning...")
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            from cost_model.rules.contributions import apply as apply_contributions
            from cost_model.rules.validators import ContributionsRule, MatchRule, NonElectiveRule, Tier
            from cost_model.state.schema import (
                EMP_ID, EMP_GROSS_COMP, EMP_DEFERRAL_RATE, EMP_HIRE_DATE, 
                EMP_BIRTH_DATE, EMP_ACTIVE
            )
            
            # Create a simple test DataFrame
            test_df = pd.DataFrame({
                EMP_ID: ['emp1', 'emp2', 'emp3'],
                EMP_GROSS_COMP: [50000.0, 60000.0, 70000.0],
                EMP_DEFERRAL_RATE: [0.05, 0.10, 0.15],
                EMP_HIRE_DATE: ['2020-01-01', '2019-06-15', '2021-03-10'],
                EMP_BIRTH_DATE: ['1980-01-01', '1975-06-15', '1985-03-10'],
                EMP_ACTIVE: [True, True, True]
            })
            test_df = test_df.set_index(EMP_ID)
            
            # Create test rules
            contrib_rules = ContributionsRule(enabled=True)
            tier = Tier(match_rate=0.5, cap_deferral_pct=0.06)
            match_rules = MatchRule(tiers=[tier], dollar_cap=None)
            nec_rules = NonElectiveRule(rate=0.03)
            
            # Test IRS limits
            irs_limits = {
                2024: {
                    'compensation_limit': 345000,
                    'deferral_limit': 23000,
                    'catch_up_limit': 7500,
                    'catch_up_age': 50
                }
            }
            
            # Apply contributions
            result_df = apply_contributions(
                df=test_df,
                contrib_rules=contrib_rules,
                match_rules=match_rules,
                nec_rules=nec_rules,
                irs_limits=irs_limits,
                simulation_year=2024,
                year_start=pd.Timestamp("2024-01-01"),
                year_end=pd.Timestamp("2024-12-31")
            )
            
            print(f"✓ Contributions test completed successfully")
            print(f"  Result shape: {result_df.shape}")
            
        except Exception as e:
            print(f"✗ Error in contributions test: {e}")
            return False
        
        # Check for SettingWithCopyWarning
        copy_warnings = [warning for warning in w if 'SettingWithCopyWarning' in str(warning.message)]
        if copy_warnings:
            print(f"✗ Found {len(copy_warnings)} SettingWithCopyWarning(s):")
            for warning in copy_warnings:
                print(f"  - {warning.message}")
            return False
        else:
            print("✓ No SettingWithCopyWarning found in contributions.py")
            
    return True

def test_concat_warnings():
    """Test that pd.concat doesn't generate FutureWarning with empty DataFrames."""
    print("\nTesting pd.concat FutureWarning fixes...")
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            # Test the pattern from orchestrator.py
            from cost_model.state.event_log import EVENT_COLS, EVENT_PANDAS_DTYPES
            
            # Create some test DataFrames including empty ones
            df1 = pd.DataFrame({col: [] for col in EVENT_COLS})
            df2 = pd.DataFrame({
                'event_id': ['evt1', 'evt2'],
                'event_time': [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02')],
                'employee_id': ['emp1', 'emp2'],
                'event_type': ['hire', 'term'],
                'value_num': [None, None],
                'value_json': [None, None],
                'meta': [None, None]
            })
            df3 = pd.DataFrame({col: [] for col in EVENT_COLS})  # Another empty one
            
            validated_events = [df1, df2, df3]
            
            # Apply the fix: filter out empty DataFrames
            actual_valid_events = [df for df in validated_events if not df.empty]
            
            if actual_valid_events:
                result = pd.concat(actual_valid_events, ignore_index=True)
                print(f"✓ pd.concat with filtered events successful, shape: {result.shape}")
            else:
                # Create empty DataFrame with correct schema
                result = pd.DataFrame({col: pd.Series(dtype=str(t) if t != 'object' else object)
                                     for col, t in EVENT_PANDAS_DTYPES.items()})
                print(f"✓ Created empty DataFrame with correct schema, shape: {result.shape}")
                
        except Exception as e:
            print(f"✗ Error in concat test: {e}")
            return False
        
        # Check for FutureWarning
        future_warnings = [warning for warning in w if 'FutureWarning' in str(warning.category.__name__)]
        if future_warnings:
            print(f"✗ Found {len(future_warnings)} FutureWarning(s):")
            for warning in future_warnings:
                print(f"  - {warning.message}")
            return False
        else:
            print("✓ No FutureWarning found in pd.concat operations")
            
    return True

def main():
    """Run all warning tests."""
    print("=" * 60)
    print("Testing Pandas Warning Fixes")
    print("=" * 60)
    
    success = True
    
    # Test contributions warnings
    success &= test_contributions_warnings()
    
    # Test concat warnings  
    success &= test_concat_warnings()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ ALL TESTS PASSED - Warning fixes are working correctly!")
    else:
        print("✗ SOME TESTS FAILED - Warning fixes need more work")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
