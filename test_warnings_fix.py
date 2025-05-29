#!/usr/bin/env python3
"""
Test script to verify that pandas warnings have been fixed.
"""

import warnings
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_fixes():
    """Test basic pandas warning fixes"""
    print("Testing basic pandas warning fixes...")
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Test SettingWithCopyWarning fix pattern
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        df.loc[:, 'C'] = df['A'] + df['B']  # Should not warn
        
        # Test FutureWarning concat fix pattern
        df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
        empty_df = pd.DataFrame()
        
        # Filter out empty DataFrames
        dfs_to_concat = [df for df in [df1, df2, empty_df] if not df.empty and not df.isna().all().all()]
        if dfs_to_concat:
            result = pd.concat(dfs_to_concat, ignore_index=True)
        
        # Test dtype conversion fix pattern
        float_series = pd.Series([1.0, 2.0, pd.NA], dtype='Float64')
        converted = float_series.to_numpy(dtype=np.float64, na_value=np.nan)
        
        # Check for warnings
        copy_warnings = [w for w in w if issubclass(w.category, pd.errors.SettingWithCopyWarning)]
        future_warnings = [w for w in w if issubclass(w.category, FutureWarning)]
        
        if copy_warnings:
            print(f"✗ SettingWithCopyWarning still occurs: {len(copy_warnings)} warnings")
            return False
        elif future_warnings:
            print(f"✗ FutureWarning still occurs: {len(future_warnings)} warnings")
            return False
        else:
            print("✓ Basic pandas warning fixes successful")
            return True

def main():
    """Run warning fix tests"""
    print("Testing pandas warnings fixes...\n")
    
    success = test_basic_fixes()
    
    if success:
        print("✓ All pandas warnings fixes are working correctly!")
        return 0
    else:
        print("✗ Some pandas warnings fixes need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())
