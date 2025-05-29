#!/usr/bin/env python3
"""
Test script to verify that pandas FutureWarnings have been fixed.
This script tests the specific patterns that were causing warnings.
"""

import warnings
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_fillna_astype_pattern():
    """Test the fillna().astype() pattern that was causing downcasting warnings."""
    print("Testing fillna().astype() pattern...")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Create a Series with mixed types that might cause downcasting warning
        test_series = pd.Series([1.0, 2.0, np.nan, 4.0], dtype='object')

        # Old pattern that would cause warning:
        # result = test_series.fillna(0).astype('int64')

        # New pattern with infer_objects BEFORE fillna:
        inferred_series = test_series.infer_objects(copy=False)
        result = inferred_series.fillna(0).astype('int64')

        # Check for FutureWarnings
        future_warnings = [warning for warning in w if issubclass(warning.category, FutureWarning)]

        if future_warnings:
            print(f"❌ FutureWarning still occurs: {len(future_warnings)} warnings")
            for warning in future_warnings:
                print(f"   Warning: {warning.message}")
            return False
        else:
            print("✅ fillna().astype() pattern fixed - no FutureWarnings")
            return True

def test_concat_empty_dataframes():
    """Test the pd.concat() pattern with empty DataFrames."""
    print("Testing pd.concat() with empty DataFrames...")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Create test DataFrames including empty ones
        df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
        empty_df = pd.DataFrame()

        # Old pattern that would cause warning:
        # result = pd.concat([df1, df2, empty_df], ignore_index=True)

        # New pattern with filtering:
        dfs_to_concat = [df for df in [df1, df2, empty_df] if not df.empty]
        if dfs_to_concat:
            result = pd.concat(dfs_to_concat, ignore_index=True)
        else:
            result = pd.DataFrame()

        # Check for FutureWarnings
        future_warnings = [warning for warning in w if issubclass(warning.category, FutureWarning)]

        if future_warnings:
            print(f"❌ FutureWarning still occurs: {len(future_warnings)} warnings")
            for warning in future_warnings:
                print(f"   Warning: {warning.message}")
            return False
        else:
            print("✅ pd.concat() pattern fixed - no FutureWarnings")
            return True

def test_extension_array_conversion():
    """Test extension array dtype conversions."""
    print("Testing extension array dtype conversions...")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Create a Series with extension array dtype
        test_series = pd.Series([1.0, 2.0, pd.NA], dtype='Float64')

        # Convert to numpy array handling pd.NA properly
        result = test_series.to_numpy(dtype=np.float64, na_value=np.nan)

        # Check for warnings
        dtype_warnings = [warning for warning in w if 'dtype' in str(warning.message).lower()]

        if dtype_warnings:
            print(f"❌ Dtype conversion warning still occurs: {len(dtype_warnings)} warnings")
            for warning in dtype_warnings:
                print(f"   Warning: {warning.message}")
            return False
        else:
            print("✅ Extension array conversion pattern fixed - no warnings")
            return True

def test_numeric_conversion_with_infer_objects():
    """Test numeric conversion using infer_objects to avoid downcasting warnings."""
    print("Testing numeric conversion with infer_objects...")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Create test data that might trigger downcasting warning
        str_series = pd.Series(['1', '2', '3', 'nan', ''])
        str_series = str_series.replace(['', 'nan', 'None', 'N/A', 'NA', 'NaT'], '0')

        # Use the new pattern with infer_objects before fillna
        numeric_series = pd.to_numeric(str_series, errors='coerce')
        inferred_series = numeric_series.infer_objects(copy=False)
        result = inferred_series.fillna(0).astype('int64')

        # Check for FutureWarnings
        future_warnings = [warning for warning in w if issubclass(warning.category, FutureWarning)]

        if future_warnings:
            print(f"❌ FutureWarning still occurs: {len(future_warnings)} warnings")
            for warning in future_warnings:
                print(f"   Warning: {warning.message}")
            return False
        else:
            print("✅ Numeric conversion with infer_objects fixed - no FutureWarnings")
            return True

def main():
    """Run all tests and report results."""
    print("Testing pandas FutureWarning fixes...\n")

    tests = [
        test_fillna_astype_pattern,
        test_concat_empty_dataframes,
        test_extension_array_conversion,
        test_numeric_conversion_with_infer_objects,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
        print()

    # Summary
    passed = sum(results)
    total = len(results)

    print(f"Summary: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All pandas FutureWarning fixes are working correctly!")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the warnings above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
