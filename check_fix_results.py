#!/usr/bin/env python3
"""
Check if the summary statistics fix worked correctly.
"""

import pandas as pd
import sys

def main():
    try:
        # Load the summary statistics
        df = pd.read_parquet('test_output_fixed/test_fix_summary_statistics.parquet')
        
        print("=== Summary Statistics Results ===")
        print(df.to_string())
        print()
        
        print("=== Key Columns Check ===")
        key_columns = ['total_terminations', 'avg_compensation', 'total_compensation']
        
        all_good = True
        for col in key_columns:
            if col in df.columns:
                zero_count = (df[col] == 0).sum()
                total_rows = len(df)
                print(f"{col}:")
                print(f"  - {zero_count} zeros out of {total_rows} rows")
                if zero_count == 0:
                    print(f"  ✅ SUCCESS: {col} has no zero values")
                    print(f"  - Sample values: {df[col].tolist()}")
                else:
                    print(f"  ❌ ISSUE: {col} still has {zero_count} zero values")
                    all_good = False
            else:
                print(f"❌ MISSING: {col} column not found")
                all_good = False
            print()
        
        if all_good:
            print("🎉 SUCCESS: All key columns are properly populated with non-zero values!")
        else:
            print("❌ ISSUES REMAIN: Some columns still have zero values or are missing")
            
        return all_good
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
