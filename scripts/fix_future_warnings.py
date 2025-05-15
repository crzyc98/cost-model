#!/usr/bin/env python3
"""
Script to identify and fix FutureWarnings related to DataFrame concatenation with empty entries.

According to the memory, these warnings occur in cost_model/engines/run_one_year.py (lines 65, 119, 147).
"""

import os
import re
import sys
from pathlib import Path
import warnings

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def find_concat_with_empty_df(file_path):
    """Find instances of pd.concat that might trigger FutureWarnings."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Look for pd.concat calls
    concat_pattern = r'pd\.concat\s*\(\s*\[([^]]+)\]'
    matches = re.findall(concat_pattern, content)
    
    issues = []
    for match in matches:
        # Check if the concat might include empty DataFrames
        if 'empty' in match.lower() or 'if ' in match or 'for ' in match:
            issues.append(match.strip())
    
    return issues

def suggest_fix(concat_code):
    """Suggest a fix for the concat issue."""
    # Basic suggestion is to filter out empty DataFrames before concatenation
    return f"""
# Original code:
{concat_code}

# Suggested fix:
# Filter out empty DataFrames before concatenation
dfs_to_concat = [df for df in [{concat_code}] if df is not None and not df.empty]
if dfs_to_concat:
    result = pd.concat(dfs_to_concat)
else:
    # Create an empty DataFrame with the expected columns
    result = pd.DataFrame()  # Add appropriate columns if needed
"""

def main():
    """Main function to identify and suggest fixes for FutureWarnings."""
    print("Checking for potential FutureWarnings in DataFrame concatenation...")
    
    # Check run_one_year.py specifically
    run_one_year_path = PROJECT_ROOT / "cost_model" / "engines" / "run_one_year_engine.py"
    if run_one_year_path.exists():
        print(f"\nChecking {run_one_year_path.relative_to(PROJECT_ROOT)}...")
        issues = find_concat_with_empty_df(run_one_year_path)
        
        if issues:
            print(f"Found {len(issues)} potential issues:")
            for i, issue in enumerate(issues, 1):
                print(f"\nIssue {i}:")
                print(f"  {issue}")
                print("\nSuggested fix:")
                print(suggest_fix(issue))
        else:
            print("No potential issues found.")
    else:
        print(f"File not found: {run_one_year_path.relative_to(PROJECT_ROOT)}")
    
    # Print general advice
    print("\nGeneral advice for fixing FutureWarnings with DataFrame concatenation:")
    print("1. Always filter out empty DataFrames before concatenation")
    print("2. Use a consistent approach for handling empty DataFrames")
    print("3. Consider using pandas.concat with ignore_index=True for simpler concatenation")
    print("4. When concatenating DataFrames with different columns, consider using join='outer'")

if __name__ == "__main__":
    main()
