#!/usr/bin/env python3
"""
Fix census file by removing employees with future hire dates.

This script identifies and removes employees from the census file who have
hire dates in the future relative to the simulation start year.
"""

import pandas as pd
import sys
from pathlib import Path

def fix_census_future_dates(census_path: str, start_year: int = 2024, backup: bool = True):
    """
    Remove employees with future hire dates from census file.
    
    Args:
        census_path: Path to census file
        start_year: Simulation start year (employees hired after this are removed)
        backup: Whether to create backup of original file
    """
    print(f"Loading census file: {census_path}")
    census = pd.read_parquet(census_path)
    
    print(f"Original census: {len(census)} employees")
    
    # Identify future hires
    hire_dates = pd.to_datetime(census['employee_hire_date'])
    cutoff_date = pd.Timestamp(f"{start_year}-12-31")
    future_hires = hire_dates > cutoff_date
    
    print(f"Found {future_hires.sum()} employees with hire dates after {cutoff_date}")
    
    if future_hires.any():
        print("Employees with future hire dates:")
        future_employees = census[future_hires][['employee_id', 'employee_hire_date']]
        print(future_employees.to_string())
        
        # Create backup if requested
        if backup:
            backup_path = census_path.replace('.parquet', '_backup.parquet')
            census.to_parquet(backup_path)
            print(f"Created backup: {backup_path}")
        
        # Remove future hires
        cleaned_census = census[~future_hires].copy()
        print(f"Cleaned census: {len(cleaned_census)} employees")
        
        # Save cleaned census
        cleaned_census.to_parquet(census_path)
        print(f"Saved cleaned census to: {census_path}")
        
        return len(census) - len(cleaned_census)
    else:
        print("No future hire dates found - no changes needed")
        return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_census_future_dates.py <census_path> [start_year]")
        sys.exit(1)
    
    census_path = sys.argv[1]
    start_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2024
    
    removed_count = fix_census_future_dates(census_path, start_year)
    print(f"\nSummary: Removed {removed_count} employees with future hire dates")
