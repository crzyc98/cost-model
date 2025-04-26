"""
Functions for loading and cleaning census data.
"""

import pandas as pd
import os
from datetime import datetime

def load_and_clean_census(filepath, expected_cols):
    """Loads a census file and performs basic cleaning."""
    try:
        df = pd.read_csv(filepath) # Add parameters as needed (sep, encoding)
        print(f"Loaded: {filepath} ({df.shape[0]} rows)")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

    # Basic column check
    if not all(col in df.columns for col in expected_cols['required']):
        print(f"Error: Missing required columns in {filepath}.")
        print(f"Expected: {expected_cols['required']}")
        print(f"Found: {list(df.columns)}")
        return None

    # Ensure data types
    date_cols = ['birth_date', 'hire_date', 'termination_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    numeric_cols = ['gross_compensation'] + expected_cols.get('contributions', [])
    for col in numeric_cols:
         if col in df.columns:
             df[col] = pd.to_numeric(df[col], errors='coerce')#.fillna(0) # Keep NaNs for now

    if 'ssn' in df.columns:
        df['ssn'] = df['ssn'].astype(str)

    # Infer plan year end date (assuming file name format or use a fixed logic)
    try:
        filename = os.path.basename(filepath)
        year_str = filename.split('_')[-1].split('.')[0] # Basic extraction
        year = int(year_str)
         # Ensure year is reasonable before creating date
        current_system_year = datetime.now().year
        if 1980 < year <= current_system_year + 1: # Plausibility check
             df['plan_year_end_date'] = pd.to_datetime(f"{year}-12-31")
        else:
             print(f"Warning: Could not reliably determine year from filename '{filename}'. Setting year end based on current time.")
             # Fallback logic if year extraction fails or seems wrong
             df['plan_year_end_date'] = pd.to_datetime(f'{current_system_year-1}-12-31') # Assume previous year end

    except Exception as e:
        print(f"Warning: Could not parse year from filename {filename}. Error: {e}. Setting year end based on current time.")
        # Fallback logic if parsing fails
        df['plan_year_end_date'] = pd.to_datetime(f'{datetime.now().year-1}-12-31') # Assume previous year end


    return df
