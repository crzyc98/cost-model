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

    # Parse dates
    for col in ['employee_hire_date', 'employee_termination_date', 'employee_birth_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Remove any duplicate rows
    df = df.drop_duplicates()

    # Remove any rows with missing employee_ssn
    if 'employee_ssn' in df.columns:
        df = df[df['employee_ssn'].notna()]

    # Ensure data types
    numeric_cols = ['gross_compensation'] + expected_cols.get('contributions', [])
    for col in numeric_cols:
         if col in df.columns:
             df[col] = pd.to_numeric(df[col], errors='coerce')#.fillna(0) # Keep NaNs for now

    if 'employee_ssn' in df.columns:
        df['employee_ssn'] = df['employee_ssn'].astype(str)

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

def assign_employment_status(df, start_year):
    """Assign employment_status and status columns using ABM logic and EmploymentStatus enum."""
    from utils.status_enums import EmploymentStatus
    from utils.constants import active_STATUS, unknown_STATUS
    import numpy as np
    df = df.copy()
    df['employee_hire_date'] = pd.to_datetime(df['employee_hire_date'], errors='coerce')
    df['employee_termination_date'] = pd.to_datetime(df['employee_termination_date'], errors='coerce')
    df['employee_birth_date'] = pd.to_datetime(df['employee_birth_date'], errors='coerce')

    status = np.full(len(df), EmploymentStatus.UNKNOWN.value, dtype=object)

    # Not hired yet
    status[df['employee_hire_date'].dt.year > start_year] = EmploymentStatus.NOT_HIRED.value

    # New hire this year
    status[df['employee_hire_date'].dt.year == start_year] = EmploymentStatus.NEW_HIRE.value

    # Previously terminated
    status[(df['employee_termination_date'].notna()) & (df['employee_termination_date'].dt.year < start_year)] = EmploymentStatus.PREV_TERMINATED.value

    # Active continuous: hired before start year, no termination
    status[(df['employee_hire_date'].dt.year < start_year) & (df['employee_termination_date'].isna())] = EmploymentStatus.ACTIVE_CONTINUOUS.value

    # Active initial: hired in start year, no termination
    status[(df['employee_hire_date'].dt.year == start_year) & (df['employee_termination_date'].isna())] = EmploymentStatus.ACTIVE_INITIAL.value

    df['employment_status'] = status
    df['status'] = np.where(
        df['employment_status'].isin([
            EmploymentStatus.ACTIVE_INITIAL.value,
            EmploymentStatus.ACTIVE_CONTINUOUS.value,
            EmploymentStatus.NEW_HIRE.value
        ]),
        active_STATUS,
        unknown_STATUS
    )
    return df
