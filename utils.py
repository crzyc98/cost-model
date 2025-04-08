"""
Utility functions for data calculations.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def calculate_age(birth_date, reference_date):
    """Calculates age as of a reference date."""
    if pd.isna(birth_date): return np.nan
    birth_date = pd.to_datetime(birth_date).tz_localize(None)
    reference_date = pd.to_datetime(reference_date).tz_localize(None)
    age = reference_date.year - birth_date.year - ((reference_date.month, reference_date.day) < (birth_date.month, birth_date.day))
    return age

def calculate_tenure(hire_date, reference_date, termination_date=None):
    """Calculates tenure in years as of a reference date."""
    if pd.isna(hire_date): return np.nan
    hire_date = pd.to_datetime(hire_date).tz_localize(None)
    reference_date = pd.to_datetime(reference_date).tz_localize(None)
    end_date = reference_date
    if pd.notna(termination_date):
        termination_date = pd.to_datetime(termination_date).tz_localize(None)
        end_date = min(termination_date, reference_date)
    if hire_date > end_date: return 0
    tenure = (end_date - hire_date).days / 365.25
    return tenure

def generate_new_ssn(existing_ssns, num_new):
    """Generates unique placeholder SSNs."""
    new_ssns = []
    max_existing_numeric = 0
    try:
        numeric_parts = pd.to_numeric(existing_ssns.astype(str).str.extract(r'(\d+)$', expand=False), errors='coerce')
        if numeric_parts.notna().any(): max_existing_numeric = int(numeric_parts.max())
        else: max_existing_numeric = 900000000 + len(existing_ssns)
    except Exception: max_existing_numeric = 900000000 + len(existing_ssns)

    current_max = max_existing_numeric
    temp_new_ssns = set()
    while len(new_ssns) < num_new:
        current_max += 1
        potential_ssn = f"NEW_{current_max}"
        if potential_ssn not in existing_ssns and potential_ssn not in temp_new_ssns:
             new_ssns.append(potential_ssn)
             temp_new_ssns.add(potential_ssn)
             if current_max > max_existing_numeric + num_new * 100: # Safety break
                  print("Warning: Potential issue generating unique SSNs. Using fallback.")
                  break
    needed = num_new - len(new_ssns)
    if needed > 0:
        print(f"Using fallback SSN generation for {needed} hires.")
        base_ts = pd.Timestamp.now(tz='America/New_York').value
        for i in range(needed):
           fallback_ssn = f"FALLBACK_{base_ts}_{np.random.randint(10000,99999)}_{i}"
           new_ssns.append(fallback_ssn)
    return new_ssns
