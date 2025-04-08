"""
Function to create dummy census data for testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_dummy_census_files(num_years=3, base_year=2022, n_employees=10000):
    """Creates dummy historical CSV files for demonstration."""
    print(f"\n--- Creating {num_years} Dummy Census Files ---")
    base_date = datetime(base_year, 12, 31)
    all_ssns = set()

    for i in range(num_years):
        year = base_year + i
        plan_end_date = datetime(year, 12, 31)
        print(f"Generating data for year ending {plan_end_date.date()}...")

        data = {
            'ssn': [],
            'birth_date': [],
            'hire_date': [],
            'termination_date': [],
            'gross_compensation': []
            # Add other columns as needed
        }

        # Simulate employee population for the year
        current_year_employees = n_employees + int(n_employees * 0.02 * i) # Slight growth

        for j in range(current_year_employees):
            # Generate SSN
            ssn = f"DUMMY_{year}_{j:04d}"
            all_ssns.add(ssn)
            data['ssn'].append(ssn)

            # Simulate Birth Date (ages 20-65)
            birth_year = plan_end_date.year - np.random.randint(22, 65)
            birth_month = np.random.randint(1, 13)
            birth_day = np.random.randint(1, 29) # Keep it simple
            data['birth_date'].append(f"{birth_year}-{birth_month:02d}-{birth_day:02d}")

            # Simulate Hire Date (up to plan_end_date)
            hire_days_ago = np.random.randint(1, 15 * 365) # Hired in the last 15 years
            hire_date = plan_end_date - timedelta(days=hire_days_ago)
            data['hire_date'].append(hire_date.strftime('%Y-%m-%d'))

            # Simulate Compensation
            comp = np.random.lognormal(mean=np.log(60000), sigma=0.4) # Log-normal distribution
            data['gross_compensation'].append(round(comp, 2))

            # Simulate Termination Date (some turnover)
            term_date = pd.NaT
            if np.random.rand() < 0.05: # 5% chance of being terminated *in this year*
                # Terminate sometime between start of year and end of year
                 term_days_offset = np.random.randint(1, 365)
                 term_date = datetime(year, 1, 1) + timedelta(days=term_days_offset)
                 # Ensure term_date is not after plan_end_date (edge case for Dec 31st)
                 term_date = min(term_date, plan_end_date)
                 data['termination_date'].append(term_date.strftime('%Y-%m-%d'))
            else:
                 data['termination_date'].append(pd.NaT)


        df = pd.DataFrame(data)
        # Ensure correct types after creation
        df['birth_date'] = pd.to_datetime(df['birth_date'])
        df['hire_date'] = pd.to_datetime(df['hire_date'])
        df['termination_date'] = pd.to_datetime(df['termination_date'], errors='coerce')
        df['gross_compensation'] = pd.to_numeric(df['gross_compensation'])
        df['ssn'] = df['ssn'].astype(str)

        # Remove employees terminated *before* the start of the plan year for realism
        year_start_date = datetime(year, 1, 1)
        df = df[df['termination_date'].isna() | (df['termination_date'] >= year_start_date)]

        filename = f"dummy_census_{year}.csv"
        filepath = os.path.join(os.getcwd(), filename) # Save in current dir
        df.to_csv(filepath, index=False)
        print(f"Saved: {filename} ({len(df)} rows)")

    print("--- Dummy file creation complete ---")
    # Return the list of created filenames, sorted
    created_files = [os.path.join(os.getcwd(), f"dummy_census_{base_year + i}.csv") for i in range(num_years)]
    return sorted(created_files)
