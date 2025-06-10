#!/usr/bin/env python3
"""
Analyze the starting age distribution from the census template file.
This script calculates the baseline age distribution that should be preserved.
"""

# Import age band utilities
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(".")
from cost_model.state.age import AGE_BAND_CATEGORICAL_DTYPE, assign_age_band


def calculate_age_from_birth_date(birth_date, as_of_date=None):
    """Calculate age from birth date."""
    if as_of_date is None:
        as_of_date = pd.Timestamp("2025-01-01")  # Default simulation start

    birth_date = pd.to_datetime(birth_date)
    age = (as_of_date - birth_date).dt.days / 365.25
    return age.round(1)


def analyze_starting_census():
    """Analyze the starting census age distribution."""

    # Load the census template
    census_path = Path("data/census_template.parquet")
    if not census_path.exists():
        print(f"Error: Census file not found at {census_path}")
        return None

    print(f"Loading census from: {census_path}")
    df = pd.read_parquet(census_path)

    print(f"Census contains {len(df)} employees")
    print(f"Columns: {list(df.columns)}")

    # Calculate ages as of simulation start (2025-01-01)
    simulation_start = pd.Timestamp("2025-01-01")

    if "employee_birth_date" in df.columns:
        print("\nCalculating ages from birth dates...")
        df["calculated_age"] = calculate_age_from_birth_date(
            df["employee_birth_date"], simulation_start
        )
        df["calculated_age_band"] = (
            df["calculated_age"].map(assign_age_band).astype(AGE_BAND_CATEGORICAL_DTYPE)
        )

        print(f"Age range: {df['calculated_age'].min():.1f} to {df['calculated_age'].max():.1f}")

        # Calculate age distribution
        age_dist = df["calculated_age_band"].value_counts(normalize=True).sort_index()
        age_dist_dict = {str(band): float(count) for band, count in age_dist.items()}

        print("\n" + "=" * 60)
        print("STARTING CENSUS AGE DISTRIBUTION (BASELINE TARGET)")
        print("=" * 60)

        for band, proportion in age_dist_dict.items():
            count = int(proportion * len(df))
            print(f"{band:>8}: {proportion:6.3f} ({count:3d} employees)")

        print(f"\nTotal: {sum(age_dist_dict.values()):.3f} ({len(df)} employees)")

        # Also show raw age statistics
        print("\n" + "=" * 40)
        print("RAW AGE STATISTICS")
        print("=" * 40)
        print(f"Mean age: {df['calculated_age'].mean():.1f}")
        print(f"Median age: {df['calculated_age'].median():.1f}")
        print(f"Std dev: {df['calculated_age'].std():.1f}")

        return age_dist_dict

    elif "AgeAtYearStart" in df.columns:
        print("\nUsing existing AgeAtYearStart column...")
        df["age_band"] = (
            df["AgeAtYearStart"].map(assign_age_band).astype(AGE_BAND_CATEGORICAL_DTYPE)
        )

        age_dist = df["age_band"].value_counts(normalize=True).sort_index()
        age_dist_dict = {str(band): float(count) for band, count in age_dist.items()}

        print("\n" + "=" * 60)
        print("STARTING CENSUS AGE DISTRIBUTION (BASELINE TARGET)")
        print("=" * 60)

        for band, proportion in age_dist_dict.items():
            count = int(proportion * len(df))
            print(f"{band:>8}: {proportion:6.3f} ({count:3d} employees)")

        print(f"\nTotal: {sum(age_dist_dict.values()):.3f} ({len(df)} employees)")

        return age_dist_dict

    else:
        print("Error: No age or birth date columns found in census")
        return None


if __name__ == "__main__":
    baseline_age_dist = analyze_starting_census()

    if baseline_age_dist:
        print("\n" + "=" * 60)
        print("SUMMARY FOR TUNING SYSTEM")
        print("=" * 60)
        print("This age distribution should be used as the baseline_age target")
        print("in the load_baseline_distributions() function.")
        print("\nPython dict format:")
        print("baseline_age = {")
        for band, prop in baseline_age_dist.items():
            print(f'    "{band}": {prop:.3f},')
        print("}")
