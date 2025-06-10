#!/usr/bin/env python3
"""
Analyze census data to understand why compensation is declining.
"""

from pathlib import Path

import pandas as pd


def analyze_census():
    """Analyze the census template to understand compensation distribution."""

    census_path = Path("data/census_template.parquet")
    df = pd.read_parquet(census_path)

    print("=== CENSUS ANALYSIS ===")
    print(f"Total employees: {len(df)}")
    print(f"Average compensation: ${df['employee_gross_compensation'].mean():,.0f}")
    print(
        f"Compensation range: ${df['employee_gross_compensation'].min():,.0f} - ${df['employee_gross_compensation'].max():,.0f}"
    )
    print()

    print("BY JOB LEVEL:")
    level_stats = (
        df.groupby("employee_level")
        .agg({"employee_gross_compensation": ["mean", "count"], "employee_tenure": "mean"})
        .round(1)
    )
    print(level_stats)
    print()

    # Compare with job level base salaries from config
    print("=== JOB LEVEL BASE SALARIES FROM CONFIG ===")
    base_salaries = {
        0: 40000,  # Hourly
        1: 65000,  # Staff
        2: 95000,  # Manager
        3: 140000,  # SrMgr
        4: 250000,  # Director
    }

    print("Level | Config Base | Census Avg | Difference")
    print("------|-------------|------------|------------")
    for level in sorted(df["employee_level"].unique()):
        if level in base_salaries:
            config_base = base_salaries[level]
            census_avg = df[df["employee_level"] == level]["employee_gross_compensation"].mean()
            diff_pct = (census_avg - config_base) / config_base * 100
            print(f"{level:5d} | ${config_base:10,} | ${census_avg:10,.0f} | {diff_pct:+6.1f}%")

    print()
    print("=== KEY INSIGHTS ===")
    current_avg = df["employee_gross_compensation"].mean()

    # What would new hires at base salaries do to the average?
    level_dist = df["employee_level"].value_counts(normalize=True).sort_index()
    expected_new_hire_avg = sum(
        base_salaries[level] * pct for level, pct in level_dist.items() if level in base_salaries
    )

    print(f"Current workforce average: ${current_avg:,.0f}")
    print(f"Expected new hire average (at base): ${expected_new_hire_avg:,.0f}")
    print(
        f"New hire disadvantage: {(expected_new_hire_avg - current_avg) / current_avg * 100:.1f}%"
    )

    if expected_new_hire_avg < current_avg:
        print(f"\n🔍 ROOT CAUSE IDENTIFIED:")
        print(f"New hires start at config base salaries (~${expected_new_hire_avg:,.0f})")
        print(f"But existing employees earn much more (~${current_avg:,.0f})")
        print(
            f"This creates structural pay deflation as low-paid new hires replace higher-paid departing employees"
        )


if __name__ == "__main__":
    analyze_census()
