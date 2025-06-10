#!/usr/bin/env python3
import sys

import pandas as pd


def main(path):
    df = pd.read_parquet(path)
    print(f"Loaded '{path}'")
    print(f"→ Total rows: {len(df)}")
    print()

    # 1) Terminations during calendar year 2025
    term_2025 = df[df["employee_termination_date"].dt.year == 2025]
    print(f"Terminations in 2025: {len(term_2025)}")

    # 2) New hires after 2025 (i.e. hires late in Year 1)
    late_hires = df[df["employee_hire_date"].dt.year > 2025]
    print(f"Hires with hire_date > 2025 (late-year hires): {len(late_hires)}")
    print()

    # 3) Some summary stats
    print("Headcount breakdown by status:")
    print(df["status"].value_counts(dropna=False))
    print()

    print("Headcount breakdown by participation:")
    print(df["is_participating"].value_counts(dropna=False))
    print()

    # 4) Peek at a few incumbents and a few late hires
    print("=== First 5 incumbents (hire_date ≤ 2025) ===")
    print(
        df[df["employee_hire_date"].dt.year <= 2025].head()[
            [
                "employee_ssn",
                "employee_hire_date",
                "employee_termination_date",
                "status",
                "is_eligible",
                "is_participating",
            ]
        ]
    )
    print()

    print("=== First 5 late-year hires (hire_date > 2025) ===")
    print(
        late_hires.head()[
            [
                "employee_ssn",
                "employee_hire_date",
                "employee_termination_date",
                "status",
                "is_eligible",
                "is_participating",
            ]
        ]
    )
    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_year1_output.py <path/to/Baseline_year1.parquet>")
        sys.exit(1)
    main(sys.argv[1])
