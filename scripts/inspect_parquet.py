#!/usr/bin/env python3
import sys

import pandas as pd


def main(path):
    # 1) Load the Parquet
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"❌ Failed to read {path!r}: {e}")
        sys.exit(1)

    # 2) Show basic info
    print(f"\nDataFrame shape: {df.shape}")
    print("\nColumn dtypes:")
    print(df.dtypes)

    # 3) Count nulls per column
    null_counts = df.isna().sum().sort_values(ascending=False)
    print("\nMissing values per column:")
    print(null_counts[null_counts > 0])

    # 4) Print first few rows for inspection
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_parquet.py /path/to/Baseline_year1.parquet")
        sys.exit(1)
    main(sys.argv[1])
