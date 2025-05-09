#!/usr/bin/env python3
import argparse
import pandas as pd


def main():
    p = argparse.ArgumentParser(
        description="Quick inspection of a plan-rules Parquet output"
    )
    p.add_argument("parquet_file", help="Path to a .parquet file")
    args = p.parse_args()

    # 1) load
    df = pd.read_parquet(args.parquet_file)

    # 2) basics
    print(f"\nLoaded {args.parquet_file!r}")
    print(f"→ Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    # 3) dtypes
    print("Column dtypes:")
    print(df.dtypes.to_string())
    print()

    # 4) missing
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if not miss.empty:
        print("Missing values per column:")
        print(miss.to_string())
    else:
        print("No missing values!\n")

    # 5) head & tail
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print("\nLast 5 rows:")
    print(df.tail().to_string(index=False))


if __name__ == "__main__":
    main()
