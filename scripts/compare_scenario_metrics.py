#!/usr/bin/env python3
"""
Compare headcount & total‐contributions metrics between two or more scenario CSVs.
Usage:
  python scripts/compare_scenario_metrics.py --metrics_dir plan_outputs --scenarios Baseline ReEnroll_Below_Cap
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def load_metrics(metrics_dir: Path, scenario: str, hc_col: str, contrib_col: str):
    path = metrics_dir / f"{scenario}_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found for scenario '{scenario}': {path}")
    df = pd.read_csv(path, usecols=["year", hc_col, contrib_col])
    return df.rename(
        columns={hc_col: f"hc_{scenario}", contrib_col: f"contrib_{scenario}"}
    )

def main(metrics_dir: Path, scenarios: list[str], 
         hc_col: str = "headcount", contrib_col: str = "total_contributions", 
         tol: float = 1e-6):
    # Load and rename each scenario’s metrics
    dfs = [load_metrics(metrics_dir, s, hc_col, contrib_col) for s in scenarios]
    # Merge all on year
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="year", how="outer")

    # Compare
    base_hc = merged[f"hc_{scenarios[0]}"]
    base_contrib = merged[f"contrib_{scenarios[0]}"]
    merged["hc_equal"] = True
    merged["contrib_equal"] = True

    for s in scenarios[1:]:
        merged["hc_equal"] &= merged[f"hc_{s}"].fillna(-999) == base_hc.fillna(-999)
        merged["contrib_equal"] &= np.isclose(
            merged[f"contrib_{s}"].fillna(np.nan), 
            base_contrib.fillna(np.nan),
            atol=tol,
            equal_nan=True
        )

    print(merged.to_string(index=False))
    if merged["hc_equal"].all() and merged["contrib_equal"].all():
        print("\n✅ All scenarios match exactly (within tol={}).".format(tol))
    else:
        print("\n❌ Discrepancies found:")
        print(merged.loc[~(merged["hc_equal"] & merged["contrib_equal"])])

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Compare scenario metrics for headcount and contributions"
    )
    p.add_argument("--metrics_dir", "-d", type=Path, required=True,
                   help="Directory containing `<scenario>_metrics.csv` files")
    p.add_argument("--scenarios", "-s", nargs="+", required=True,
                   help="List of scenario names to compare")
    p.add_argument("--hc_col", default="headcount",
                   help="Column name for headcount (default: headcount)")
    p.add_argument("--contrib_col", default="total_contributions",
                   help="Column name for contributions (default: total_contributions)")
    p.add_argument("--tol", type=float, default=1e-6,
                   help="Tolerance when comparing floating‐point contributions")
    args = p.parse_args()
    try:
        main(args.metrics_dir, args.scenarios, args.hc_col, args.contrib_col, args.tol)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        exit(1)