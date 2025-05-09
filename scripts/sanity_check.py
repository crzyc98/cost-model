"""
Sanity check headcounts and total compensation between HR snapshots (Phase I) and plan rules metrics (Phase II).
Usage:
  python scripts/sanity_check.py --snapshots snapshots --plan_outputs plan_outputs
"""

import os
from pathlib import Path
import pandas as pd
import argparse


def load_hr_metrics(snapshots_dir):
    records = []
    for fname in sorted(os.listdir(snapshots_dir)):
        if not fname.endswith(".parquet"):
            continue
        year = int(fname.replace("base_run_year", "").replace(".parquet", ""))
        df = pd.read_parquet(os.path.join(snapshots_dir, fname))
        # sum employee gross comp; fallback to gross_compensation
        comp_field = (
            "employee_gross_compensation"
            if "employee_gross_compensation" in df.columns
            else "gross_compensation"
        )
        records.append(
            {
                "scenario": Path(snapshots_dir).name or "baseline",
                "year": year,
                "hc_hr": len(df),
                "comp_hr": df[comp_field].sum(),
            }
        )
    return pd.DataFrame(records)


def load_plan_metrics(plan_outputs_dir):
    dfs = []
    for fname in os.listdir(plan_outputs_dir):
        if fname.endswith("_metrics.csv"):
            df = pd.read_csv(os.path.join(plan_outputs_dir, fname))
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def main(snapshots_dir, plan_outputs_dir):
    hr_df = load_hr_metrics(snapshots_dir)
    plan_df = load_plan_metrics(plan_outputs_dir)
    merged = pd.merge(hr_df, plan_df, on=["scenario", "year"], how="outer")
    merged["hc_match"] = merged["hc_hr"] == merged["headcount"]
    merged["comp_match"] = merged["comp_hr"].round(2) == merged[
        "total_plan_year_compensation"
    ].round(2)
    ok = merged["hc_match"] & merged["comp_match"]
    if ok.all():
        print("Sanity check passed: all headcounts and compensation match.")
    else:
        print("Discrepancies found:")
        print(
            merged.loc[
                ~ok,
                [
                    "scenario",
                    "year",
                    "hc_hr",
                    "headcount",
                    "comp_hr",
                    "total_plan_year_compensation",
                ],
            ]
        )


doc = __doc__
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument(
        "--snapshots", required=True, help="Phase I snapshots directory"
    )
    parser.add_argument(
        "--plan_outputs", required=True, help="Phase II outputs directory"
    )
    args = parser.parse_args()
    main(args.snapshots, args.plan_outputs)
