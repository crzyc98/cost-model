# scripts/run_hr_snapshots.py

#!/usr/bin/env python3
"""
Script to export a single, shared HR snapshot for each projection year.
This is Phase I of the two‑phase split: headcount, turnover, comp bumps & new hires
are all driven once (from the baseline scenario’s settings) and then reused
by each plan‑rules scenario.
"""

import os
import sys
import argparse
import pandas as pd
import yaml

# ensure we can import project_hr
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.projection_utils import project_hr

def main():
    parser = argparse.ArgumentParser(
        description="Phase I: run HR dynamics exactly once (baseline) and dump snapshots"
    )
    parser.add_argument(
        "--config", "-c", required=True,
        help="Path to YAML config (must include a 'baseline' scenario)"
    )
    parser.add_argument(
        "--census", "-d", required=True,
        help="Path to census CSV (with hire_date, termination_date, birth_date)"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output directory for base_run_year{n}.parquet files"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # Load global params + baseline overrides
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    global_params = cfg.get("global_parameters", {})
    baseline = cfg.get("scenarios", {}).get("baseline")
    if baseline is None:
        sys.exit("ERROR: 'baseline' scenario not found in config.")

    # Build the HR config from global_params + baseline (excluding plan_rules)
    hr_cfg = {**global_params}
    for k, v in baseline.items():
        if k not in ("plan_rules", "scenario_name", "name"):
            hr_cfg[k] = v

    # Load census data
    start_df = pd.read_csv(
        args.census,
        parse_dates=["hire_date", "termination_date", "birth_date"]
    )

    # Run the HR projection once
    os.makedirs(args.output, exist_ok=True)
    print("Running HR dynamics _once_ (baseline settings)…")
    hr_snapshots = project_hr(start_df, hr_cfg, random_seed=args.seed)

    # Write out one parquet per year
    for year, df in hr_snapshots.items():
        out_path = os.path.join(args.output, f"base_run_year{year}.parquet")
        df.to_parquet(out_path)
        print(f"Wrote HR snapshot: {out_path}")

if __name__ == "__main__":
    main()