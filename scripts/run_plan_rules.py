# scripts/run_plan_rules.py

"""
Script to apply plan rules for each scenario and year.
"""

#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import pandas as pd
import yaml
from utils.projection_utils import apply_plan_rules

def main(config_path: str, snapshots_dir: str, output_dir: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # extract global parameters and plan_rules
    global_params = cfg.get("global_parameters", {})
    global_pr = global_params.get("plan_rules", {})
    # ensure output dir
    os.makedirs(output_dir, exist_ok=True)
    for key, sc in cfg.get('scenarios', {}).items():
        # 1) Start from global defaults (including full plan_rules)
        scen_cfg = dict(global_params)

        # 2) Override top-level fields (excluding plan_rules)
        for k, v in sc.items():
            if k != 'plan_rules':
                scen_cfg[k] = v

        # 3) Merge scenario plan_rules atop global ones
        merged_pr = dict(global_pr)
        merged_pr.update(sc.get('plan_rules', {}))
        scen_cfg['plan_rules'] = merged_pr
        name = sc.get('scenario_name') or sc.get('name') or key
        print(f"Applying plan rules for scenario: {name}")
        metrics = []
        # read every base_run_year*.parquet from the flat snapshots_dir
        for fname in sorted(os.listdir(snapshots_dir)):
            if not fname.startswith("base_run_year") or not fname.endswith(".parquet"):
                continue
            year_num = int(fname[len("base_run_year"): -len(".parquet")])
            df = pd.read_parquet(os.path.join(snapshots_dir, fname))
            out_df = apply_plan_rules(df, scen_cfg, year_num)
            # save detailed output per year
            out_path = os.path.join(output_dir, f"{name}_year{year_num}.parquet")
            out_df.to_parquet(out_path)
            print(f"Wrote {out_path}")
            # compute summary metrics
            # use original snapshot metrics to avoid FP drift
            hc = len(df)
            comp = df['gross_compensation'].sum()
            summary = {
                'scenario': name,
                'year': year_num,
                'headcount': hc,
                'total_comp': comp,
                # add more as needed
            }
            metrics.append(summary)
        # write metrics table
        metrics_df = pd.DataFrame(metrics)
        metrics_path = os.path.join(output_dir, f"{name}_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Wrote metrics {metrics_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase II: apply plan rules')
    parser.add_argument('--config', required=True, help='YAML config file path')
    parser.add_argument('--snapshots', required=True, help='Directory of HR snapshots')
    parser.add_argument('--outdir', required=True, help='Output directory for plan outputs')
    args = parser.parse_args()
    main(args.config, args.snapshots, args.outdir)
