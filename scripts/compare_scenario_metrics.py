"""
Compare metrics between two or more scenario metrics CSVs for headcount and total_comp equality.
Usage:
  python scripts/compare_scenario_metrics.py --metrics_dir plan_outputs --scenarios Baseline ReEnroll_Below_Cap
"""
import os
import argparse
import pandas as pd

def main(metrics_dir, scenarios):
    dfs = {}
    for scen in scenarios:
        path = os.path.join(metrics_dir, f"{scen}_metrics.csv")
        if not os.path.exists(path):
            print(f"Missing metrics file for scenario: {scen}")
            return
        dfs[scen] = pd.read_csv(path)
    # merge on year
    merged = dfs[scenarios[0]][['year','headcount','total_comp']].rename(
        columns={'headcount':f'hc_{scenarios[0]}','total_comp':f'comp_{scenarios[0]}'}
    )
    for scen in scenarios[1:]:
        df = dfs[scen][['year','headcount','total_comp']].rename(
            columns={'headcount':f'hc_{scen}','total_comp':f'comp_{scen}'}
        )
        merged = merged.merge(df, on='year')
    # compare
    merged['hc_equal'] = True
    merged['comp_equal'] = True
    base_hc = merged[f'hc_{scenarios[0]}']
    base_comp = merged[f'comp_{scenarios[0]}']
    for scen in scenarios[1:]:
        merged['hc_equal'] &= (merged[f'hc_{scen}'] == base_hc)
        merged['comp_equal'] &= (merged[f'comp_{scen}'].round(8) == base_comp.round(8))
    print(merged)
    if merged['hc_equal'].all() and merged['comp_equal'].all():
        print('All scenarios match exactly.')
    else:
        print('Discrepancies found between scenarios.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics_dir', required=True)
    parser.add_argument('--scenarios', nargs='+', required=True)
    args = parser.parse_args()
    main(args.metrics_dir, args.scenarios)
