#!/usr/bin/env python3
import sys
import argparse
import yaml
import pandas as pd
import numpy as np

def check_census(year, df, cfg):
    # 1. Age bounds
    min_age, max_age = cfg['global_parameters'].get('min_age', 18), cfg['global_parameters'].get('max_age', 70)
    ages = df['employee_birth_date'].apply(lambda bd: pd.to_datetime(bd))
    ages = ages.apply(lambda bd: year - bd.year - ((12,31) < (bd.month, bd.day)))
    assert ages.between(min_age, max_age).all(), f"❌ Some ages in {year} are outside [{min_age},{max_age}]"

    # 2. Headcount nonzero
    hc = len(df)
    assert hc > 0, f"❌ Year {year} census is empty!"
    # 3. Compensation positive & within 10× of median
    gc = df['employee_gross_compensation']
    med = gc.median()
    assert (gc > 0).all(), f"❌ Non‐positive comp in {year}"
    assert (gc < med * 10).all(), f"❌ Some comps in {year} exceed 10× median"

    print(f"✅ Census {year}: {hc} rows, ages OK, comp OK")

def check_plan_summary(path, cfg):
    df = pd.read_csv(path)
    # 1. Participation rates between 0–1
    for col in ('participation_rate_eligible','participation_rate_total'):
        assert df[col].between(0,1).all(), f"❌ {col} out of bounds in {path}"
    # 2. Headcount growth within ±50% per year
    hc = df['headcount'].values
    rel = np.diff(hc) / hc[:-1]
    assert np.all(np.abs(rel) < 0.5), f"❌ Year-to-year headcount change >50%: {rel}"
    # 3. Employer cost as % of comp between 0–1
    for pct in ('employer_cost_pct_plan_comp','employer_cost_pct_capped_comp'):
        assert df[pct].between(0,1).all(), f"❌ {pct} out of bounds in {path}"

    print(f"✅ Plan summary {path} OK")

def main(config, census_dir, summary_csv):
    with open(config) as f:
        cfg = yaml.safe_load(f)

    # load just year 1 census
    df1 = pd.read_parquet(f"{census_dir}/base_run_year1.parquet")
    start_year = cfg['global_parameters']['start_year']
    check_census(start_year, df1, cfg)

    # run the plan summary checks
    check_plan_summary(summary_csv, cfg)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config",    required=True)
    p.add_argument("--snapshots", required=True)
    p.add_argument("--summary",   required=True)
    args = p.parse_args()
    try:
        main(args.config, args.snapshots, args.summary)
    except AssertionError as e:
        print(str(e))
        sys.exit(1)
    print("🎉 All fast-fail checks passed.")