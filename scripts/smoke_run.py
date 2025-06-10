#!/usr/bin/env python3
"""
A quick one-year smoke run against a tiny, fabricated census to ensure Phase I & II plumbing is intact.
"""
import logging
import sys
from pathlib import Path

import pandas as pd

# make imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_hr_snapshots import main as run_hr
from scripts.run_plan_rules import main as run_pr


def make_tiny_census(path: Path):
    # create a minimal census: 10 employees, random ages/tenures
    df = pd.DataFrame(
        {
            "ssn": [f"000-00-000{i}" for i in range(10)],
            "role": ["Staff"] * 10,
            "birth_date": pd.to_datetime("1980-01-01"),
            "hire_date": pd.to_datetime("2020-01-01"),
            "termination_date": [pd.NaT] * 10,
            "gross_compensation": [50000 + i * 1000 for i in range(10)],
            "plan_year_compensation": [50000 + i * 1000 for i in range(10)],
            "capped_compensation": [48000 + i * 1000 for i in range(10)],
            "employee_deferral_pct": [0.05] * 10,
            "employee_contribution_amt": [2500 + i * 100 for i in range(10)],
            "employer_match_contribution_amt": [1250 + i * 50 for i in range(10)],
            "employer_core_contribution_amt": [0] * 10,
        }
    )
    df.to_parquet(path)


def smoke():
    logging.basicConfig(level=logging.INFO)

    tmp_hr = Path("output/hr_snapshots-smoke")
    tmp_out = Path("output/plan_outputs-smoke")
    cfg = "configs/dev_local.yaml"

    # ensure clean
    for d in (tmp_hr, tmp_out):
        if d.exists():
            for f in d.iterdir():
                f.unlink()
        d.mkdir(parents=True, exist_ok=True)

    # generate tiny census input
    census = Path("data/tiny_census.parquet")
    make_tiny_census(census)

    # run Phase I
    run_hr(config_path=cfg, census=str(census), output=str(tmp_hr), seed=42)

    # run Phase II
    run_pr(config_path=cfg, snapshots_dir=str(tmp_hr), output_dir=str(tmp_out))

    print("✅ Smoke run completed without errors.")


if __name__ == "__main__":
    smoke()
