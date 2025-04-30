# tests/debug_elig.py
import sys, os
from pathlib import Path

# ensure project root (one up from tests/) is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from utils.rules.eligibility import apply as apply_eligibility
from utils.rules.validators import EligibilityRule

# 1) Load your base‐year snapshot
df = pd.read_parquet("output/hr_snapshots/base_run_year1.parquet")

# 2) Examine the first few rows
print("Columns:", df.columns.tolist())
print(df[['employee_hire_date','employee_birth_date','tenure']].head())

# 3) Build the rule from your config
elig_cfg = dict(min_age=21, min_service_months=0)
elig_rule = EligibilityRule(**elig_cfg)

# 4) Manually compute the masks
year_end = pd.Timestamp("2025-12-31")
df = df.copy()
# same logic as EligibilityRule internally:
ages = (year_end - pd.to_datetime(df['employee_birth_date'])).dt.days // 365
service_months = df['tenure'] * 12  # if tenure is in years
mask_age = ages >= elig_cfg['min_age']
mask_service = service_months >= elig_cfg['min_service_months']
print(f"Total rows: {len(df)}")
print(f"  meets age ≥21:      {mask_age.sum()}")
print(f"  meets service ≥0:   {mask_service.sum()}")

# 5) Run the real apply and compare
out = apply_eligibility(df, elig_rule, year_end)
print("After apply_eligibility, is_eligible.sum() =", int(out['is_eligible'].sum()))