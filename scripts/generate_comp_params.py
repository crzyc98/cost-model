#!/usr/bin/env python3
"""
QuickStart: Generate Compensation Parameters

This script generates compensation parameters for new hires and role-based compensation
using historical census data. It outputs parameters that can be used in workforce projections.

Example Usage:

# 1. Programmatic Usage
```python
import pandas as pd

# 1. Load your census data (must contain employee_birth_date and employee_role columns)
df = pd.read_parquet("data/census_preprocessed.parquet")

# 2. Generate compensation parameters
comp_params = generate_compensation_params(df)

# 3. View the generated parameters
print("\nNew Hire Compensation Parameters:")
for key, value in comp_params["new_hire_compensation_params"].items():
    print(f"{key}: {value}")

print("\nRole-Based Compensation Parameters:")
for role, params in comp_params["role_compensation_params"].items():
    print(f"\nRole: {role}")
    for key, value in params.items():
        print(f"{key}: {value}")
```

# 2. Command Line Usage
```bash
# Run the script directly from the command line
python scripts/generate_comp_params.py

# The script will:
# 1. Read data from data/census_preprocessed.parquet
# 2. Generate compensation parameters
# 3. Save the parameters to config/compensation_params.yaml
```

The script generates:
1. New hire compensation parameters (median salary, age factor, etc.)
2. Role-specific compensation parameters (percentiles, age factors)

Required Input:
- Parquet file containing employee data with columns:
  - employee_birth_date
  - employee_role
  - employee_gross_compensation

Output:
- Dictionary containing:
  - new_hire_compensation_params: Parameters for new hires
  - role_compensation_params: Parameters per role

"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression

# 1) Load your census
CENSUS_PARQUET = Path("data/census_preprocessed.parquet")
df = pd.read_parquet(CENSUS_PARQUET)

# 2) Compute age as of Jan 1 of plan year
PLAN_YEAR = datetime.today().year - 1
ref = pd.Timestamp(f"{PLAN_YEAR}-01-01")
df["age"] = ((ref - df["employee_birth_date"]).dt.days / 365.25).astype(int)

out = {
    "new_hire_compensation_params": {},
    # role_compensation_params removed as part of schema refactoring
}

# 3) Overall “default” (all roles combined)
all_comp = df["employee_gross_compensation"]
med = all_comp.median()
p10 = np.percentile(all_comp, 10)
p90 = np.percentile(all_comp, 90)
log_dev = np.log(all_comp / med).std()

# Age factor: fit comp ~ age (linear)
X = df[["age"]]
y = df["employee_gross_compensation"]
model = LinearRegression().fit(X, y)
age_factor = model.coef_[0] / med  # normalize as % of base per year

out["new_hire_compensation_params"] = {
    "comp_base_salary": float(med),
    "comp_min_salary": float(p10),
    "comp_max_salary": float(p90),
    "comp_age_factor": float(age_factor),
    "comp_stochastic_std_dev": float(log_dev),
    "new_hire_age_mean": float(df["age"].mean().round(1)),
    "new_hire_age_std": float(df["age"].std().round(1)),
}

# 4) Role-based compensation parameters removed as part of schema refactoring
# All compensation is now based on employee_level instead of employee_role

# 5) Dump YAML
print(yaml.dump(out, sort_keys=False))
