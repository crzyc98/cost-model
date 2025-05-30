"""
QuickStart: Generate All Simulation Parameters

This script generates all necessary parameters for the workforce simulation from historical data,
including compensation parameters, attrition rates, new hire rates, and behavioral parameters.

Example Usage:

# 1. Command Line Usage
```bash
# Run the script directly from the command line
python scripts/generate_all_params.py

# The script will:
# 1. Read data from:
#    - data/census_preprocessed.parquet
#    - data/census_boy.parquet (if available)
# 2. Generate all simulation parameters
# 3. Save the parameters to config/simulation_params.yaml
```

# 2. Programmatic Usage
```python
from scripts.generate_all_params import generate_all_parameters

# Load your census data
boy_df = pd.read_parquet("data/census_boy.parquet")
eoy_df = pd.read_parquet("data/census_preprocessed.parquet")

# Generate all parameters
params = generate_all_parameters(eoy_df, boy_df)

# View generated parameters
print(yaml.dump(params, sort_keys=False))
```

The script generates:
1. Compensation parameters (new hires and role-based)
2. Attrition rates by tenure band
3. New hire rates and replacement premiums
4. Deferral rate distributions
5. Productivity curves for new hires

Required Input:
- Parquet files containing employee data with columns:
  - employee_birth_date
  - employee_role
  - employee_gross_compensation
  - employee_termination_date (for attrition)
  - employee_hire_date
  - employee_deferral_rate (for behavioral params)
  - performance_rating (for productivity curves)

Output:
- YAML configuration with all simulation parameters
"""

import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.linear_model import LinearRegression


def generate_compensation_params(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate compensation parameters from census data, filtering out low comp."""
    # Filter out employees earning less than $10k
    df = df[df["employee_gross_compensation"] >= 10000].copy()

    # Compute age as of Jan 1 of plan year
    PLAN_YEAR = datetime.today().year - 1
    ref = pd.Timestamp(f"{PLAN_YEAR}-01-01")
    df["age"] = ((ref - df["employee_birth_date"]).dt.days / 365.25).astype(int)
    df["tenure"] = ((ref - df["employee_hire_date"]).dt.days / 365.25)

    # Select employees with 1-3 years tenure for new hire parameters
    new_hires = df[(df["tenure"] >= 1) & (df["tenure"] < 4)].copy()

    # Calculate new hire compensation parameters
    all_comp = new_hires["employee_gross_compensation"]
    med = all_comp.median()
    p10 = np.percentile(all_comp, 10)
    p90 = np.percentile(all_comp, 90)
    log_dev = np.log(all_comp / med).std()

    model = LinearRegression().fit(new_hires[["age"]], all_comp)
    age_factor = model.coef_[0] / med  # normalize as % of base per year

    new_hire_params = {
        "comp_base_salary": float(med),
        "comp_min_salary": float(p10),
        "comp_max_salary": float(p90),
        "comp_age_factor": float(age_factor),
        "comp_stochastic_std_dev": float(log_dev),
        "new_hire_age_mean": float(new_hires["age"].mean().round(1)),
        "new_hire_age_std": float(new_hires["age"].std().round(1)),
    }

    # Role-based compensation parameters removed as part of schema refactoring
    # All compensation is now based on employee_level instead of employee_role

    return {
        "new_hire_compensation_params": new_hire_params,
        # role_compensation_params removed
    }


def generate_attrition_rates(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate attrition rates by tenure band."""
    # Compute tenure for terminated employees
    df_term = df.dropna(subset=["employee_termination_date"]).copy()
    df_term["tenure_at_term"] = (
        (pd.to_datetime(df_term["employee_termination_date"]) -
         pd.to_datetime(df_term["employee_hire_date"]))
        .dt.days / 365.25
    )

    # Define tenure bands
    bins = [0, 1, 3, 5, 100]
    labels = ["0-1", "1-3", "3-5", "5+"]

    # Bin terminated employees
    df_term["tenure_band"] = pd.cut(
        df_term["tenure_at_term"],
        bins=bins,
        labels=labels,
        right=False
    )

    # Compute tenure for active employees
    ref = pd.Timestamp(f"{datetime.today().year}-01-01")
    df["tenure"] = ((ref - df["employee_hire_date"]).dt.days / 365.25).astype(int)
    df["tenure_band"] = pd.cut(
        df["tenure"],
        bins=bins,
        labels=labels,
        right=False
    )

    # Calculate attrition rates
    headcounts = df.groupby("tenure_band", observed=True).size()
    terms = df_term.groupby("tenure_band", observed=True).size()
    attrition = (terms / headcounts).fillna(0).to_dict()

    return {
        "annual_termination_rate": float(df_term.shape[0] / df.shape[0]),
        "termination_rate_by_tenure": attrition
    }


def generate_new_hire_params(df_boy: pd.DataFrame, df_eoy: pd.DataFrame) -> Dict[str, Any]:
    """Generate new hire rate and replacement premium parameters."""
    # Calculate new hire rate
    new_hires_count = df_eoy[~df_eoy["employee_id"].isin(df_boy["employee_id"])].shape[0]
    eoy_headcount = df_eoy.shape[0]
    new_hire_rate = new_hires_count / eoy_headcount

    # Calculate replacement premium
    term_median = df_boy["employee_gross_compensation"].median()
    new_median = df_eoy["employee_gross_compensation"].median()
    replacement_premium = (new_median / term_median) - 1

    return {
        "new_hire_rate": float(new_hire_rate),
        "replacement_hire_premium": float(replacement_premium)
    }


def generate_behavioral_params(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate behavioral parameters from deferral rates."""
    def_rates = df["employee_deferral_rate"].dropna()
    return {
        "behavioral_params": {
            "voluntary_default_deferral": float(def_rates.mean()),
            "voluntary_window_days": 180
        }
    }


def generate_productivity_curve(df: pd.DataFrame) -> dict:
    """Generate productivity curve using employees with 1–3 years tenure and comp ≥ $10k."""
    # Filter out low comp and compute tenure
    df = df[df["employee_gross_compensation"] >= 10000].copy()
    ref = pd.Timestamp(f"{datetime.today().year}-01-01")
    df["tenure"] = ((ref - df["employee_hire_date"]).dt.days / 365.25)

    # Select employees with 1–3 years tenure
    bucket = df[(df["tenure"] >= 1) & (df["tenure"] < 3)].copy()

    # Calculate months since hire
    bucket["months_since_hire"] = (
        (ref - bucket["employee_hire_date"]).dt.days / 30
    ).astype(int)

    # Compute average productivity by month
    prod = bucket.groupby("months_since_hire")["performance_rating"].mean().to_dict()

    # Build first 6 months of the curve
    return {
        "onboarding": {
            "productivity_curve": [prod.get(m, 1.0) for m in range(6)]
        }
    }


def generate_all_parameters(
    eoy_df: pd.DataFrame,
    boy_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Generate all simulation parameters from census data."""
    # Initialize output structure
    out = {
        "scenarios": {
            "baseline": {
                "name": "Baseline",
                "description": "Default simulation parameters generated from historical data"
            }
        },
        "global_parameters": {
            "compensation": {},  # Will hold both new hire and role params
            "attrition": {},
            "new_hires": {},
            "onboarding": {}
        },
        "plan_rules": {
            "behavioral_params": {}
        }
    }

    # Generate compensation parameters and nest under global_parameters.compensation
    comp_params = generate_compensation_params(eoy_df)
    out["global_parameters"]["compensation"]["new_hire"] = comp_params["new_hire_compensation_params"]
    # roles section removed as part of schema refactoring

    # Generate attrition rates and nest under global_parameters.attrition
    out["global_parameters"]["attrition"] = generate_attrition_rates(eoy_df)

    # Generate new hire parameters if BOY data is available
    if boy_df is not None:
        out["global_parameters"]["new_hires"] = generate_new_hire_params(boy_df, eoy_df)

    # Add behavioral parameters if deferral rates are available
    if "employee_deferral_rate" in eoy_df.columns:
        out["plan_rules"]["behavioral_params"] = generate_behavioral_params(eoy_df)["behavioral_params"]

    # Add productivity curve if performance ratings are available
    if "performance_rating" in eoy_df.columns:
        out["global_parameters"]["onboarding"] = generate_productivity_curve(eoy_df)["onboarding"]

    return out


def main():
    """Main entry point for command line execution."""
    # Load data
    eoy_df = pd.read_parquet("data/census_preprocessed.parquet")
    boy_df = None

    # Try to load beginning of year data if available
    try:
        boy_df = pd.read_parquet("data/census_boy.parquet")
    except FileNotFoundError:
        print("Warning: BOY census data not found. Some parameters will be omitted.")

    # Generate all parameters
    params = generate_all_parameters(eoy_df, boy_df)

    # Save to YAML
    output_path = Path("config/simulation_params.yaml")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(params, f, sort_keys=False)

    print(f"Generated parameters saved to {output_path}")


if __name__ == "__main__":
    main()
