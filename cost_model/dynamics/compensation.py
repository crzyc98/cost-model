# cost_model/dynamics/compensation.py
"""
Functions related to compensation changes during simulation dynamics.

## QuickStart

To simulate compensation changes programmatically:

```python
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from cost_model.dynamics.compensation import apply_comp_bump, apply_onboarding_bump

# Create a random number generator with a fixed seed for reproducibility
rng = np.random.default_rng(seed=42)

# Create a sample workforce snapshot
snapshot = pd.DataFrame({
    'employee_id': ['EMP001', 'EMP002', 'EMP003', 'EMP004'],
    'employee_role': ['Engineer', 'Manager', 'Engineer', 'Analyst'],
    'employee_gross_compensation': [75000.0, 85000.0, 65000.0, 55000.0],
    'active': [True, True, True, True],
    'employee_hire_date': [
        pd.Timestamp('2023-06-01'),  # 2 years of service
        pd.Timestamp('2022-03-15'),  # 3 years of service
        pd.Timestamp('2024-01-10'),  # 1.5 years of service
        pd.Timestamp('2024-11-05'),  # 0.5 years of service
    ],
    'tenure': [2.0, 3.0, 1.5, 0.5]  # Tenure in years
}).set_index('employee_id')

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("compensation_simulation")

# Method 1: Apply a flat compensation increase to all employees
flat_rate = 0.03  # 3% annual increase

updated_snapshot_flat = apply_comp_bump(
    df=snapshot.copy(),
    comp_col='employee_gross_compensation',
    rate=flat_rate,
    rng=rng,
    log=logger
)

# Calculate the changes
initial_comp = snapshot['employee_gross_compensation']
updated_comp = updated_snapshot_flat['employee_gross_compensation']
comp_change = updated_comp - initial_comp
comp_pct_change = (comp_change / initial_comp) * 100

print("Flat rate compensation increases:")
for idx in snapshot.index:
    old = initial_comp[idx]
    new = updated_comp[idx]
    change = comp_pct_change[idx]
    print(f"  {idx}: ${old:,.2f} → ${new:,.2f} (+{change:.1f}%)")

print(f"  Average increase: ${comp_change.mean():,.2f} ({comp_pct_change.mean():.1f}%)")
print(f"  Total compensation change: ${comp_change.sum():,.2f}")

# Method 2: Apply a more complex compensation increase with distribution
comp_dist_config = {
    'method': 'normal',
    'mean': 0.03,  # 3% mean increase
    'std': 0.01,   # 1% standard deviation
    'min': 0.01,   # Minimum 1% increase
    'max': 0.06    # Maximum 6% increase
}

updated_snapshot_dist = apply_comp_bump(
    df=snapshot.copy(),
    comp_col='employee_gross_compensation',
    rate=0.03,  # Base rate (may be modified by distribution)
    rng=rng,
    log=logger,
    dist=comp_dist_config
)

# Calculate the changes with distribution
initial_comp = snapshot['employee_gross_compensation']
updated_comp_dist = updated_snapshot_dist['employee_gross_compensation']
comp_change_dist = updated_comp_dist - initial_comp
comp_pct_change_dist = (comp_change_dist / initial_comp) * 100

print("\nDistributed compensation increases:")
for idx in snapshot.index:
    old = initial_comp[idx]
    new = updated_comp_dist[idx]
    change = comp_pct_change_dist[idx]
    print(f"  {idx}: ${old:,.2f} → ${new:,.2f} (+{change:.1f}%)")

print(f"  Average increase: ${comp_change_dist.mean():,.2f} ({comp_pct_change_dist.mean():.1f}%)")
print(f"  Total compensation change: ${comp_change_dist.sum():,.2f}")

# Method 3: Apply onboarding bump for new hires
new_hires = pd.DataFrame({
    'employee_id': ['NH_001', 'NH_002', 'NH_003'],
    'employee_role': ['Engineer', 'Analyst', 'Engineer'],
    'employee_gross_compensation': [70000.0, 52000.0, 68000.0],
    'active': [True, True, True],
    'employee_hire_date': [
        pd.Timestamp('2025-02-15'),
        pd.Timestamp('2025-03-10'),
        pd.Timestamp('2025-05-01')
    ],
    'tenure': [0.0, 0.0, 0.0]  # New hires with zero tenure
}).set_index('employee_id')

# Get baseline salaries from previous year's hires (using our existing snapshot)
baseline_salaries = snapshot['employee_gross_compensation']

# Configure onboarding bump
onboarding_config = {
    'enabled': True,
    'method': 'percent',  # Apply percentage increase
    'rate': 0.05          # 5% onboarding bump
}

# Apply onboarding bump
updated_new_hires = apply_onboarding_bump(
    df=new_hires.copy(),
    comp_col='employee_gross_compensation',
    ob_cfg=onboarding_config,
    baseline_hire_salaries=baseline_salaries,
    rng=rng,
    log=logger
)

# Calculate the changes for new hires
initial_nh_comp = new_hires['employee_gross_compensation']
updated_nh_comp = updated_new_hires['employee_gross_compensation']
nh_comp_change = updated_nh_comp - initial_nh_comp
nh_comp_pct_change = (nh_comp_change / initial_nh_comp) * 100

print("\nOnboarding bumps for new hires:")
for idx in new_hires.index:
    old = initial_nh_comp[idx]
    new = updated_nh_comp[idx]
    change = nh_comp_pct_change[idx]
    print(f"  {idx}: ${old:,.2f} → ${new:,.2f} (+{change:.1f}%)")

print(f"  Average increase: ${nh_comp_change.mean():,.2f} ({nh_comp_pct_change.mean():.1f}%)")
print(f"  Total compensation change: ${nh_comp_change.sum():,.2f}")

# Save the results
output_dir = Path('output/compensation')
output_dir.mkdir(parents=True, exist_ok=True)

# Combine all employees
all_employees = pd.concat([updated_snapshot_dist, updated_new_hires])
all_employees.to_parquet(output_dir / 'snapshot_with_comp_changes.parquet')

# Create a summary of changes
summary = pd.DataFrame({
    'initial_compensation': pd.concat([initial_comp, initial_nh_comp]),
    'updated_compensation': pd.concat([updated_comp_dist, updated_nh_comp]),
    'absolute_change': pd.concat([comp_change_dist, nh_comp_change]),
    'percent_change': pd.concat([comp_pct_change_dist, nh_comp_pct_change])
})
summary.to_csv(output_dir / 'compensation_changes.csv')
```

This demonstrates how to apply different types of compensation changes, including flat increases, distributed increases, and onboarding bumps for new hires.
"""

import logging
import pandas as pd
import numpy as np
from typing import Mapping, Any, Optional

# Attempt to import column constants, provide fallbacks
try:
    from ..utils.columns import EMP_GROSS_COMP
except ImportError:
    print(
        "Warning (compensation.py): Could not import column constants from utils. Using string literals."
    )
    EMP_GROSS_COMP = "employee_gross_compensation"

# Import sampling helpers if needed (e.g., for second-year bumps)
try:
    from .sampling.salary import SalarySampler, DefaultSalarySampler
except ImportError:
    print(
        "Warning (compensation.py): Could not import SalarySampler. Complex bumps might fail."
    )

    class DefaultSalarySampler:
        pass  # Dummy class


logger = logging.getLogger(__name__)


def apply_comp_bump(
    df: pd.DataFrame,
    comp_col: str,
    rate: float,
    rng: np.random.Generator,
    log: logging.Logger,
    dist: Optional[Mapping[str, Any]] = None,  # Optional config for complex bumps
    sampler: Optional[SalarySampler] = None,  # Optional sampler instance
) -> pd.DataFrame:
    """
    Applies annual compensation increase.

    Handles flat rate increases for experienced employees and potentially
    more complex sampling for second-year employees if configured.

    Args:
        df: DataFrame with employee data (must include comp_col and 'tenure').
        comp_col: Name of the compensation column to update.
        rate: The base annual increase rate (e.g., 0.03 for 3%).
        rng: NumPy random number generator.
        log: Logger instance.
        dist: Optional configuration for second-year bump distribution.
        sampler: Optional SalarySampler instance for complex bumps.

    Returns:
        DataFrame with updated compensation.
    """
    df2 = df.copy()
    if comp_col not in df2:
        log.warning(f"Compensation column '{comp_col}' not found, skipping comp bump.")
        return df2
    if "tenure" not in df2.columns:
        log.warning(
            "Column 'tenure' not found, cannot apply tenure-based comp bump logic. Applying flat rate to all."
        )
        df2[comp_col] = pd.to_numeric(df2[comp_col], errors="coerce").fillna(0.0) * (
            1 + rate
        )
        return df2

    log.debug(
        f"Applying comp bump. Base Rate={rate:.2%}. Current avg comp: {df2[comp_col].mean():.0f}"
    )

    # Ensure columns are numeric
    df2[comp_col] = (
        pd.to_numeric(df2[comp_col], errors="coerce").fillna(0.0).astype(float)
    )
    df2["tenure"] = pd.to_numeric(df2["tenure"], errors="coerce").fillna(0.0)

    # Define masks based on tenure
    mask_new_or_first_year = df2["tenure"] < 1.0  # Tenure is float, check < 1
    mask_second_year = (df2["tenure"] >= 1.0) & (df2["tenure"] < 2.0)
    mask_experienced = df2["tenure"] >= 2.0

    n_new = mask_new_or_first_year.sum()
    n_second = mask_second_year.sum()
    n_exp = mask_experienced.sum()
    log.debug(
        f"Comp Bump Groups: New/First={n_new}, Second={n_second}, Experienced={n_exp}"
    )

    # 1) New hires / First year employees: Typically no bump applied in this step
    # (Their salary was set during hiring/onboarding)
    if n_new > 0:
        log.debug(f"Skipping standard comp bump for {n_new} new/first-year employees.")

    # 2) Second-year employees: Potentially use sampler
    if n_second > 0:
        # Use sampler if provided and configured
        if sampler and dist and hasattr(sampler, "sample_second_year"):
            log.debug(f"Applying sampler bump to {n_second} second-year employees.")
            try:
                df2.loc[mask_second_year, comp_col] = sampler.sample_second_year(
                    df2.loc[mask_second_year],
                    comp_col=comp_col,
                    dist=dist,
                    rate=rate,
                    rng=rng,
                )
            except Exception:
                log.exception(
                    "Error during sampler.sample_second_year. Applying flat rate instead."
                )
                df2.loc[mask_second_year, comp_col] *= 1 + rate
        else:
            # Fallback to flat rate if no sampler/config
            log.debug(
                f"Applying flat rate bump ({rate:.1%}) to {n_second} second-year employees (no sampler/config)."
            )
            df2.loc[mask_second_year, comp_col] *= 1 + rate

    # 3) Experienced employees: Apply flat increase rate
    if n_exp > 0:
        log.debug(
            f"Applying flat rate bump ({rate:.1%}) to {n_exp} experienced employees."
        )
        df2.loc[mask_experienced, comp_col] *= 1 + rate

    # Ensure non-negative compensation
    df2[comp_col] = df2[comp_col].clip(lower=0)

    log.info("Comp bump applied. New avg comp: %.0f", df2[comp_col].mean())
    return df2


def apply_onboarding_bump(
    df: pd.DataFrame,
    comp_col: str,
    ob_cfg: Mapping[str, Any],
    baseline_hire_salaries: pd.Series,
    rng: np.random.Generator,
    log: logging.Logger,
) -> pd.DataFrame:
    """
    Apply onboarding bump for new hires based on configuration.

    Args:
        df: DataFrame of new hires.
        comp_col: Name of the compensation column.
        ob_cfg: Configuration dictionary for onboarding bump (keys: 'enabled', 'method', 'rate').
        baseline_hire_salaries: Series of salaries from previous year's hires for sampling.
        rng: NumPy random number generator.
        log: Logger instance.

    Returns:
        DataFrame with onboarding bump applied.
    """
    df2 = df.copy()
    if not ob_cfg or not ob_cfg.get("enabled", False):
        log.debug("Onboarding bump disabled or config missing.")
        return df2
    if comp_col not in df2.columns:
        log.warning(
            f"Onboarding bump enabled, but comp column '{comp_col}' not found. Skipping."
        )
        return df2

    method = ob_cfg.get("method", "")
    rate = ob_cfg.get("rate", ob_cfg.get("flat_rate", 0.0))  # Support legacy key
    log.info(f"Applying onboarding bump: method='{method}', rate={rate:.2%}")

    if method == "flat_rate":
        df2[comp_col] = pd.to_numeric(df2[comp_col], errors="coerce").fillna(0.0) * (
            1 + rate
        )
    elif method == "sample_plus_rate":
        if not baseline_hire_salaries.empty:
            log.debug(f"Sampling {len(df2)} baseline salaries for onboarding bump.")
            # Ensure choice works with Series, might need .values if it expects array
            draws = rng.choice(baseline_hire_salaries.values, size=len(df2))
            df2[comp_col] = draws * (1 + rate)
        else:
            log.warning(
                "Onboarding bump 'sample_plus_rate' method chosen, but baseline_hire_salaries is empty! Applying flat rate to existing comp instead."
            )
            df2[comp_col] = pd.to_numeric(df2[comp_col], errors="coerce").fillna(
                0.0
            ) * (
                1 + rate
            )  # Fallback
    else:
        log.warning(f"Unknown onboarding bump method '{method}', no bump applied.")

    # Ensure non-negative compensation
    df2[comp_col] = df2[comp_col].clip(lower=0)
    log.debug(
        f"Onboarding bump applied. New avg comp for hires: {df2[comp_col].mean():.0f}"
    )
    return df2
