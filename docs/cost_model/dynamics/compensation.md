
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