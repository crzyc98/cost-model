## QuickStart

To generate and use hazard tables programmatically:

```python
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
from cost_model.projections.hazard import build_hazard_table
from cost_model.utils.columns import EMP_ROLE

# Create a sample initial snapshot
initial_snapshot = pd.DataFrame({
    'employee_id': ['EMP001', 'EMP002', 'EMP003', 'EMP004'],
    'employee_role': ['Engineer', 'Manager', 'Engineer', 'Analyst'],
    'tenure_band': ['0-1', '1-3', '3-5', '5+'],
    'active': [True, True, True, True]
})

# Define projection years
projection_years = [2025, 2026, 2027, 2028, 2029]

# Create global parameters
global_params = SimpleNamespace(
    annual_termination_rate=0.12,           # 12% annual termination rate
    annual_compensation_increase_rate=0.03, # 3% annual compensation increase
    new_hire_termination_rate=0.25,        # 25% new hire termination rate
    cola_pct=0.02                          # 2% cost of living adjustment
)

# Create plan rules configuration (simplified example)
plan_rules_config = SimpleNamespace(
    auto_increase_enabled=True,
    auto_increase_rate=0.01,
    auto_increase_cap=0.10
)

# Generate the hazard table
hazard_table = build_hazard_table(
    years=projection_years,
    initial_snapshot=initial_snapshot,
    global_params=global_params,
    plan_rules_config=plan_rules_config
)

# Examine the hazard table
print(f"Generated hazard table with {len(hazard_table)} rows")
print(hazard_table.head())

# Analyze termination rates by role
role_term_rates = hazard_table.groupby(EMP_ROLE)['term_rate'].mean()
print("\nTermination rates by role:")
for role, rate in role_term_rates.items():
    print(f"{role}: {rate:.1%}")

# Analyze compensation increases by tenure band
tenure_comp_rates = hazard_table.groupby('tenure_band')['comp_raise_pct'].mean()
print("\nCompensation increase rates by tenure band:")
for band, rate in tenure_comp_rates.items():
    print(f"{band}: {rate:.1%}")

# Save the hazard table
output_dir = Path('output/hazard')
output_dir.mkdir(parents=True, exist_ok=True)
hazard_table.to_csv(output_dir / 'hazard_table.csv', index=False)

# Use the hazard table in a simulation
for year in projection_years:
    year_hazard = hazard_table[hazard_table['simulation_year'] == year]
    print(f"\nYear {year} hazard rates:")
    print(f"Average termination rate: {year_hazard['term_rate'].mean():.1%}")
    print(f"Average compensation increase: {year_hazard['comp_raise_pct'].mean():.1%}")
```

This demonstrates how to generate and analyze a hazard table for workforce projections.