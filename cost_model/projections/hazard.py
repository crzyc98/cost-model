# cost_model/projections/hazard.py
"""
Hazard module for generating hazard tables used in workforce projections.

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
"""

import pandas as pd
import logging
from typing import List
from cost_model.utils.columns import EMP_ROLE

logger = logging.getLogger(__name__)

def build_hazard_table(
    years: List[int],
    initial_snapshot: pd.DataFrame,
    global_params,
    plan_rules_config
) -> pd.DataFrame:
    """Generates the hazard table based on configuration and initial snapshot."""
    logger.info("Generating hazard table...")
    from cost_model.utils.columns import EMP_TENURE
    if EMP_ROLE in initial_snapshot.columns and 'tenure_band' in initial_snapshot.columns:
        unique_roles_tenures = initial_snapshot[[EMP_ROLE, 'tenure_band']].drop_duplicates().to_dict('records')
    else:
        logger.warning(f"'{EMP_ROLE}' or 'tenure_band' not in initial snapshot. Using default 'all'/'all'.")
        unique_roles_tenures = [{EMP_ROLE: 'all', 'tenure_band': 'all'}]

    global_term_rate = getattr(global_params, 'annual_termination_rate', 0.10)
    global_comp_raise_pct = getattr(global_params, 'annual_compensation_increase_rate', 0.03)
    global_nh_term_rate = getattr(global_params, 'new_hire_termination_rate', 0.25)
    logger.info(f"Using global rates: Term={global_term_rate}, CompPct={global_comp_raise_pct}")

    records = []
    for year in years:
        for combo in unique_roles_tenures:
            records.append({
                'simulation_year': year,
                EMP_ROLE: combo[EMP_ROLE],
                'tenure_band': combo['tenure_band'],
                'term_rate': global_term_rate,
                'comp_raise_pct': global_comp_raise_pct,
                'new_hire_termination_rate': global_nh_term_rate,
                'cola_pct': getattr(global_params, 'cola_pct', 0.0),
                'cfg': plan_rules_config
            })
    if records:
        df = pd.DataFrame(records)
        logger.info(f"Hazard table with {len(records)} rows.")
    else:
        cols = ['simulation_year', EMP_ROLE, 'tenure_band', 'term_rate', 'comp_raise_pct', 'new_hire_termination_rate', 'cola_pct', 'cfg']
        df = pd.DataFrame(columns=cols)
        logger.warning("Empty hazard table created.")
    return df
