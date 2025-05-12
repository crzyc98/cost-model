## QuickStart

To run workforce dynamics simulations programmatically:

```python
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from cost_model.dynamics.engine import run_dynamics_for_year

# Create a sample initial workforce snapshot
initial_snapshot = pd.DataFrame({
    'employee_id': ['EMP001', 'EMP002', 'EMP003', 'EMP004'],
    'employee_role': ['Engineer', 'Manager', 'Engineer', 'Analyst'],
    'employee_gross_compensation': [75000.0, 85000.0, 65000.0, 55000.0],
    'active': [True, True, True, True],
    'employee_hire_date': [
        pd.Timestamp('2024-06-01'),
        pd.Timestamp('2023-03-15'),
        pd.Timestamp('2025-01-10'),
        pd.Timestamp('2024-11-05')
    ],
    'employee_birth_date': [
        pd.Timestamp('1990-05-12'),
        pd.Timestamp('1985-08-23'),
        pd.Timestamp('1992-11-30'),
        pd.Timestamp('1995-03-15')
    ],
    'employee_termination_date': [None, None, None, None],
    'tenure_band': ['0-1', '1-3', '0-1', '0-1']
}).set_index('employee_id')

# Define simulation year and configuration
sim_year = 2025

# Create a configuration for the simulation year
year_config = {
    # Termination settings
    'term_rate': 0.12,                  # 12% annual termination rate
    'new_hire_term_rate': 0.25,         # 25% new hire termination rate
    'use_ml_turnover': False,           # Don't use ML for turnover prediction
    
    # Compensation settings
    'comp_raise_pct': 0.03,             # 3% annual compensation increase
    'cola_pct': 0.02,                   # 2% cost of living adjustment
    'promotion_pct': 0.10,              # 10% of employees get promoted
    'promotion_raise_pct': 0.15,        # 15% raise for promotions
    
    # Hiring settings
    'headcount_target': 5,              # Target 5 employees by end of year
    'headcount_floor': 3,               # Minimum 3 employees
    'new_hire_roles': {                 # Distribution of new hire roles
        'Engineer': 0.6,
        'Analyst': 0.3,
        'Manager': 0.1
    },
    'new_hire_comp': {                  # Base compensation by role
        'Engineer': 70000,
        'Analyst': 55000,
        'Manager': 90000
    }
}

# Run the dynamics simulation for the year
updated_snapshot, events_df = run_dynamics_for_year(
    current_df=initial_snapshot,
    year_config=year_config,
    sim_year=sim_year
)

# Analyze the results
print(f"Initial workforce: {len(initial_snapshot)} employees")
print(f"Updated workforce: {len(updated_snapshot)} employees")
print(f"Active employees: {updated_snapshot['active'].sum()}")

# Analyze terminations
terminated = updated_snapshot[~updated_snapshot['active']]
print(f"\nTerminated employees: {len(terminated)}")
if len(terminated) > 0:
    print("Termination details:")
    for idx, emp in terminated.iterrows():
        term_date = emp['employee_termination_date']
        print(f"  {idx}: Terminated on {term_date.strftime('%Y-%m-%d')}")

# Analyze new hires
new_hires = updated_snapshot[updated_snapshot.index.str.startswith('NH_')]
print(f"\nNew hires: {len(new_hires)}")
if len(new_hires) > 0:
    print("New hire details:")
    for idx, emp in new_hires.iterrows():
        hire_date = emp['employee_hire_date']
        role = emp['employee_role']
        comp = emp['employee_gross_compensation']
        print(f"  {idx}: {role} hired on {hire_date.strftime('%Y-%m-%d')} at ${comp:,.2f}")

# Analyze compensation changes
if 'employee_gross_compensation' in initial_snapshot.columns:
    # Only compare employees present in both snapshots
    common_employees = set(initial_snapshot.index) & set(updated_snapshot.index)
    common_employees = [emp for emp in common_employees if emp in initial_snapshot.index and emp in updated_snapshot.index]
    
    if common_employees:
        initial_comp = initial_snapshot.loc[common_employees, 'employee_gross_compensation']
        updated_comp = updated_snapshot.loc[common_employees, 'employee_gross_compensation']
        
        # Calculate changes
        comp_change = updated_comp - initial_comp
        comp_pct_change = (comp_change / initial_comp) * 100
        
        print(f"\nCompensation changes for existing employees:")
        print(f"  Average increase: ${comp_change.mean():,.2f} ({comp_pct_change.mean():.1f}%)")
        print(f"  Total compensation change: ${comp_change.sum():,.2f}")

# Analyze events
print(f"\nGenerated {len(events_df)} events:")
event_counts = events_df['event_type'].value_counts()
for event_type, count in event_counts.items():
    print(f"  {event_type}: {count}")

# Save the results
output_dir = Path('output/dynamics')
output_dir.mkdir(parents=True, exist_ok=True)

# Save the updated snapshot
updated_snapshot.to_parquet(output_dir / f'snapshot_{sim_year}.parquet')

# Save the events
events_df.to_parquet(output_dir / f'events_{sim_year}.parquet')
```

This demonstrates how to run a workforce dynamics simulation for a year, including terminations, compensation changes, and new hires.