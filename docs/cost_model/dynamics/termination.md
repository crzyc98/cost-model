## QuickStart

To simulate employee terminations programmatically:

```python
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from cost_model.dynamics.termination import apply_turnover

# Create a sample workforce snapshot
snapshot = pd.DataFrame({
    'employee_id': ['EMP001', 'EMP002', 'EMP003', 'EMP004', 'EMP005'],
    'employee_role': ['Engineer', 'Manager', 'Engineer', 'Analyst', 'Engineer'],
    'employee_gross_compensation': [75000.0, 85000.0, 65000.0, 55000.0, 72000.0],
    'active': [True, True, True, True, True],
    'employee_hire_date': [
        pd.Timestamp('2023-06-01'),  # 2 years of service
        pd.Timestamp('2022-03-15'),  # 3 years of service
        pd.Timestamp('2024-01-10'),  # 1.5 years of service
        pd.Timestamp('2024-11-05'),  # 0.5 years of service
        pd.Timestamp('2025-01-15')   # 0.25 years of service (new hire)
    ],
    'employee_termination_date': [None, None, None, None, None],
    'tenure_band': ['1-3', '3-5', '1-3', '0-1', '0-1']
}).set_index('employee_id')

# Set simulation period
start_date = pd.Timestamp('2025-01-01')
end_date = pd.Timestamp('2025-12-31')

# Method 1: Apply a flat termination rate to all employees
flat_rate = 0.15  # 15% annual termination rate

updated_snapshot_flat, term_events_flat = apply_turnover(
    df=snapshot.copy(),
    hire_col='employee_hire_date',
    probs_or_rate=flat_rate,
    start_date=start_date,
    end_date=end_date,
    term_date_col='employee_termination_date',
    active_col='active',
    new_hire_term_rate=0.25  # Higher termination rate for new hires
)

print(f"Flat rate termination results:")
print(f"  Initial employees: {len(snapshot)}")
print(f"  Remaining active: {updated_snapshot_flat['active'].sum()}")
print(f"  Terminated: {len(updated_snapshot_flat) - updated_snapshot_flat['active'].sum()}")
print(f"  Generated {len(term_events_flat)} termination events")

# Method 2: Apply different termination rates based on tenure
# Define termination probabilities per employee
tenure_based_probs = pd.Series([
    0.10,  # EMP001: 10% probability (1-3 years tenure)
    0.05,  # EMP002: 5% probability (3-5 years tenure)
    0.10,  # EMP003: 10% probability (1-3 years tenure)
    0.20,  # EMP004: 20% probability (0-1 years tenure)
    0.25   # EMP005: 25% probability (0-1 years tenure, new hire)
], index=snapshot.index)

updated_snapshot_tenure, term_events_tenure = apply_turnover(
    df=snapshot.copy(),
    hire_col='employee_hire_date',
    probs_or_rate=tenure_based_probs,
    start_date=start_date,
    end_date=end_date,
    term_date_col='employee_termination_date',
    active_col='active'
)

print(f"\nTenure-based termination results:")
print(f"  Initial employees: {len(snapshot)}")
print(f"  Remaining active: {updated_snapshot_tenure['active'].sum()}")
print(f"  Terminated: {len(updated_snapshot_tenure) - updated_snapshot_tenure['active'].sum()}")
print(f"  Generated {len(term_events_tenure)} termination events")

# Method 3: Apply role-based termination rates
# First create a mapping of roles to termination rates
role_term_rates = {
    'Engineer': 0.12,
    'Manager': 0.08,
    'Analyst': 0.15
}

# Then map these rates to each employee based on their role
role_based_probs = pd.Series(
    [role_term_rates[role] for role in snapshot['employee_role']],
    index=snapshot.index
)

updated_snapshot_role, term_events_role = apply_turnover(
    df=snapshot.copy(),
    hire_col='employee_hire_date',
    probs_or_rate=role_based_probs,
    start_date=start_date,
    end_date=end_date,
    term_date_col='employee_termination_date',
    active_col='active'
)

print(f"\nRole-based termination results:")
print(f"  Initial employees: {len(snapshot)}")
print(f"  Remaining active: {updated_snapshot_role['active'].sum()}")
print(f"  Terminated: {len(updated_snapshot_role) - updated_snapshot_role['active'].sum()}")
print(f"  Generated {len(term_events_role)} termination events")

# Examine termination dates
if len(term_events_role) > 0:
    print("\nTermination events:")
    for _, event in term_events_role.iterrows():
        emp_id = event['employee_id']
        term_date = updated_snapshot_role.loc[emp_id, 'employee_termination_date']
        print(f"  {emp_id} terminated on {term_date.strftime('%Y-%m-%d')}")

# Save the results
output_dir = Path('output/terminations')
output_dir.mkdir(parents=True, exist_ok=True)
updated_snapshot_role.to_parquet(output_dir / 'snapshot_with_terminations.parquet')
term_events_role.to_parquet(output_dir / 'termination_events.parquet')
```

This demonstrates how to simulate employee terminations using flat rates, tenure-based rates, and role-based rates.