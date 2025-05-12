## QuickStart

To simulate employee terminations programmatically:

```python
import pandas as pd
import numpy as np
from pathlib import Path
from cost_model.engines.term import run, run_new_hires
from cost_model.utils.columns import EMP_ID, EMP_ROLE, EMP_HIRE_DATE, EMP_TERM_DATE, EMP_TENURE

# Create a random number generator with a fixed seed for reproducibility
rng = np.random.default_rng(seed=42)

# Create a sample workforce snapshot
snapshot = pd.DataFrame({
    'employee_id': ['EMP001', 'EMP002', 'EMP003', 'EMP004', 'NH_001', 'NH_002'],
    'employee_role': ['Engineer', 'Manager', 'Engineer', 'Analyst', 'Engineer', 'Analyst'],
    'employee_gross_compensation': [75000.0, 85000.0, 65000.0, 55000.0, 70000.0, 52000.0],
    'active': [True, True, True, True, True, True],
    'employee_hire_date': [
        pd.Timestamp('2023-06-01'),  # Experienced employee
        pd.Timestamp('2022-03-15'),  # Experienced employee
        pd.Timestamp('2024-01-10'),  # Experienced employee
        pd.Timestamp('2024-11-05'),  # Experienced employee
        pd.Timestamp('2025-02-15'),  # New hire
        pd.Timestamp('2025-03-10')   # New hire
    ],
    'employee_termination_date': [None, None, None, None, None, None],
    'tenure_band': ['1-3', '3-5', '1-3', '0-1', '0-1', '0-1']
}).set_index('employee_id')

# Create a hazard slice for the simulation year
hazard_slice = pd.DataFrame({
    'simulation_year': [2025, 2025, 2025],
    'role': ['Engineer', 'Manager', 'Analyst'],  # Note: 'role' not 'employee_role'
    'tenure_band': ['all', 'all', 'all'],
    'term_rate': [0.12, 0.08, 0.15],             # Regular termination rates
    'new_hire_termination_rate': [0.25, 0.20, 0.30]  # Higher rates for new hires
})

# Simulation year
sim_year = 2025

# Method 1: Simulate terminations for all employees
term_events = run(
    snapshot=snapshot.copy(),
    hazard_slice=hazard_slice,
    rng=rng,
    deterministic=False  # Use random sampling (not deterministic)
)

# Examine the termination events
if term_events and not term_events[0].empty:
    term_df = term_events[0]
    print(f"Generated {len(term_df)} termination events:")
    
    # Display termination details
    print("\nTermination details:")
    for _, event in term_df.iterrows():
        emp_id = event['employee_id']
        term_date = event['event_time']
        print(f"  {emp_id}: Terminated on {term_date.strftime('%Y-%m-%d')}")
    
    # Update the snapshot with termination dates
    for _, event in term_df.iterrows():
        emp_id = event['employee_id']
        term_date = event['event_time']
        if emp_id in snapshot.index:
            snapshot.at[emp_id, 'employee_termination_date'] = term_date
            snapshot.at[emp_id, 'active'] = False
else:
    print("No termination events generated for regular employees")

# Method 2: Simulate terminations specifically for new hires
new_hire_term_events = run_new_hires(
    snapshot=snapshot.copy(),
    hazard_slice=hazard_slice,
    rng=rng,
    year=sim_year,
    deterministic=False  # Use random sampling (not deterministic)
)

# Examine the new hire termination events
if new_hire_term_events and not new_hire_term_events[0].empty:
    nh_term_df = new_hire_term_events[0]
    print(f"\nGenerated {len(nh_term_df)} termination events for new hires:")
    
    # Display termination details for new hires
    print("\nNew hire termination details:")
    for _, event in nh_term_df.iterrows():
        emp_id = event['employee_id']
        term_date = event['event_time']
        print(f"  {emp_id}: Terminated on {term_date.strftime('%Y-%m-%d')}")
    
    # Update the snapshot with termination dates for new hires
    for _, event in nh_term_df.iterrows():
        emp_id = event['employee_id']
        term_date = event['event_time']
        if emp_id in snapshot.index:
            snapshot.at[emp_id, 'employee_termination_date'] = term_date
            snapshot.at[emp_id, 'active'] = False
else:
    print("\nNo termination events generated for new hires")

# Analyze the final workforce after terminations
active_count = snapshot['active'].sum()
terminated_count = len(snapshot) - active_count

print(f"\nFinal workforce after terminations:")
print(f"  Total employees: {len(snapshot)}")
print(f"  Active employees: {active_count}")
print(f"  Terminated employees: {terminated_count}")

# Save the termination events and updated snapshot
output_dir = Path('output/terminations')
output_dir.mkdir(parents=True, exist_ok=True)

if term_events and not term_events[0].empty:
    term_df.to_parquet(output_dir / 'termination_events.parquet')

if new_hire_term_events and not new_hire_term_events[0].empty:
    nh_term_df.to_parquet(output_dir / 'new_hire_termination_events.parquet')

snapshot.to_parquet(output_dir / 'snapshot_after_terminations.parquet')
```

This demonstrates how to simulate terminations for both regular employees and new hires based on configured termination rates.