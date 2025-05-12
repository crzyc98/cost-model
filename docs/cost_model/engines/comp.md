## QuickStart

To simulate compensation changes programmatically:

```python
import pandas as pd
import numpy as np
from pathlib import Path
from cost_model.engines.comp import bump
from cost_model.utils.columns import EMP_ID, EMP_ROLE, EMP_GROSS_COMP, EMP_HIRE_DATE, EMP_TERM_DATE

# Create a random number generator with a fixed seed for reproducibility
rng = np.random.default_rng(seed=42)

# Create a sample workforce snapshot
snapshot = pd.DataFrame({
    'employee_id': ['EMP001', 'EMP002', 'EMP003', 'EMP004'],
    'employee_role': ['Engineer', 'Manager', 'Engineer', 'Analyst'],
    'employee_gross_compensation': [75000.0, 85000.0, 65000.0, 55000.0],
    'active': [True, True, True, True],
    'employee_hire_date': [
        pd.Timestamp('2023-06-01'),
        pd.Timestamp('2022-03-15'),
        pd.Timestamp('2024-01-10'),
        pd.Timestamp('2024-11-05')
    ],
    'employee_termination_date': [None, None, None, None],
    'tenure_band': ['1-3', '3-5', '1-3', '0-1']
}).set_index('employee_id')

# Create a hazard slice for the simulation year
hazard_slice = pd.DataFrame({
    'simulation_year': [2025, 2025, 2025],
    'role': ['Engineer', 'Manager', 'Analyst'],  # Note: 'role' not 'employee_role'
    'tenure_band': ['all', 'all', 'all'],
    'comp_raise_pct': [0.03, 0.04, 0.03]        # Compensation increase percentages
})

# Set the reference date for compensation changes
as_of = pd.Timestamp('2025-01-01')

# Generate compensation change events
comp_events = bump(
    snapshot=snapshot.copy(),
    hazard_slice=hazard_slice,
    as_of=as_of,
    rng=rng
)

# Examine the compensation change events
if comp_events and not comp_events[0].empty:
    comp_df = comp_events[0]
    print(f"Generated {len(comp_df)} compensation change events:")
    
    # Display compensation change details
    print("\nCompensation change details:")
    for _, event in comp_df.iterrows():
        emp_id = event['employee_id']
        event_time = event['event_time']
        new_comp = event['value_num']
        
        # Get the old compensation from the snapshot
        old_comp = snapshot.loc[emp_id, 'employee_gross_compensation']
        
        # Calculate the increase
        increase = new_comp - old_comp
        pct_increase = (increase / old_comp) * 100
        
        print(f"  {emp_id}: ${old_comp:,.2f} → ${new_comp:,.2f} (+${increase:,.2f}, +{pct_increase:.1f}%)")
    
    # Update the snapshot with new compensation values
    for _, event in comp_df.iterrows():
        emp_id = event['employee_id']
        new_comp = event['value_num']
        snapshot.at[emp_id, 'employee_gross_compensation'] = new_comp
    
    # Analyze compensation changes by role
    print("\nCompensation changes by role:")
    for role in snapshot['employee_role'].unique():
        role_df = snapshot[snapshot['employee_role'] == role]
        avg_comp = role_df['employee_gross_compensation'].mean()
        print(f"  {role}: ${avg_comp:,.2f} average")
    
    # Calculate total compensation change
    total_old_comp = sum(old_comp for _, event in comp_df.iterrows())
    total_new_comp = sum(event['value_num'] for _, event in comp_df.iterrows())
    total_increase = total_new_comp - total_old_comp
    total_pct_increase = (total_increase / total_old_comp) * 100 if total_old_comp > 0 else 0
    
    print(f"\nTotal compensation change:")
    print(f"  Before: ${total_old_comp:,.2f}")
    print(f"  After: ${total_new_comp:,.2f}")
    print(f"  Increase: ${total_increase:,.2f} (+{total_pct_increase:.1f}%)")
    
    # Save the compensation events and updated snapshot
    output_dir = Path('output/compensation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comp_df.to_parquet(output_dir / 'compensation_events.parquet')
    snapshot.to_parquet(output_dir / 'snapshot_after_compensation.parquet')
else:
    print("No compensation change events generated")
```

This demonstrates how to simulate compensation changes based on role-specific increase percentages from the hazard table.