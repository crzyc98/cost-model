
## QuickStart

To simulate cost-of-living adjustments programmatically:

```python
import pandas as pd
import numpy as np
from pathlib import Path
from cost_model.engines.cola import cola
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
    'employee_termination_date': [None, None, None, None]
}).set_index('employee_id')

# Create a hazard slice for the simulation year
hazard_slice = pd.DataFrame({
    'simulation_year': [2025, 2025, 2025],
    'role': ['Engineer', 'Manager', 'Analyst'],  # Note: 'role' not 'employee_role'
    'tenure_band': ['all', 'all', 'all'],
    'cola_pct': [0.02, 0.02, 0.02]              # 2% cost-of-living adjustment
})

# Set the reference date for COLA
as_of = pd.Timestamp('2025-01-01')

# Generate COLA events (applied 90 days into the year with 30 days of jitter)
cola_events = cola(
    snapshot=snapshot.copy(),
    hazard_slice=hazard_slice,
    as_of=as_of,
    days_into_year=90,    # Apply COLA 90 days into the year (around April 1)
    jitter_days=30,       # Add random jitter of +/- 15 days
    rng=rng
)

# Examine the COLA events
if cola_events and not cola_events[0].empty:
    cola_df = cola_events[0]
    print(f"Generated {len(cola_df)} COLA events:")
    
    # Display COLA details
    print("\nCOLA details:")
    for _, event in cola_df.iterrows():
        emp_id = event['employee_id']
        event_time = event['event_time']
        new_comp = event['value_num']
        
        # Get the old compensation from the snapshot
        old_comp = snapshot.loc[emp_id, 'employee_gross_compensation']
        
        # Calculate the increase
        increase = new_comp - old_comp
        pct_increase = (increase / old_comp) * 100
        
        print(f"  {emp_id}: ${old_comp:,.2f} → ${new_comp:,.2f} (+${increase:,.2f}, +{pct_increase:.1f}%) on {event_time.strftime('%Y-%m-%d')}")
    
    # Update the snapshot with new compensation values
    for _, event in cola_df.iterrows():
        emp_id = event['employee_id']
        new_comp = event['value_num']
        snapshot.at[emp_id, 'employee_gross_compensation'] = new_comp
    
    # Calculate total compensation change
    total_old_comp = sum(snapshot.loc[emp_id, 'employee_gross_compensation'] - increase for emp_id, increase in zip(cola_df['employee_id'], cola_df['delta']))
    total_new_comp = sum(event['value_num'] for _, event in cola_df.iterrows())
    total_increase = total_new_comp - total_old_comp
    total_pct_increase = (total_increase / total_old_comp) * 100 if total_old_comp > 0 else 0
    
    print(f"\nTotal COLA impact:")
    print(f"  Before: ${total_old_comp:,.2f}")
    print(f"  After: ${total_new_comp:,.2f}")
    print(f"  Increase: ${total_increase:,.2f} (+{total_pct_increase:.1f}%)")
    
    # Analyze the distribution of COLA timing
    cola_dates = pd.to_datetime(cola_df['event_time'])
    min_date = cola_dates.min()
    max_date = cola_dates.max()
    mean_date = cola_dates.mean()
    
    print(f"\nCOLA timing distribution:")
    print(f"  Earliest: {min_date.strftime('%Y-%m-%d')}")
    print(f"  Latest: {max_date.strftime('%Y-%m-%d')}")
    print(f"  Average: {mean_date.strftime('%Y-%m-%d')}")
    
    # Save the COLA events and updated snapshot
    output_dir = Path('output/cola')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cola_df.to_parquet(output_dir / 'cola_events.parquet')
    snapshot.to_parquet(output_dir / 'snapshot_after_cola.parquet')
else:
    print("No COLA events generated")
```

This demonstrates how to simulate cost-of-living adjustments with configurable timing and jitter for a more realistic distribution of adjustment dates.