## QuickStart

To generate hire events programmatically:

```python
import pandas as pd
import json
import logging
import datetime
from pathlib import Path
from types import SimpleNamespace
import numpy as np
try:
    from cost_model.dynamics.sampling.salary import DefaultSalarySampler
except ImportError:
    # Fallback if module not found
    class DefaultSalarySampler:
        def __init__(self, rng=None):
            self.rng = rng or np.random.default_rng()
        
        def sample_new_hires(self, size: int, params: Dict[str, float], ages: Optional[np.ndarray] = None, rng: Optional[np.random.Generator] = None):
            # Fallback to simple normal distribution if DefaultSalarySampler is not available
            base_salary = params.get('comp_base_salary', 70000)
            std_dev = params.get('comp_stochastic_std_dev', 5000)
            active_rng = rng if rng is not None else self.rng
            return pd.Series(active_rng.normal(base_salary, std_dev, size=size))
from cost_model.state.event_log import EVENT_COLS
from cost_model.utils.columns import EMP_ID, EMP_ROLE, EMP_GROSS_COMP, EMP_HIRE_DATE, EMP_BIRTH_DATE, EVT_HIRE
from cost_model.utils.event import create_event

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
    'employee_birth_date': [
        pd.Timestamp('1990-05-12'),
        pd.Timestamp('1985-08-23'),
        pd.Timestamp('1992-11-30'),
        pd.Timestamp('1995-03-15')
    ],
    'employee_termination_date': [None, None, None, None]
}).set_index('employee_id')

# Create a hazard slice for the simulation year
hazard_slice = pd.DataFrame({
    'simulation_year': [2025, 2025, 2025],
    'employee_role': ['Engineer', 'Manager', 'Analyst'],
    'tenure_band': ['all', 'all', 'all'],
    'term_rate': [0.12, 0.08, 0.15],
    'comp_raise_pct': [0.03, 0.04, 0.03],
    'new_hire_termination_rate': [0.25, 0.20, 0.30],
    'cola_pct': [0.02, 0.02, 0.02]
})

# Define global parameters
global_params = SimpleNamespace(
    # New hire parameters
    new_hire_age_mean=30,
    new_hire_age_std=5,
    new_hire_age_min=22,
    new_hire_age_max=45,
    
    # Compensation parameters
    new_hire_comp_base={
        'Engineer': 70000,
        'Manager': 90000,
        'Analyst': 55000
    },
    new_hire_comp_age_factor={
        'Engineer': 1000,
        'Manager': 1500,
        'Analyst': 800
    },
    new_hire_comp_std={
        'Engineer': 5000,
        'Manager': 7000,
        'Analyst': 4000
    }
)

# Path to a census template (for reference data)
census_template_path = 'data/census_template.parquet'

# Number of hires to make
hires_to_make = 5

# Generate hire events
hire_events = run(
    snapshot=snapshot,
    hires_to_make=hires_to_make,
    hazard_slice=hazard_slice,
    rng=rng,
    census_template_path=census_template_path,
    global_params=global_params
)

# Examine the hire events
if len(hire_events) >= 2 and not hire_events[0].empty:
    # First DataFrame contains hire events
    hire_df = hire_events[0]
    # Second DataFrame contains compensation events for the new hires
    comp_df = hire_events[1]
    
    print(f"Generated {len(hire_df)} hire events:")
    
    # Combine the events to analyze the new hires
    hire_info = {}
    for _, event in hire_df.iterrows():
        emp_id = event['employee_id']
        hire_date = event['event_time']
        value_json = event['value_json']
        
        # Parse the JSON data
        if value_json and pd.notna(value_json):
            try:
                data = json.loads(value_json)
                role = data.get('role', 'Unknown')
                birth_date = data.get('birth_date', None)
                hire_info[emp_id] = {'hire_date': hire_date, 'role': role, 'birth_date': birth_date}
            except json.JSONDecodeError:
                hire_info[emp_id] = {'hire_date': hire_date, 'role': 'Unknown', 'birth_date': None}
    
    # Add compensation information
    for _, event in comp_df.iterrows():
        emp_id = event['employee_id']
        comp = event['value_num']
        if emp_id in hire_info:
            hire_info[emp_id]['compensation'] = comp
    
    # Display the new hire information
    print("\nNew hire details:")
    for emp_id, info in hire_info.items():
        hire_date = info.get('hire_date')
        role = info.get('role', 'Unknown')
        comp = info.get('compensation', 0)
        birth_date = info.get('birth_date')
        
        # Calculate age if birth date is available
        age_str = ""
        if birth_date and hire_date:
            birth_date = pd.Timestamp(birth_date)
            age = hire_date.year - birth_date.year - ((hire_date.month, hire_date.day) < (birth_date.month, birth_date.day))
            age_str = f", Age: {age}"
            
        print(f"  {emp_id}: {role}, Hired: {hire_date.strftime('%Y-%m-%d')}{age_str}, Salary: ${comp:,.2f}")
    
    # Analyze the distribution of roles
    roles = [info.get('role', 'Unknown') for info in hire_info.values()]
    role_counts = pd.Series(roles).value_counts()
    
    print("\nRole distribution:")
    for role, count in role_counts.items():
        print(f"  {role}: {count} ({count/len(roles)*100:.1f}%)")
    
    # Save the events
    output_dir = Path('output/hiring')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    hire_df.to_parquet(output_dir / 'hire_events.parquet')
    comp_df.to_parquet(output_dir / 'new_hire_comp_events.parquet')
else:
    print("No hire events generated")
```

This demonstrates how to generate hire events with the appropriate role distribution and compensation based on the current workforce snapshot.
"""