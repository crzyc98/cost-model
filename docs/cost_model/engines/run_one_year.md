## QuickStart

To run a complete simulation year programmatically:

```python
import pandas as pd
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from cost_model.engines.run_one_year import run_one_year
from cost_model.utils.columns import EMP_ID, EMP_ROLE, EMP_GROSS_COMP, EMP_HIRE_DATE, EMP_TERM_DATE

# Create a random number generator with a fixed seed for reproducibility
rng = np.random.default_rng(seed=42)

# Create a sample initial event log
event_log = pd.DataFrame(columns=[
    'event_time', 'employee_id', 'event_type', 'value_num', 'value_json', 'meta'
])

# Create a sample initial workforce snapshot
initial_snapshot = pd.DataFrame({
    'employee_id': ['EMP001', 'EMP002', 'EMP003', 'EMP004'],
    'employee_role': ['Engineer', 'Manager', 'Engineer', 'Analyst'],
    'employee_gross_compensation': [75000.0, 85000.0, 65000.0, 55000.0],
    'employee_deferral_rate': [0.06, 0.08, 0.04, 0.03],
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
    'employee_termination_date': [None, None, None, None],
    'tenure_band': ['1-3', '3-5', '1-3', '0-1']
}).set_index('employee_id')

# Create a hazard table for the simulation
hazard_table = pd.DataFrame({
    'simulation_year': [2025, 2025, 2025],
    'role': ['Engineer', 'Manager', 'Analyst'],
    'tenure_band': ['all', 'all', 'all'],
    'term_rate': [0.12, 0.08, 0.15],                 # Regular termination rates
    'new_hire_termination_rate': [0.25, 0.20, 0.30], # Higher rates for new hires
    'comp_raise_pct': [0.03, 0.04, 0.03],           # Annual compensation increases
    'cola_pct': [0.02, 0.02, 0.02],                 # Cost of living adjustment
    'target_headcount': [15, 5, 10]                  # Target headcount by role
})

# Set up global parameters
global_params = SimpleNamespace(
    simulation_start_year=2025,
    simulation_end_year=2030,
    deterministic=False,
    cola_days_into_year=90,      # Apply COLA 90 days into the year
    cola_jitter_days=30,         # Add random jitter of +/- 15 days to COLA timing
    comp_days_into_year=180,     # Apply compensation changes 180 days into the year
    comp_jitter_days=45,         # Add random jitter of +/- 22 days to comp change timing
    headcount_growth_rate=0.05,  # 5% annual headcount growth
    new_hire_age_mean=32,        # Mean age for new hires
    new_hire_age_std=5,          # Standard deviation for new hire ages
    output_dir=Path('output/simulation')
)

# Set up plan rules
plan_rules = SimpleNamespace(
    eligibility_age=21,                 # Minimum age for plan eligibility
    eligibility_service_months=3,       # Minimum service for plan eligibility
    auto_enrollment_rate=0.03,          # Initial auto-enrollment rate
    auto_enrollment_opt_out_rate=0.10,  # Percentage who opt out of auto-enrollment
    auto_increase_rate=0.01,            # Annual automatic increase percentage
    auto_increase_cap=0.10,             # Cap for automatic increases
    auto_increase_opt_out_rate=0.15,    # Percentage who opt out of auto-increase
    proactive_decrease_pct=0.20,        # Percentage who proactively decrease contribution
    proactive_decrease_amount=0.02,     # Amount of proactive decrease
    employer_match_rate=0.50,           # Employer match rate
    employer_match_cap=0.06             # Cap on employer match
)

# Path to census template (for generating new hires)
census_template_path = 'data/census_template.csv'

# Simulation year to run
sim_year = 2025

# Run the simulation for one year
result = run_one_year(
    event_log=event_log,
    prev_snapshot=initial_snapshot,
    year=sim_year,
    global_params=global_params,
    plan_rules=plan_rules,
    hazard_table=hazard_table,
    rng=rng,
    census_template_path=census_template_path
)

# Unpack the results
new_event_log, new_snapshot = result

# Analyze the simulation results
print(f"Simulation results for year {sim_year}:")

# Count events by type
event_counts = new_event_log['event_type'].value_counts()
print("\nEvents generated:")
for event_type, count in event_counts.items():
    print(f"  {event_type}: {count}")

# Analyze workforce changes
active_count = new_snapshot['active'].sum()
terminated_count = new_snapshot[new_snapshot['employee_termination_date'].notna()].shape[0]
new_hire_count = new_snapshot[new_snapshot['employee_hire_date'].dt.year == sim_year].shape[0]

print(f"\nWorkforce summary:")
print(f"  Active employees: {active_count}")
print(f"  Terminated employees: {terminated_count}")
print(f"  New hires: {new_hire_count}")

# Analyze by role
role_counts = new_snapshot[new_snapshot['active']].groupby('employee_role').size()
print("\nActive employees by role:")
for role, count in role_counts.items():
    print(f"  {role}: {count}")

# Analyze compensation
avg_comp = new_snapshot[new_snapshot['active']]['employee_gross_compensation'].mean()
print(f"\nAverage compensation: ${avg_comp:,.2f}")

# Analyze contribution rates
avg_deferral = new_snapshot[new_snapshot['active']]['employee_deferral_rate'].mean() * 100
print(f"Average deferral rate: {avg_deferral:.1f}%")

# Save the results
output_dir = Path('output/simulation')
output_dir.mkdir(parents=True, exist_ok=True)

new_event_log.to_parquet(output_dir / f'event_log_{sim_year}.parquet')
new_snapshot.to_parquet(output_dir / f'snapshot_{sim_year}.parquet')

print(f"\nSimulation results saved to {output_dir}")
```

This demonstrates how to run a complete simulation year, coordinating all workforce dynamics including terminations, compensation changes, COLA adjustments, hiring, and plan rule applications.
