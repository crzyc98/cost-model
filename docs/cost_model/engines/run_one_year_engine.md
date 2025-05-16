## QuickStart

To run a complete simulation year programmatically:

```python
import pandas as pd
import numpy as np
from types import SimpleNamespace
from cost_model.engines.run_one_year_engine import run_one_year
from cost_model.utils.columns import (
    EMP_ID, EMP_ROLE, EMP_GROSS_COMP, EMP_HIRE_DATE, EMP_TERM_DATE,
    EMP_BIRTH_DATE, EMP_ACTIVE, EMP_TENURE_BAND
)

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
    'tenure_band': ['1-3', '1-3', '1-3', '1-3']
})

# Create a sample hazard table
hazard_table = pd.DataFrame({
    'simulation_year': [2025],
    'role': ['All Roles'],
    'tenure_band': ['All Bands'],
    'term_rate': [0.15],
    'comp_raise_pct': [0.03],
    'new_hire_termination_rate': [0.25],
    'cola_pct': [0.02],
    'cfg': [{'global': True}]
})

# Create global parameters namespace
global_params = SimpleNamespace(
    days_into_year_for_cola=90,
    cola_jitter_days=10,
    days_into_year_for_promotion=180,
    promotion_rules={
        'Engineer': {'raise_pct': 0.05},
        'Manager': {'raise_pct': 0.07}
    }
)

# Add compensation configuration
global_params.compensation = {
    'COLA_rate': 0.02,                # Example 2% COLA
    'promo_raise_pct': {'1_to_2': 0.05}, # 5% promotion bump from level 1 to 2
    'merit_dist': {
        'Staff': {'mu': 0.03, 'sigma': 0.01}  # Merit increases distribution for Staff
    }
}

# Create plan rules namespace
plan_rules = SimpleNamespace(
    eligibility_rules={
        'min_age': 18,
        'min_service': 0
    },
    enrollment_rules={
        'auto_enroll': True,
        'default_rate': 0.06
    }
)

# Run the simulation year
event_log_for_year, final_snapshot = run_one_year(
    event_log=event_log,
    prev_snapshot=initial_snapshot,
    year=2025,
    global_params=global_params,
    plan_rules=plan_rules,
    hazard_table=hazard_table,
    rng=rng,
    census_template_path="path/to/census_template",
    rng_seed_offset=0,
    deterministic_term=False
)

print(f"\nEvents generated for year 2025:")
print(event_log_for_year)
print(f"\nFinal snapshot after year 2025:")
print(final_snapshot)
```

This example demonstrates how to:
1. Set up initial data structures (event log, snapshot, hazard table)
2. Configure simulation parameters (global and plan rules)
3. Run the simulation year
4. Process the results (event log and final snapshot)

The engine handles:
- Compensation adjustments (COLA, raises)
- Terminations (both regular and new hire)
- New hires and their compensation
- Plan rule processing (eligibility, enrollment, contributions)
- Snapshot updates and event logging
