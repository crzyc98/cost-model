# cost_model/engines/hire.py
"""
Engine for generating hire events during workforce simulations.

## QuickStart

To generate hire events programmatically:

```python
import pandas as pd
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from cost_model.engines.hire import run
from cost_model.state.event_log import EVENT_COLS
from cost_model.utils.columns import EMP_ID, EMP_ROLE, EMP_GROSS_COMP, EMP_HIRE_DATE, EMP_BIRTH_DATE

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

import json
import logging
from typing import List
import pandas as pd
import numpy as np

from cost_model.state.event_log import EVT_TERM, EVENT_COLS, EVT_HIRE, EVT_COMP, create_event
from cost_model.utils.columns import EMP_ID, EMP_TERM_DATE, EMP_ROLE, EMP_GROSS_COMP, EMP_HIRE_DATE
from cost_model.dynamics.sampling.new_hires import sample_new_hire_compensation

logger = logging.getLogger(__name__)

from cost_model.utils.columns import EMP_TENURE

from types import SimpleNamespace
from cost_model.utils.columns import EMP_BIRTH_DATE

def run(
    snapshot: pd.DataFrame,
    hires_to_make: int,
    hazard_slice: pd.DataFrame, 
    rng: np.random.Generator,
    census_template_path: str,
    global_params: SimpleNamespace,
    terminated_events: pd.DataFrame = None,
) -> List[pd.DataFrame]:
    """
    Determine how many hires to generate so that the *expected* active headcount at
    year-end meets `target_eoy`. Gross-up using the new-hire term rate from hazard_slice.
    Returns two DataFrames (in the list):
      1. A 'hire' events DataFrame
      2. A first 'comp' event DataFrame for those hires (their starting comp)
    """
    if hazard_slice.empty:
        logger.warning("[HIRE.RUN] Hazard slice is empty. Cannot determine simulation year or new hire term rates. Returning no hires.")
        return [pd.DataFrame(columns=EVENT_COLS), pd.DataFrame(columns=EVENT_COLS)]
    
    simulation_year = hazard_slice['simulation_year'].iloc[0] # Derive simulation_year from hazard_slice
    logger.info(f"[HIRE.RUN YR={simulation_year}] Hires to make (passed-in): {hires_to_make}")
    
    # No need to recalculate hires_to_make - the caller has already done this math
    # including the gross-up for new hire termination rate if needed
    if hires_to_make <= 0:
        logger.info(f"[HIRE.RUN YR={simulation_year}] No hires to make as passed-in value is zero or negative.")
        return [pd.DataFrame(columns=EVENT_COLS), pd.DataFrame(columns=EVENT_COLS)]
    # Assign hires to roles according to proportions
    # Choose roles based on current distribution in snapshot
    # Fall back to a default role if snapshot is empty
    role_counts = snapshot[EMP_ROLE].value_counts(normalize=True)
    roles = role_counts.index.tolist() or ['Staff']
    probs = role_counts.values.tolist() if not role_counts.empty else [1.0]
    role_choices = rng.choice(roles, size=hires_to_make, p=probs)
    # Generate unique employee_ids (assume string IDs)
    existing_ids = (
        set(snapshot[EMP_ID])
        if EMP_ID in snapshot.columns
        else set(snapshot.index) 
    )
    new_ids = []
    i = 1
    while len(new_ids) < hires_to_make:
        # Use simulation_year for generating new hire IDs for consistency
        eid = f"NH_{simulation_year}_{i:04d}"
        if eid not in existing_ids:
            new_ids.append(eid)
        i += 1
    # Generate hire dates uniformly in the year
    start = pd.Timestamp(f"{simulation_year}-01-01")
    end = pd.Timestamp(f"{simulation_year}-12-31")
    days = (end - start).days + 1
    hire_dates = [
        start + pd.Timedelta(days=int(d))
        for d in rng.integers(0, days, size=hires_to_make)
    ]
    # ----- Termination-based sampling with parameterized premium and age jitter -----
    ext_prem = getattr(global_params, 'replacement_hire_premium', 0.02)
    age_sd = getattr(global_params, 'replacement_hire_age_sd', 2)
    pool = None
    if terminated_events is not None and not terminated_events.empty:
        terms = terminated_events[terminated_events.event_type == EVT_TERM]
        # Ensure EMP_ID is only a column, not both index and column
        if EMP_ID in snapshot.index.names and EMP_ID in snapshot.columns:
            snap = snapshot.reset_index(drop=True)
        elif EMP_ID in snapshot.index.names:
            snap = snapshot.reset_index()
        else:
            snap = snapshot
        terms = terms.merge(
            snap[[EMP_ID, EMP_HIRE_DATE, EMP_BIRTH_DATE, EMP_GROSS_COMP]],
            on=EMP_ID, how='left'
        ).drop_duplicates(subset=EMP_ID)
        pool = terms[[EMP_ID, EMP_GROSS_COMP, EMP_HIRE_DATE, EMP_BIRTH_DATE]]
    if pool is not None and len(pool) >= hires_to_make:
        choice_idx = rng.choice(pool.index, size=hires_to_make, replace=True)
        clones = pool.loc[choice_idx].copy().reset_index(drop=True)
        # bump salary by premium
        clones[EMP_GROSS_COMP] *= (1 + ext_prem)
        clones['clone_of'] = pool.loc[choice_idx, EMP_ID].values
        # jitter birth_date ± age_sd years
        bd = pd.to_datetime(clones[EMP_BIRTH_DATE])
        jitter_days = rng.normal(0, age_sd * 365.25, size=len(bd)).astype(int)
        clones[EMP_BIRTH_DATE] = bd + pd.to_timedelta(jitter_days, unit='D')
        # keep hire_date same or optionally reset to uniform in year
        clones[EMP_HIRE_DATE] = clones[EMP_HIRE_DATE]
        starting_comps = clones[EMP_GROSS_COMP].values
        birth_dates = pd.to_datetime(clones[EMP_BIRTH_DATE]).dt.strftime('%Y-%m-%d').values
        clone_of = clones['clone_of'].tolist()
    else:
        prev_salaries = snapshot[EMP_GROSS_COMP].dropna().values
        hires_df = pd.DataFrame({EMP_ID: new_ids, EMP_ROLE: role_choices})
        hires_df = sample_new_hire_compensation(hires_df, comp_col='sampled_comp', prev_salaries=prev_salaries, rng=rng)
        starting_comps = hires_df['sampled_comp'].tolist()
        birth_dates = ['1990-01-01'] * hires_to_make
        clone_of = [''] * hires_to_make
    # Build a DataFrame with one row per new hire for output
    hires_df = pd.DataFrame({EMP_ID: new_ids, EMP_ROLE: role_choices,
                            'sampled_comp': starting_comps,
                            EMP_HIRE_DATE: hire_dates,
                            EMP_BIRTH_DATE: birth_dates,
                            'clone_of': clone_of})

    logger.info(
        f"[HIRE.RUN YR={simulation_year}] Sampled new hire salaries: mean=${np.mean(starting_comps):,.0f}, min=${np.min(starting_comps):,.0f}, max=${np.max(starting_comps):,.0f}"
    )

    # 1. Derive year from hazard_slice['simulation_year']
    simulation_year = int(hazard_slice['simulation_year'].iloc[0])
    # Again, use EOY to filter terminations for placeholder logic
    as_of = pd.Timestamp(f"{simulation_year}-12-31")


    # 3. (Optional) Gross up by new-hire term rate (not implemented here)
    # 4. (Optional) Read census template for realistic new hire sampling (scaffold only)
    # import pandas as pd
    # census_df = pd.read_parquet(census_template_path)
    # TODO: Sample from census_df by role, etc.

    # 5. (Optional) Pull compensation defaults from plan_rules_config if available (scaffold only)
    # base_comp = plan_rules_config.new_hire_compensation_params.comp_base_salary
    # comp_std = plan_rules_config.new_hire_compensation_params.comp_std
    # ...

    # Placeholder logic    # Generate hire events
    hire_events = []
    for idx, (eid, role, dt) in enumerate(zip(new_ids, role_choices, hire_dates)):
        bd = birth_dates[idx]
        co = clone_of[idx]
        hire_events.append(create_event(
            event_time=dt,
            employee_id=eid,
            event_type=EVT_HIRE,
            value_json=json.dumps({'role': role, 'birth_date': bd, 'clone_of': co}),
            meta=f"Hire event for {eid} in {simulation_year}"
        ))
    comp_events = []
    end_of_year = pd.Timestamp(f"{simulation_year}-12-31")
    for eid, dt, comp in zip(new_ids, hire_dates, starting_comps):
        days_worked = (end_of_year - dt).days + 1
        prorated = comp * (days_worked / 365.25)
        value_json = json.dumps({
            "reason": "starting_salary",
            "full_year": comp,
            "days_worked": days_worked
        })
        comp_events.append(
            create_event(
                event_time=dt,
                employee_id=eid,
                event_type=EVT_COMP,
                value_num=prorated,
                value_json=None,
                meta=f"Prorated comp for {eid}"
            )
        )
    hire_df = pd.DataFrame(hire_events, columns=EVENT_COLS).sort_values("event_time", ignore_index=True)
    comp_df = pd.DataFrame(comp_events, columns=EVENT_COLS).sort_values("event_time", ignore_index=True)
    return [hire_df, comp_df]
