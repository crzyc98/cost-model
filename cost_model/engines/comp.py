# cost_model/engines/comp.py
"""
Engine for simulating compensation changes during workforce simulations.

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
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import List

from cost_model.state.event_log import EVENT_COLS, EVT_COMP
from cost_model.utils.columns import (
    EMP_ID, EMP_TERM_DATE, EMP_ROLE, EMP_GROSS_COMP, EMP_HIRE_DATE
)
from cost_model.dynamics.sampling.salary import DefaultSalarySampler

logger = logging.getLogger(__name__)
from typing import List
from cost_model.state.event_log import EVENT_COLS, EVT_COMP
from cost_model.utils.columns import EMP_ID, EMP_TERM_DATE, EMP_ROLE, EMP_GROSS_COMP
import logging

logger = logging.getLogger(__name__)

def bump(
    snapshot: pd.DataFrame,
    hazard_slice: pd.DataFrame,
    as_of: pd.Timestamp,
    rng: np.random.Generator
) -> List[pd.DataFrame]:
    """
    Apply the comp_raise_pct from hazard_slice for each active employee,
    and emit one DataFrame of compensation bump events adhering to EVENT_COLS.
    """
    from cost_model.utils.columns import EMP_TENURE
    # 1) Derive year and filter active
    year = int(hazard_slice["simulation_year"].iloc[0])
    as_of = pd.Timestamp(as_of)
    active = snapshot[
        snapshot[EMP_TERM_DATE].isna() | (snapshot[EMP_TERM_DATE] > as_of)
    ].copy()

    # 2) Ensure EMP_ID is a column
    if EMP_ID not in active.columns:
        if active.index.name == EMP_ID:
            active = active.reset_index()
        else:
            raise ValueError(f"{EMP_ID} not found in active snapshot")

    # 3) Merge in the raise pct
    # ensure 'role' from hazard_slice is mapped to EMP_ROLE for merge
    hz = hazard_slice[['role', 'tenure_band', 'comp_raise_pct']].rename(columns={'role': EMP_ROLE})
    df = active.merge(
        hz,
        on=[EMP_ROLE, 'tenure_band'],
        how='left'
    ).fillna({'comp_raise_pct': 0})

    # 4) Only rows with a positive raise
    df = df[df["comp_raise_pct"] > 0].copy()
    if df.empty:
        return [pd.DataFrame(columns=EVENT_COLS)]

    # --- 3. compute old + new comp ---
    df["old_comp"] = df[EMP_GROSS_COMP].astype(float).fillna(0.0)
    df["new_comp"] = (df["old_comp"] * (1 + df["comp_raise_pct"])).round(2)

    # tenure in years as of Jan1
    jan1 = pd.Timestamp(f"{year}-01-01")
    hire_dates = pd.to_datetime(df[EMP_HIRE_DATE], errors="coerce")
    tenure = ((jan1 - hire_dates).dt.days / 365.25).astype(int)
    df["tenure"] = tenure  # REQUIRED for sampler's mask
    mask_second = tenure == 1
    if mask_second.any():
        sampler = DefaultSalarySampler(rng=rng)
        # Use normal distribution for second-year bumps
        mean = df.loc[mask_second, "comp_raise_pct"].mean()
        df.loc[mask_second, "new_comp"] = sampler.sample_second_year(
            df.loc[mask_second],
            comp_col="old_comp",
            dist={"type": "normal", "mean": mean, "std": 0.01},
            rng=rng
        )

    df["event_id"]    = df.index.map(lambda i: f"evt_comp_{year}_{i:04d}")
    df["event_time"]  = as_of
    df["event_type"]  = EVT_COMP
    # **this** is what snapshot.update will write into EMP_GROSS_COMP
    df["value_num"]   = df["new_comp"]
    # keep pct / audit in JSON
    df["value_json"] = df.apply(lambda row: json.dumps({
        "reason": "cola_bump",
        "pct": row["comp_raise_pct"],
        "old_comp": row["old_comp"],
        "new_comp": row["new_comp"]
    }), axis=1)
    df["meta"] = df.apply(lambda row: f"COLA bump for {row[EMP_ID]}: {row['old_comp']} -> {row['new_comp']} (+{row['comp_raise_pct']*100:.2f}%)", axis=1)
    # Remove notes column if present
    if "notes" in df.columns:
        df = df.drop(columns=["notes"])

    # 6) Slice to exactly the EVENT_COLS schema
    events = df[EVENT_COLS]
    return [events]