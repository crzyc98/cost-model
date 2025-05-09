# cost_model/engines/hire.py

from typing import List
import pandas as pd
import numpy as np
from cost_model.state.event_log import EVENT_COLS, EVT_HIRE, EVT_COMP
from cost_model.utils.columns import EMP_ID, EMP_TERM_DATE, EMP_ROLE
import math
import json


def run(
    snapshot: pd.DataFrame,
    target_eoy: int,
    hazard_slice: pd.DataFrame, 
    rng: np.random.Generator,
) -> List[pd.DataFrame]:
    """
    Determine how many hires to generate so that the *expected* active headcount at
    year-end meets `target_eoy`. Gross-up using the new-hire term rate from hazard_slice.
    Returns two DataFrames (in the list):
      1. A 'hire' events DataFrame
      2. A first 'comp' event DataFrame for those hires (their starting comp)
    """
    year = pd.Timestamp.now().year 
    as_of = pd.Timestamp(f"{year}-01-01")
    survivors = snapshot[
        (snapshot[EMP_TERM_DATE].isna()) | (snapshot[EMP_TERM_DATE] > as_of)
    ].copy()
    curr = len(survivors)
    needed = max(0, target_eoy - curr)
    if needed == 0:
        return [pd.DataFrame(columns=EVENT_COLS), pd.DataFrame(columns=EVENT_COLS)]
    # New-hire term rates for tenure_band == '0-1'
    nh_slice = hazard_slice[hazard_slice["tenure_band"] == "0-1"]
    nh_rates = nh_slice.set_index(EMP_ROLE)["term_rate"].to_dict()
    # Role proportions among survivors
    role_counts = survivors[EMP_ROLE].value_counts(normalize=True).to_dict()
    # If there are no survivors, assign all to a random role in nh_rates
    if not role_counts and nh_rates:
        role_counts = {list(nh_rates.keys())[0]: 1.0}
    # Average new-hire term rate (weighted)
    avg_nh = sum(role_counts.get(r, 0) * nh_rates.get(r, 0) for r in role_counts)
    hires_to_make = int(math.ceil(needed / (1 - avg_nh))) if avg_nh < 1 else 0
    if hires_to_make == 0:
        return [pd.DataFrame(columns=EVENT_COLS), pd.DataFrame(columns=EVENT_COLS)]
    # Assign hires to roles according to proportions
    roles = list(role_counts.keys()) 
    role_choices = rng.choice(roles, size=hires_to_make, p=list(role_counts.values()))
    # Generate unique employee_ids (assume string IDs)
    existing_ids = (
        set(snapshot[EMP_ID])
        if EMP_ID in snapshot.columns
        else set(snapshot.index) 
    )
    new_ids = []
    i = 1
    while len(new_ids) < hires_to_make:
        eid = f"NH_{year}_{i:04d}"
        if eid not in existing_ids:
            new_ids.append(eid)
        i += 1
    # Generate hire dates uniformly in the year
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")
    days = (end - start).days + 1
    hire_dates = [
        start + pd.Timedelta(days=int(d))
        for d in rng.integers(0, days, size=hires_to_make)
    ]
    # Assign starting comp (use mean or fixed value for demo)
    # Here, just use 50,000 + 10,000*role index for demo
    base_comp = 50000
    comp_step = 10000
    role_to_idx = {r: idx for idx, r in enumerate(roles)} 
    starting_comps = [base_comp + comp_step * role_to_idx[r] for r in role_choices]
    # Build hire events
    # Placeholder birth date for new hires
    placeholder_birth_date = "1990-01-01"
    hire_events = [
        {
            "event_time": dt,
            EMP_ID: eid,
            "event_type": EVT_HIRE,
            "value_num": np.nan, 
            "value_json": json.dumps({"role": r, "birth_date": placeholder_birth_date}),
        }
        for eid, dt, r in zip(new_ids, hire_dates, role_choices)
    ]
    # Build comp events
    comp_events = [
        {
            "event_time": dt,
            EMP_ID: eid,
            "event_type": EVT_COMP,
            "value_num": comp, 
            "meta": json.dumps({"initial_comp": comp}),
        }
        for eid, dt, comp in zip(new_ids, hire_dates, starting_comps)
    ]
    return [
        pd.DataFrame(hire_events, columns=EVENT_COLS),
        pd.DataFrame(comp_events, columns=EVENT_COLS),
    ]
