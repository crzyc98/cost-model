# cost_model/engines/hire.py

from typing import List
import pandas as pd
import numpy as np
from cost_model.state.event_log import EVENT_COLS, EVT_HIRE, EVT_COMP
from cost_model.utils.columns import EMP_ID, EMP_TERM_DATE, EMP_ROLE
import math
import json
import logging

logger = logging.getLogger(__name__)

def run(
    snapshot: pd.DataFrame,
    target_eoy: int,
    hazard_slice: pd.DataFrame, 
    rng: np.random.Generator,
    census_template_path: str,  # New parameter for census template
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
    as_of = pd.Timestamp(f"{simulation_year}-01-01")
    survivors = snapshot[
        (snapshot[EMP_TERM_DATE].isna()) | (snapshot[EMP_TERM_DATE] > as_of)
    ].copy()
    curr = len(survivors)
    needed = max(0, target_eoy - curr)
    logger.info(f"[HIRE.RUN YR={simulation_year}] Initial survivors (curr): {curr}, Target EOY: {target_eoy}, Needed before gross-up: {needed}")

    if needed == 0:
        logger.info(f"[HIRE.RUN YR={simulation_year}] No hires needed as target is met or exceeded by current survivors.")
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
    # Ensure (1 - avg_nh) is not zero or extremely small to prevent division issues or huge numbers
    denominator = 1 - avg_nh
    if denominator <= 0.001: # If rate is 100% or very close, gross-up isn't meaningful or safe
        hires_to_make = needed if avg_nh < 1 else 0 # If 100% term rate, effectively 0 net hires unless needed is already 0
        logger.warning(f"[HIRE.RUN YR={simulation_year}] Avg NH Term Rate ({avg_nh:.4f}) is >= 99.9%. Gross-up is not applied or hires_to_make set to 0.")
    else:
        hires_to_make = int(math.ceil(needed / denominator))

    logger.info(f"[HIRE.RUN YR={simulation_year}] Avg NH Term Rate: {avg_nh:.4f}, Hires to make (grossed-up): {hires_to_make}")

    if hires_to_make == 0:
        logger.info(f"[HIRE.RUN YR={simulation_year}] Zero hires to make after gross-up calculation (or needed was zero).")
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
    # Assign starting comp using config-driven logic
    plan_rules_config = hazard_slice.iloc[0]['cfg']
    role_comp_params = plan_rules_config.role_compensation_params
    global_comp_params = plan_rules_config.new_hire_compensation_params
    
    starting_comps = []
    for r in role_choices:
        comp = None
        if role_comp_params and r in role_comp_params:
            role_cfg = role_comp_params[r]
            comp = getattr(role_cfg, 'comp_base_salary', None)
        if comp is None and global_comp_params:
            comp = getattr(global_comp_params, 'comp_base_salary', None)
        if comp is None:
            comp = 50000
        starting_comps.append(comp)

    # 1. Derive year from hazard_slice['simulation_year']
    simulation_year = int(hazard_slice['simulation_year'].iloc[0])
    as_of = pd.Timestamp(f"{simulation_year}-01-01")

    # 2. Filter active survivors
    survivors = snapshot[
        snapshot[EMP_TERM_DATE].isna() | (snapshot[EMP_TERM_DATE] > as_of)
    ]
    curr = len(survivors)
    needed = max(0, target_eoy - curr)

    # 3. (Optional) Gross up by new-hire term rate (not implemented here)
    # 4. (Optional) Read census template for realistic new hire sampling (scaffold only)
    # import pandas as pd
    # census_df = pd.read_parquet(census_template_path)
    # TODO: Sample from census_df by role, etc.

    # 5. (Optional) Pull compensation defaults from plan_rules_config if available (scaffold only)
    # base_comp = plan_rules_config.new_hire_compensation_params.comp_base_salary
    # comp_std = plan_rules_config.new_hire_compensation_params.comp_std
    # ...

    # Placeholder logic (existing):
    placeholder_birth_date = "1990-01-01"
    hire_events = [
        {
            "event_id": f"evt_hire_{simulation_year}_{idx:04d}",
            EMP_ID: eid,
            "event_type": EVT_HIRE,
            "event_date": dt,
            "year": simulation_year,
            "value_num": np.nan,
            "value_json": json.dumps({"role": r, "birth_date": placeholder_birth_date}),
            "notes": f"Hire event for {eid} in {simulation_year}",
        }
        for idx, (eid, r, dt) in enumerate(zip(new_ids, role_choices, hire_dates))
    ]
    comp_events = [
        {
            "event_id": f"evt_comp_{simulation_year}_{idx:04d}",
            EMP_ID: eid,
            "event_type": EVT_COMP,
            "event_date": dt,
            "year": simulation_year,
            "value_num": comp,
            "value_json": json.dumps({"reason": "starting_salary"}),
            "notes": f"Starting salary for new hire {eid} in {simulation_year}",
        }
        for idx, (eid, dt, comp) in enumerate(zip(new_ids, hire_dates, starting_comps))
    ]
    return [
        pd.DataFrame(hire_events, columns=EVENT_COLS),
        pd.DataFrame(comp_events, columns=EVENT_COLS),
    ]
