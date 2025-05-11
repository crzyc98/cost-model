# cost_model/engines/hire.py

import json
import logging
from typing import List
import pandas as pd
import numpy as np

from cost_model.state.event_log import EVENT_COLS, EVT_HIRE, EVT_COMP, create_event
from cost_model.utils.columns import EMP_ID, EMP_TERM_DATE, EMP_ROLE, EMP_GROSS_COMP
from cost_model.dynamics.sampling.new_hires import sample_new_hire_compensation

logger = logging.getLogger(__name__)

def run(
    snapshot: pd.DataFrame,
    hires_to_make: int,
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
    # Sample starting comp from historical salaries in snapshot
    from cost_model.dynamics.sampling.new_hires import sample_new_hire_compensation
    prev_salaries = snapshot[EMP_GROSS_COMP].dropna().values
    # Build a DataFrame with one row per new hire for sampling
    hires_df = pd.DataFrame({EMP_ID: new_ids, EMP_ROLE: role_choices})
    hires_df = sample_new_hire_compensation(hires_df, comp_col="sampled_comp", prev_salaries=prev_salaries, rng=rng)
    starting_comps = hires_df["sampled_comp"].values
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

    # Placeholder logic (existing):
    placeholder_birth_date = "1990-01-01"
    hire_events = [
        create_event(
            event_time=dt,
            employee_id=eid,
            event_type=EVT_HIRE,
            value_num=None,
            value_json=json.dumps({"role": r, "birth_date": placeholder_birth_date}),
            meta=f"Hire event for {eid} in {simulation_year}"
        )
        for eid, r, dt in zip(new_ids, role_choices, hire_dates)
    ]
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
                value_json=value_json,
                meta=f"Prorated comp for {eid}"
            )
        )
    hire_df = pd.DataFrame(hire_events, columns=EVENT_COLS).sort_values("event_time", ignore_index=True)
    comp_df = pd.DataFrame(comp_events, columns=EVENT_COLS).sort_values("event_time", ignore_index=True)
    return [hire_df, comp_df]
