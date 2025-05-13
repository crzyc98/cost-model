# cost_model/engines/hire.py
"""
Engine for generating hire events during workforce simulations.
QuickStart: see docs/cost_model/engines/hire.md
"""

import json
import logging
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import datetime

from cost_model.state.event_log import EVT_TERM, EVENT_COLS, EVT_HIRE, EVT_COMP, create_event
from cost_model.utils.columns import EMP_ID, EMP_TERM_DATE, EMP_ROLE, EMP_GROSS_COMP, EMP_HIRE_DATE
from cost_model.dynamics.sampling.new_hires import sample_new_hire_compensation
from cost_model.dynamics.sampling.salary import DefaultSalarySampler

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
    
    # Get role and default compensation parameters from various possible locations
    role_comp_params = {}
    default_params = {}
    
    # Try to get role_compensation_params from different possible locations
    # 1. Try from global_params.compensation.roles (nested structure in dev_tiny.yaml)
    if hasattr(global_params, 'compensation') and hasattr(global_params.compensation, 'roles'):
        logger.info(f"[HIRE.RUN YR={simulation_year}] Using role_comp_params from global_params.compensation.roles")
        role_comp_params = global_params.compensation.roles
        if not isinstance(role_comp_params, dict):
            role_comp_params = vars(role_comp_params) if hasattr(role_comp_params, '__dict__') else {}
    # 2. Try direct attribute on global_params
    elif hasattr(global_params, 'role_compensation_params'):
        logger.info(f"[HIRE.RUN YR={simulation_year}] Using role_comp_params from global_params.role_compensation_params")
        role_comp_params = global_params.role_compensation_params
        if not isinstance(role_comp_params, dict):
            role_comp_params = vars(role_comp_params) if hasattr(role_comp_params, '__dict__') else {}
    
    # Try to get new_hire_compensation_params from different possible locations
    # 1. Try from global_params.compensation.new_hire (nested structure in dev_tiny.yaml)
    if hasattr(global_params, 'compensation') and hasattr(global_params.compensation, 'new_hire'):
        logger.info(f"[HIRE.RUN YR={simulation_year}] Using default_params from global_params.compensation.new_hire")
        default_params = global_params.compensation.new_hire
        if not isinstance(default_params, dict):
            default_params = vars(default_params) if hasattr(default_params, '__dict__') else {}
    # 2. Try direct attribute on global_params
    elif hasattr(global_params, 'new_hire_compensation_params'):
        logger.info(f"[HIRE.RUN YR={simulation_year}] Using default_params from global_params.new_hire_compensation_params")
        default_params = global_params.new_hire_compensation_params
        if not isinstance(default_params, dict):
            default_params = vars(default_params) if hasattr(default_params, '__dict__') else {}
            
    # Log available roles for debugging
    logger.info(f"[HIRE.RUN YR={simulation_year}] Available roles in role_comp_params: {list(role_comp_params.keys()) if isinstance(role_comp_params, dict) else 'None'}")

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
        # Fix: Generate realistic birth dates based on parameters rather than hardcoding
        # --- Use DefaultSalarySampler for config-driven salary sampling ---
        # Try to get age parameters from multiple possible locations
        # 1. From default_params dict
        new_hire_age_mean = default_params.get("new_hire_average_age", None)
        new_hire_age_std = default_params.get("new_hire_age_std_dev", None)
        new_hire_age_min = default_params.get("new_hire_age_min", None)
        new_hire_age_max = default_params.get("new_hire_age_max", None)
        
        # 2. If not found, try direct attributes on global_params
        if new_hire_age_mean is None:
            new_hire_age_mean = getattr(global_params, "new_hire_average_age", 30)
        if new_hire_age_std is None:
            new_hire_age_std = getattr(global_params, "new_hire_age_std_dev", 5)
        if new_hire_age_min is None:
            new_hire_age_min = getattr(global_params, "new_hire_age_min", 22)
        if new_hire_age_max is None:
            new_hire_age_max = getattr(global_params, "new_hire_age_max", 45)
        # Generate ages using normal distribution with truncation
        ages = rng.normal(new_hire_age_mean, new_hire_age_std, size=hires_to_make)
        ages = np.clip(ages, new_hire_age_min, new_hire_age_max)

        # Prepare role-specific compensation parameters
        sampler = DefaultSalarySampler(rng)
        starting_comps = []
        for idx, role in enumerate(role_choices):
            # Use per-role params with fallback to default params if role not found
            if role in role_comp_params:
                params = role_comp_params[role]
                logger.info(f"[HIRE.RUN YR={simulation_year}] Using role-specific params for {role}")
            else:
                # Fall back to default params if role not found
                logger.warning(f"[HIRE.RUN YR={simulation_year}] Role '{role}' not found in role_comp_params. Using default params.")
                params = default_params
            
            comp = sampler.sample_new_hires(
                size=1,
                params=params,
                ages=np.array([ages[idx]]),
                rng=rng
            ).iloc[0]
            starting_comps.append(comp)

        # Convert ages to birth dates based on hire dates
        birth_dates = []
        for hire_date, age in zip(hire_dates, ages):
            # Calculate birth year by subtracting age from hire date year
            birth_year = hire_date.year - int(age)
            
            # Random month and day
            month = rng.integers(1, 13)  # 1-12
            max_days = 28 if month == 2 else 30 if month in [4, 6, 9, 11] else 31
            day = rng.integers(1, max_days + 1)  # 1-28/30/31
            
            # Create birth date
            birth_date = pd.Timestamp(f"{birth_year}-{month:02d}-{day:02d}")
            
            # Adjust if the birth date would make them older/younger than intended
            # by checking if their birthday has occurred yet this year
            actual_age = hire_date.year - birth_date.year - ((hire_date.month, hire_date.day) < (birth_date.month, birth_date.day))
            if actual_age != int(age):
                # Adjust the year up or down by 1 to get the correct age
                birth_year += (int(age) - actual_age)
                birth_date = pd.Timestamp(f"{birth_year}-{month:02d}-{day:02d}")
            
            birth_dates.append(birth_date.strftime('%Y-%m-%d'))
        
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
    # Log detailed salary distribution statistics
    logger.info(f"[HIRE.RUN YR={simulation_year}] Salary distribution stats:")
    logger.info(f"[HIRE.RUN YR={simulation_year}]   25th percentile: ${np.percentile(starting_comps, 25):,.0f}")
    logger.info(f"[HIRE.RUN YR={simulation_year}]   Median: ${np.percentile(starting_comps, 50):,.0f}")
    logger.info(f"[HIRE.RUN YR={simulation_year}]   75th percentile: ${np.percentile(starting_comps, 75):,.0f}")

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

    # Generate hire events
    hire_events = []
    for eid, role, dt, bd_raw, co in zip(new_ids, role_choices, hire_dates, birth_dates, clone_of):
        # Debug: Log birth date values and types
        logger.debug(f"Processing birth date for {eid}: type={type(bd_raw)}, value={bd_raw}")
        
        # Ensure birth date is properly converted to datetime
        try:
            if pd.isna(bd_raw):
                bd = pd.NaT
                bd_str = None
            elif isinstance(bd_raw, (pd.Timestamp, datetime.date, datetime.datetime)):
                bd = pd.to_datetime(bd_raw)
                bd_str = bd.strftime("%Y-%m-%d")
            else:
                # Try to parse string as datetime
                bd = pd.to_datetime(bd_raw, errors='coerce')
                bd_str = bd.strftime("%Y-%m-%d") if not pd.isna(bd) else None
                
            # Debug: Log final birth date value
            logger.debug(f"Processed birth date for {eid}: final={bd}, str={bd_str}")
            
            payload = {
                'role': role,
                'birth_date': bd_str,
                'clone_of': co or ''
            }
            
            hire_events.append(create_event(
                event_time=dt,
                employee_id=eid,
                event_type=EVT_HIRE,
                value_num=None,
                value_json=json.dumps(payload),
                meta=f"Hire event for {eid} in {simulation_year}"
            ))
        except Exception as e:
            logger.error(f"Error processing birth date for {eid}: {str(e)}")
            raise
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
    