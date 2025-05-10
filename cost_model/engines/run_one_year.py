# cost_model/engines/run_one_year.py

import datetime
import json
import logging

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from math import ceil
from types import SimpleNamespace

from . import comp, term, hire
from cost_model.state.event_log import EVENT_COLS
from cost_model.state import snapshot
from cost_model.utils.columns import EMP_TERM_DATE

from cost_model.plan_rules.eligibility import run as eligibility_run
from cost_model.plan_rules.eligibility_events import run as eligibility_events_run
from cost_model.plan_rules.enrollment import run as enrollment_run
from cost_model.plan_rules.contribution_increase import run as contrib_increase_run
from cost_model.plan_rules.proactive_decrease import run as proactive_decrease_run

# Initialize logger for this module
logger = logging.getLogger(__name__)

def run_one_year(
    event_log: pd.DataFrame,
    prev_snapshot: pd.DataFrame,
    year: int,
    config: SimpleNamespace,
    hazard_table: pd.DataFrame,
    rng: np.random.Generator,
    census_template_path: str,
    *,
    rng_seed_offset: int = 0,
    deterministic_term: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    1. extract Jan 1 snapshot (prev_snapshot)
    2. hazard_slice = hazard_table[hazard_table.simulation_year == year]
    3. comp_events  = comp.bump(prev_snapshot, hazard_slice, as_of=Jan1)
    4. term_events  = term.run(prev_snapshot, hazard_slice, rng, deterministic_term)
    5. temporary_snapshot = snapshot.update(prev_snapshot, comp_events+term_events)
    6. survivors = count active in temporary_snapshot
    7. target = ceil(active * (1+growth_rate))
    8. hire_events, hire_comp_events = hire.run(temporary_snapshot, target, hazard_slice, rng, census_template_path)
    9. all_new_events = comp_events + term_events + hire_events + hire_comp_events
    10. updated_snapshot = snapshot.update(temporary_snapshot, all_new_events)
    11. return updated_snapshot, event_log.append(all_new_events)
    """
    # 1. prev_snapshot is already Jan 1
    # 2. only keep the rows for THIS simulation_year
    hazard_slice = hazard_table[hazard_table['simulation_year'] == year]
    if hazard_slice.empty:
        logger.warning(f"[RUN_ONE_YEAR YR={year}] No hazard rates found for year {year}. Using default zero rates.")
        # Create a default slice with all expected columns to prevent downstream errors
        default_rates = {
            'simulation_year': year,
            'term_rate': 0.0,
            'growth_rate': 0.0,
            'comp_raise_pct': 0.0,
            'new_hire_termination_rate': 0.0
        }
        hazard_slice = pd.DataFrame([default_rates])

    # Log the key rates being used for the year from hazard_slice
    log_term_rate = hazard_slice['term_rate'].mean() if 'term_rate' in hazard_slice.columns and not hazard_slice.empty else "N/A"
    log_growth_rate = hazard_slice['growth_rate'].mean() if 'growth_rate' in hazard_slice.columns and not hazard_slice.empty else "N/A"
    log_comp_rate = hazard_slice['comp_raise_pct'].mean() if 'comp_raise_pct' in hazard_slice.columns and not hazard_slice.empty else "N/A"
    log_new_hire_term_rate = hazard_slice['new_hire_termination_rate'].mean() if 'new_hire_termination_rate' in hazard_slice.columns and not hazard_slice.empty else "N/A"
    logger.info(f"[RUN_ONE_YEAR YR={year}] Hazard Slice Rates: Term={log_term_rate}, Growth={log_growth_rate}, CompRaise={log_comp_rate}, NewHireTerm={log_new_hire_term_rate}")
    
    # Use rng_seed_offset for reproducibility if provided
    if rng_seed_offset:
        year_specific_rng = np.random.default_rng(rng_seed_offset)
    else:
        year_specific_rng = rng

    as_of = pd.Timestamp(f"{year}-01-01")

    # --- PHASE 4 PLAN RULE ENGINES ---
    all_event_dfs_for_year = []
    prev_as_of = pd.Timestamp(f"{year-1}-01-01")
    new_events = pd.DataFrame([], columns=EVENT_COLS)

    # Extract scenario config for this year
    cfg = hazard_slice.iloc[0].cfg

    # Eligibility
    evs = eligibility_run(prev_snapshot, as_of, getattr(cfg, 'eligibility', None))
    if evs:
        evs_nonempty = [df for df in evs if isinstance(df, pd.DataFrame) and not df.empty]
        if evs_nonempty:
            new_events = pd.concat([new_events, *evs_nonempty], ignore_index=True)
        # else: skip concat if evs_nonempty is empty

    # Eligibility Milestones
    eligibility_events_cfg = getattr(cfg, 'eligibility_events', None)
    if eligibility_events_cfg: 
        evs_ee = eligibility_events_run(
            snapshot=prev_snapshot, 
            events=new_events, 
            as_of=as_of, 
            prev_as_of=prev_as_of, 
            cfg=eligibility_events_cfg
        )
        new_events = pd.concat([new_events, *evs_ee], ignore_index=True)

    # Enrollment
    evs = enrollment_run(prev_snapshot, new_events, as_of, getattr(cfg, 'enrollment', None))
    if evs:
        evs_nonempty = [df for df in evs if isinstance(df, pd.DataFrame) and not df.empty]
        if evs_nonempty:
            new_events = pd.concat([new_events, *evs_nonempty], ignore_index=True)
        # else: skip concat if evs_nonempty is empty

    # Contribution Increase
    evs = contrib_increase_run(prev_snapshot, new_events, as_of, getattr(cfg, 'contribution_increase', None))
    if evs:
        evs_nonempty = [df for df in evs if isinstance(df, pd.DataFrame) and not df.empty]
        if evs_nonempty:
            new_events = pd.concat([new_events, *evs_nonempty], ignore_index=True)
        # else: skip concat if evs_nonempty is empty

    # Proactive Decrease
    evs = proactive_decrease_run(prev_snapshot, new_events, as_of, getattr(cfg, 'proactive_decrease', None))
    logger.debug("Raw proactive decrease return: %s", evs)
    if evs and isinstance(evs[0], pd.DataFrame):
        logger.debug("Proactive decrease DataFrame head:\n%s", evs[0].head())
        logger.debug("Event_type values in proactive decrease DataFrame: %s", evs[0]['event_type'].unique())
    if evs:
        evs_nonempty = [df for df in evs if isinstance(df, pd.DataFrame) and not df.empty]
        if evs_nonempty:
            new_events = pd.concat([new_events, *evs_nonempty], ignore_index=True)
    # Check for proactive decrease events after concat
    logger.debug("Events after proactive decrease: %s", new_events[new_events['event_type'] == 'EVT_PROACTIVE_DECREASE'])

    # --- CORE DYNAMICS ---
    # Capture starting headcount (pre-termination) from the previous snapshot
    # Only count ACTIVE employees based on the 'active' column if it exists
    if 'active' in prev_snapshot.columns:
        # Use the 'active' column which was properly set in the runner.py file
        active_employees = prev_snapshot[prev_snapshot['active'] == True]
        start_count = len(active_employees)
        logger.info(f"[RUN_ONE_YEAR YR={year}] Using 'active' column for headcount: {start_count} active employees")
    else:
        # Fall back to using termination dates if 'active' column doesn't exist
        start_count = ((prev_snapshot[EMP_TERM_DATE].isna()) | (prev_snapshot[EMP_TERM_DATE] > as_of)).sum()
        logger.info(f"[RUN_ONE_YEAR YR={year}] Using termination dates for headcount: {start_count} active employees")
    
    # 3. Compensation bumps
    comp_events = comp.bump(prev_snapshot, hazard_slice, as_of)
    
    # 4. Terminations
    # Terminations are determined by the 'term_rate' in hazard_slice, processed by term.run.
    # The configured rates from hazard_slice will be used directly by the term.run engine.
    logger.info(f"[RUN_ONE_YEAR YR={year}] Applying terminations based on configured rates in hazard_slice.")
    term_events = term.run(prev_snapshot, hazard_slice, year_specific_rng, deterministic_term)
    # Filter for actual DataFrame events for logging count
    term_events_dfs = [e for e in term_events if isinstance(e, pd.DataFrame) and not e.empty]
    num_term_events_existing = sum(len(df) for df in term_events_dfs)
    logger.info(f"[RUN_ONE_YEAR YR={year}] Generated {num_term_events_existing} termination events for existing employees.")

    # 5. Update snapshot with comp and term events
    temp_events = comp_events + term_events
    # Flatten and concat to single DataFrame, ensuring empty ones are skipped
    temp_events_list = [df for df in temp_events if isinstance(df, pd.DataFrame) and not df.empty]
    temp_events_df = pd.DataFrame() # Initialize as empty
    if temp_events_list:
        temp_events_df = pd.concat(temp_events_list, ignore_index=True)
        logger.info(f"[RUN_ONE_YEAR YR={year}] Combined {len(temp_events_df)} comp and existing term events.")
    else:
        logger.info(f"[RUN_ONE_YEAR YR={year}] No comp or existing term events to combine.")

    # Determine survivors after comp and term events applied to a temporary snapshot
    current_snapshot_for_survivors = prev_snapshot.copy()
    if 'employee_id' not in current_snapshot_for_survivors.index.name and 'employee_id' in current_snapshot_for_survivors.columns:
        current_snapshot_for_survivors = current_snapshot_for_survivors.set_index('employee_id', drop=False)
    
    updated_snapshot_for_survivors, _ = snapshot.update(current_snapshot_for_survivors, temp_events_df)
    n_survivors = updated_snapshot_for_survivors[
        (updated_snapshot_for_survivors[EMP_TERM_DATE].isna()) |
        (updated_snapshot_for_survivors[EMP_TERM_DATE] > as_of)
    ].shape[0]
    logger.info(f"[RUN_ONE_YEAR YR={year}] After comp and term events (existing employees), n_survivors = {n_survivors}")

    # 6. Count survivors
    survivors = updated_snapshot_for_survivors[
        updated_snapshot_for_survivors[EMP_TERM_DATE].isna()
        | (updated_snapshot_for_survivors[EMP_TERM_DATE] > as_of)
    ]
    n_survivors = survivors.shape[0]
    
    # Calculate how many terminations occurred
    n_terminated = start_count - n_survivors
    logger.info(f"[RUN_ONE_YEAR YR={year}] Starting count: {start_count}, Terminated: {n_terminated}, Survivors: {n_survivors}")
    # 7. Get growth rate from hazard_slice (assume all same for now)
    growth_rate = (
        hazard_slice["growth_rate"].iloc[0] if "growth_rate" in hazard_slice.columns and not hazard_slice.empty else 0.0
    )
    
    # Calculate target based on growth rate
    # The key change: we need to ensure we're growing from the PREVIOUS year's headcount,
    # not just from the survivors after termination
    
    # Capture starting headcount (pre-termination) from the previous snapshot
    # Only count ACTIVE employees based on the 'active' column if it exists
    if 'active' in prev_snapshot.columns:
        # Use the 'active' column which was properly set in the runner.py file
        active_employees = prev_snapshot[prev_snapshot['active'] == True]
        start_count = len(active_employees)
        logger.info(f"[RUN_ONE_YEAR YR={year}] Using 'active' column for headcount: {start_count} active employees")
    else:
        # Fall back to using termination dates if 'active' column doesn't exist
        start_count = prev_snapshot[prev_snapshot[EMP_TERM_DATE].isna() | 
                                   (prev_snapshot[EMP_TERM_DATE] > as_of)].shape[0]
        logger.info(f"[RUN_ONE_YEAR YR={year}] Using termination dates for headcount: {start_count} active employees")
    
    # Calculate target based on starting headcount (not survivors)
    # This ensures we're replacing terminated employees AND adding growth
    target = int(ceil(start_count * (1 + growth_rate)))
    
    # Calculate how many new hires we need to reach the target
    # This is the target minus survivors, which equals:
    # (start_count - n_survivors) + (start_count * growth_rate)
    # = terminations + growth headcount
    hires_needed = max(0, target - n_survivors)
    logger.info(f"[RUN_ONE_YEAR YR={year}] Hires needed (net before new hire term adjustment): {hires_needed}")
    
    # Adjust for new hire termination rate to ensure we hit our target
    new_hire_term_rate_val = hazard_slice['new_hire_termination_rate'].mean() if 'new_hire_termination_rate' in hazard_slice.columns and not hazard_slice.empty else 0.0
    if new_hire_term_rate_val > 0 and hires_needed > 0: # Only adjust if there's a rate and hires are needed
        gross_hires_needed = int(ceil(hires_needed / (1 - new_hire_term_rate_val)))
        logger.info(f"[RUN_ONE_YEAR YR={year}] Adjusting hires from {hires_needed} to {gross_hires_needed} to account for new hire termination rate of {new_hire_term_rate_val:.4f}")
        hires_needed = gross_hires_needed # This becomes the actual number of hires to generate
    else:
        logger.info(f"[RUN_ONE_YEAR YR={year}] No adjustment for new hire termination rate (Rate: {new_hire_term_rate_val:.4f}, Hires Needed: {hires_needed})")

    # 8. Generate Hires
    # The hire.run function should aim to generate 'hires_needed' (which is gross_hires_needed if adjusted)
    hire_events_tuple = hire.run(updated_snapshot_for_survivors, hires_needed, hazard_slice, year_specific_rng, census_template_path) # Ensure hire.run uses the (potentially grossed-up) hires_needed
    
    # Unpack hire_events and hire_comp_events if hire.run returns a tuple
    if isinstance(hire_events_tuple, tuple) and len(hire_events_tuple) == 2:
        hire_events, hire_comp_events = hire_events_tuple
    elif isinstance(hire_events_tuple, list): # Assuming it might return just a list of hire events
        hire_events = hire_events_tuple
        hire_comp_events = [] # No separate comp events from hires in this case
    else:
        logger.error(f"[RUN_ONE_YEAR YR={year}] Unexpected output format from hire.run. Expected tuple or list.")
        hire_events = []
        hire_comp_events = []

    # Filter for actual DataFrame events for logging count
    hire_events_dfs = [e for e in hire_events if isinstance(e, pd.DataFrame) and not e.empty]
    num_hire_events = sum(len(df) for df in hire_events_dfs)
    logger.info(f"[RUN_ONE_YEAR YR={year}] Generated {num_hire_events} hire events (aimed for {hires_needed}).")

    # Combine comp, term (existing), and hire events before applying new hire terminations
    # Ensure proactive_decrease_events is initialized if not generated
    if 'proactive_decrease_events' not in locals():
        proactive_decrease_events = []
        
    events_before_nh_term = temp_events + hire_events + proactive_decrease_events + hire_comp_events # Added hire_comp_events
    events_before_nh_term_list = [df for df in events_before_nh_term if isinstance(df, pd.DataFrame) and not df.empty]
    events_before_nh_term_df = pd.DataFrame()
    if events_before_nh_term_list:
        events_before_nh_term_df = pd.concat(events_before_nh_term_list, ignore_index=True)
        logger.info(f"[RUN_ONE_YEAR YR={year}] Combined {len(events_before_nh_term_df)} events before new hire terminations.")
    else:
        logger.info(f"[RUN_ONE_YEAR YR={year}] No events to combine before new hire terminations.")

    # Apply these events to get snapshot including new hires (before their termination)
    # Use updated_snapshot_for_survivors as the base for applying these events
    snapshot_with_hires, _ = snapshot.update(updated_snapshot_for_survivors, events_before_nh_term_df)
    logger.info(f"[RUN_ONE_YEAR YR={year}] Snapshot updated with hires, active before NH term: {snapshot_with_hires['active'].sum()}")

    # 9. Terminate New Hires
    new_hire_term_events_list = term.run_new_hires(
        snapshot_with_hires,
        hazard_slice,
        year_specific_rng,
        year,
        deterministic_term
    )
    
    # Filter for actual DataFrame events and log count
    new_hire_term_events_dfs = [e for e in new_hire_term_events_list if isinstance(e, pd.DataFrame) and not e.empty]
    num_new_hire_term_events = sum(len(df) for df in new_hire_term_events_dfs)
    logger.info(f"[RUN_ONE_YEAR YR={year}] Generated {num_new_hire_term_events} new hire termination events.")

    new_hire_term_events_df = pd.DataFrame()
    if new_hire_term_events_dfs:
        new_hire_term_events_df = pd.concat(new_hire_term_events_dfs, ignore_index=True)
        logger.info(f"[RUN_ONE_YEAR YR={year}] Combined {len(new_hire_term_events_df)} new hire termination events.")
    else:
        logger.info(f"[RUN_ONE_YEAR YR={year}] No new hire termination events to combine.")

    # Combine all events for the year
    all_event_dfs_for_year = []
    if not events_before_nh_term_df.empty:
        all_event_dfs_for_year.append(events_before_nh_term_df)
    if not new_hire_term_events_df.empty:
        all_event_dfs_for_year.append(new_hire_term_events_df)

    full_event_log_for_year = pd.DataFrame(columns=EVENT_COLS) # Ensure schema if empty
    if all_event_dfs_for_year:
        full_event_log_for_year = pd.concat(all_event_dfs_for_year, ignore_index=True).sort_values(by=['event_date', 'event_type'])
        logger.info(f"[RUN_ONE_YEAR YR={year}] Final event log for year has {len(full_event_log_for_year)} total events.")
    else:
        logger.info(f"[RUN_ONE_YEAR YR={year}] No events recorded for the entire year.")

    # 11. return the events and the **new** snapshot with all events applied
    final_snapshot, _ = snapshot.update(snapshot_with_hires, new_hire_term_events_df)
    return full_event_log_for_year, final_snapshot
