# cost_model/engines/run_one_year.py

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from math import ceil

from . import comp, term, hire
from cost_model.state.event_log import EVENT_COLS
from cost_model.state import snapshot
from cost_model.utils.columns import EMP_TERM_DATE

from cost_model.plan_rules.eligibility import run as eligibility_run
from cost_model.plan_rules.eligibility_events import run as eligibility_events_run
from cost_model.plan_rules.enrollment import run as enrollment_run
from cost_model.plan_rules.contribution_increase import run as contrib_increase_run
from cost_model.plan_rules.proactive_decrease import run as proactive_decrease_run


def run_one_year(
    year: int,
    prev_snapshot: pd.DataFrame,
    event_log: pd.DataFrame,
    hazard_table: pd.DataFrame,
    rng: np.random.Generator,
    deterministic_term: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    1. extract Jan 1 snapshot (prev_snapshot)
    2. hazard_slice = hazard_table[hazard_table.year == year]
    3. comp_events  = comp.bump(prev_snapshot, hazard_slice, as_of=Jan1)
    4. term_events  = term.run(prev_snapshot, hazard_slice, rng, deterministic_term)
    5. temporary_snapshot = snapshot.update(prev_snapshot, comp_events+term_events)
    6. survivors = count active in temporary_snapshot
    7. target = ceil(active * (1+growth_rate_from_cfg))
    8. hire_events, hire_comp_events = hire.run(temporary_snapshot, target, hazard_slice, rng)
    9. all_new_events = comp_events + term_events + hire_events + hire_comp_events
    10. updated_snapshot = snapshot.update(temporary_snapshot, all_new_events)
    11. return updated_snapshot, event_log.append(all_new_events)
    """
    # 1. prev_snapshot is already Jan 1
    # 2. hazard_slice for this year
    hazard_slice = hazard_table[hazard_table["year"] == year]
    as_of = pd.Timestamp(f"{year}-01-01")

    # --- PHASE 4 PLAN RULE ENGINES ---
    events = event_log.copy()
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
    print("[DEBUG] Raw proactive decrease return:", evs)
    if evs and isinstance(evs[0], pd.DataFrame):
        print("[DEBUG] proactive decrease DataFrame head:\n", evs[0].head())
        print("[DEBUG] event_type values in proactive decrease DataFrame:", evs[0]['event_type'].unique())
    if evs:
        evs_nonempty = [df for df in evs if isinstance(df, pd.DataFrame) and not df.empty]
        if evs_nonempty:
            new_events = pd.concat([new_events, *evs_nonempty], ignore_index=True)
    # DEBUG: Check for proactive decrease events after concat
    print("[DEBUG] Events after proactive decrease:", new_events[new_events['event_type'] == 'EVT_PROACTIVE_DECREASE'])

    # --- CORE DYNAMICS ---
    # 3. Compensation bumps
    comp_events = comp.bump(prev_snapshot, hazard_slice, as_of)
    # 4. Terminations
    term_events = term.run(prev_snapshot, hazard_slice, rng, deterministic_term)
    # 5. Update snapshot with comp and term events
    temp_events = comp_events + term_events
    # Flatten and concat to single DataFrame
    temp_events_list = [df for df in temp_events if isinstance(df, pd.DataFrame) and not df.empty]
    if temp_events_list:
        temp_events_df = pd.concat(temp_events_list, ignore_index=True)
    else:
        temp_events_df = pd.DataFrame([], columns=EVENT_COLS)
    temporary_snapshot = snapshot.update(prev_snapshot, temp_events_df)
    # 6. Count survivors
    survivors = temporary_snapshot[
        temporary_snapshot[EMP_TERM_DATE].isna()
        | (temporary_snapshot[EMP_TERM_DATE] > as_of)
    ]
    n_survivors = survivors.shape[0]
    # 7. Get growth rate from hazard_slice (assume all same for now)
    growth_rate = (
        hazard_slice["growth_rate"].iloc[0] if "growth_rate" in hazard_slice else 0.0
    )
    target = int(ceil(n_survivors * (1 + growth_rate)))
    # 8. Hires
    hire_out = hire.run(temporary_snapshot, target, hazard_slice, rng)
    if isinstance(hire_out, list) and len(hire_out) == 2:
        hire_events, hire_comp_events = hire_out
    else:
        hire_events, hire_comp_events = hire_out, pd.DataFrame([], columns=EVENT_COLS)
    # 9. All new events
    all_new_events = comp_events + term_events + [hire_events, hire_comp_events]
    # Flatten and filter empty
    all_new_events = [
        df for df in all_new_events if isinstance(df, pd.DataFrame) and not df.empty
    ]
    if all_new_events:
        all_new_events_df = pd.concat(all_new_events, ignore_index=True)
    else:
        all_new_events_df = pd.DataFrame([], columns=EVENT_COLS)
    # 10. Update snapshot with all new events
    # Flatten and concat hire events for update
    hire_event_dfs = [df for df in [hire_events, hire_comp_events] if isinstance(df, pd.DataFrame) and not df.empty]
    if hire_event_dfs:
        hire_events_df = pd.concat(hire_event_dfs, ignore_index=True)
    else:
        hire_events_df = pd.DataFrame([], columns=EVENT_COLS)
    updated_snapshot = snapshot.update(temporary_snapshot, hire_events_df)
    # 11. Return updated snapshot and full event log (including all plan-rule events)
    full_event_log = pd.concat(
        [events, new_events, all_new_events_df],
        ignore_index=True,
    )
    # DEBUG: Check for proactive decrease events in full_event_log
    print("[DEBUG] Events in full_event_log:", full_event_log[full_event_log['event_type'] == 'EVT_PROACTIVE_DECREASE'])
    return updated_snapshot, full_event_log
