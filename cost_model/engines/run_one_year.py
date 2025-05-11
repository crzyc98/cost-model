# cost_model/engines/run_one_year.py

import json
import logging
from math import ceil
from types import SimpleNamespace
from typing import Tuple, List

import numpy as np
import pandas as pd

from . import comp, term, hire
from cost_model.state.event_log import EVENT_COLS
from cost_model.state import snapshot
from cost_model.utils.columns import (
    EMP_ID,
    EMP_HIRE_DATE,
    EMP_BIRTH_DATE,
    EMP_ROLE,
    EMP_TERM_DATE,
    EMP_GROSS_COMP,
    EMP_DEFERRAL_RATE,
)

from cost_model.plan_rules.eligibility import run as eligibility_run
from cost_model.plan_rules.eligibility_events import run as eligibility_events_run
from cost_model.plan_rules.enrollment import run as enrollment_run
from cost_model.plan_rules.contribution_increase import run as contrib_increase_run
from cost_model.plan_rules.proactive_decrease import run as proactive_decrease_run

logger = logging.getLogger(__name__)

def _dbg(label: str, df: pd.DataFrame, year: int):
    uniq = df[EMP_ID].nunique() if EMP_ID in df.columns else df.index.nunique()
    rows = len(df)
    act = df['active'].sum() if 'active' in df.columns else 'N/A'
    logger.debug(f"[DBG YR={year}] {label:<25} rows={rows:5d} uniq_ids={uniq:5d} act={act}")

def run_one_year(
    event_log: pd.DataFrame,
    prev_snapshot: pd.DataFrame,
    year: int,
    global_params: SimpleNamespace,
    plan_rules: SimpleNamespace,
    hazard_table: pd.DataFrame,
    rng: np.random.Generator,
    census_template_path: str,
    *,
    rng_seed_offset: int = 0,
    deterministic_term: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs a single projection year.
    Returns (full_event_log_for_year, final_snapshot).
    """

    # --- 1. Extract the hazard slice for this year ---
    hazard_slice = hazard_table[hazard_table['simulation_year'] == year]
    if hazard_slice.empty:
        logger.warning(f"[RUN_ONE_YEAR YR={year}] No hazard rates for {year}, using zeros.")
        hazard_slice = pd.DataFrame([{
            'simulation_year': year,
            'term_rate': 0.0,
            'growth_rate': 0.0,
            'comp_raise_pct': 0.0,
            'new_hire_termination_rate': 0.0,
            'cfg': config
        }])

    # Log rates
    log_term_rate = hazard_slice['term_rate'].mean()
    log_growth_rate = hazard_slice['growth_rate'].mean()
    log_comp_rate = hazard_slice['comp_raise_pct'].mean()
    log_nh_term_rate = hazard_slice['new_hire_termination_rate'].mean()
    logger.info(f"[RUN_ONE_YEAR YR={year}] Rates → term={log_term_rate:.3f}, growth={log_growth_rate:.3f}, comp={log_comp_rate:.3f}, nh_term={log_nh_term_rate:.3f}")

    # reproducible RNG per year
    year_rng = np.random.default_rng(rng_seed_offset or rng.bit_generator._seed_seq.entropy)

    as_of = pd.Timestamp(f"{year}-01-01")
    prev_as_of = pd.Timestamp(f"{year-1}-01-01")

    # --- 2. Plan rules engines (eligibility → etc) ---
    all_new_events: List[pd.DataFrame] = []
    cfg = hazard_slice.iloc[0]['cfg']

    # Eligibility
    evs = eligibility_run(prev_snapshot, as_of, getattr(cfg, 'eligibility', None))
    for df in evs or []:
        if not df.empty:
            all_new_events.append(df)
    # Eligibility milestones
    ee_cfg = getattr(cfg, 'eligibility_events', None)
    if ee_cfg:
        ee_evs = eligibility_events_run(prev_snapshot, pd.concat(all_new_events, ignore_index=True), as_of, prev_as_of, ee_cfg)
        for df in ee_evs or []:
            if not df.empty:
                all_new_events.append(df)
    # Enrollment
    evs = enrollment_run(prev_snapshot, pd.concat(all_new_events, ignore_index=True), as_of, getattr(cfg, 'enrollment', None))
    for df in evs or []:
        if not df.empty:
            all_new_events.append(df)
    # Contribution increase
    evs = contrib_increase_run(prev_snapshot, pd.concat(all_new_events, ignore_index=True), as_of, getattr(cfg, 'contribution_increase', None))
    for df in evs or []:
        if not df.empty:
            all_new_events.append(df)
    # Proactive decrease
    evs = proactive_decrease_run(prev_snapshot, pd.concat(all_new_events, ignore_index=True), as_of, getattr(cfg, 'proactive_decrease', None))
    for df in evs or []:
        if not df.empty:
            all_new_events.append(df)

    # Flatten plan-rule events
    plan_rule_events = [df for df in all_new_events if isinstance(df, pd.DataFrame)]
    plan_rule_df = pd.concat(plan_rule_events, ignore_index=True) if plan_rule_events else pd.DataFrame(columns=EVENT_COLS)

    # --- 3. Compensation bump & terminations for existing employees ---
    # Count START-of-year active
    if 'active' in prev_snapshot.columns:
        start_count = int(prev_snapshot['active'].sum())
    else:
        start_count = int(((prev_snapshot[EMP_TERM_DATE].isna()) | (prev_snapshot[EMP_TERM_DATE] > as_of)).sum())
    logger.info(f"[RUN_ONE_YEAR YR={year}] SOY active headcount: {start_count}")

    # 3a. comp bump
    comp_events = comp.bump(prev_snapshot, hazard_slice, as_of, year_rng)
    # 3b. term events
    term_events = term.run(prev_snapshot, hazard_slice, year_rng, deterministic_term)

    # merge comp+term
    core_events = []
    for lst in (comp_events, term_events):
        for df in lst or []:
            if isinstance(df, pd.DataFrame) and not df.empty:
                core_events.append(df)
    core_df = pd.concat(core_events, ignore_index=True) if core_events else pd.DataFrame(columns=EVENT_COLS)

    # 3c. update snapshot with comp+term
    temp_snap = snapshot.update(prev_snapshot.set_index(EMP_ID, drop=False), core_df)
    _dbg("post-term snapshot", temp_snap, year)

    # count survivors
    survivors = temp_snap.loc[
        temp_snap[EMP_TERM_DATE].isna() | (temp_snap[EMP_TERM_DATE] > as_of)
    ]
    n_active_survivors = int(survivors['active'].sum()) if 'active' in survivors.columns else len(survivors)
    n_terminated = start_count - n_active_survivors
    logger.info(f"[RUN_ONE_YEAR YR={year}] Terminations: {n_terminated}, survivors active: {n_active_survivors}")

    # --- 4. Gross-up new-hire rate logic ---
    import math
    # Capture start-of-year headcount
    if 'active' in prev_snapshot.columns:
        start_count = int(prev_snapshot['active'].sum())
    else:
        start_count = int(((prev_snapshot[EMP_TERM_DATE].isna()) | (prev_snapshot[EMP_TERM_DATE] > as_of)).sum())

    nh_rate = getattr(global_params, 'new_hire_rate', 0.0)
    nh_term_rate = getattr(global_params, 'new_hire_termination_rate', 0.0)
    net_hires = int(math.ceil(start_count * nh_rate))
    gross_hires = int(math.ceil(net_hires / (1 - nh_term_rate))) if nh_term_rate < 1.0 else 0

    logger.info(
        f"[RUN_ONE_YEAR YR={year}] start={start_count}, "
        f"net_hires={net_hires} ({nh_rate*100:.1f}%), "
        f"gross_hires={gross_hires} (to allow for {nh_term_rate*100:.1f}% NH term)"
    )

    # --- 5. Generate gross_hires new-hire events + starting comp ---
    hire_tuple = hire.run(temp_snap, gross_hires, hazard_slice, year_rng, census_template_path)
    if isinstance(hire_tuple, tuple) and len(hire_tuple) == 2:
        hire_events, hire_comp_events = hire_tuple
    else:
        hire_events = hire_tuple or []
        hire_comp_events = []

    hire_dfs = [df for df in hire_events + hire_comp_events if isinstance(df, pd.DataFrame) and not df.empty]
    hires_df = pd.concat(hire_dfs, ignore_index=True) if hire_dfs else pd.DataFrame(columns=EVENT_COLS)
    logger.info(f"[RUN_ONE_YEAR YR={year}] Generated {len(hires_df)} new-hire events")

    # --- 6. Apply hires to snapshot (pre NH-term) ---
    before_nh_df = pd.concat([core_df, hires_df], ignore_index=True) if not hires_df.empty else core_df
    snap_with_hires = snapshot.update(temp_snap, before_nh_df)
    _dbg("with hires snapshot", snap_with_hires, year)

    # --- 7. New-hire terminations ---
    nh_term_list = term.run_new_hires(snap_with_hires, hazard_slice, year_rng, year, deterministic_term)
    nh_term_frames = [df for df in (nh_term_list or []) if isinstance(df, pd.DataFrame) and not df.empty]
    if nh_term_frames:
        nh_term_df = pd.concat(nh_term_frames, ignore_index=True)
    else:
        nh_term_df = pd.DataFrame(columns=EVENT_COLS)
    logger.info(f"[RUN_ONE_YEAR YR={year}] New-hire terminations: {len(nh_term_df)}")

    # --- 8. Final event log for the year ---
    all_events = pd.concat([plan_rule_df, core_df, hires_df, nh_term_df], ignore_index=True)
    full_event_log_for_year = all_events.sort_values(['event_time', 'event_type'], ignore_index=True)

    # --- 9. Final snapshot update (apply NH terminations) ---
    final_snapshot = snapshot.update(snap_with_hires, nh_term_df)

    return full_event_log_for_year, final_snapshot