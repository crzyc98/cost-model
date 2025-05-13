# cost_model/engines/run_one_year.py
"""
Orchestration engine for running a complete simulation year, coordinating all workforce dynamics.
QuickStart: see docs/cost_model/engines/run_one_year.md
"""

import json
import logging
from math import ceil
from types import SimpleNamespace
from typing import Tuple, List

import numpy as np
import pandas as pd

from . import comp, term, hire
from .cola import cola
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
    if df is None:
        logger.error(f"[DBG YR={year}] {label:<25} DataFrame is None!")
        return
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

    # —————————————————————————————————————————————
    # Ensure hazard_table is loaded with all required columns
    logger.debug(f"Hazard‐table columns before rename: {hazard_table.columns.tolist()}")
    if 'simulation_year' not in hazard_table.columns:
        if 'year' in hazard_table.columns:
            hazard_table = hazard_table.rename(columns={'year': 'simulation_year'})
            logger.debug("Renamed hazard_table.year → hazard_table.simulation_year")
        else:
            raise KeyError(
                "Your hazard_table is missing both 'simulation_year' and 'year' columns; "
                "cannot project term/comp rates. Columns present: "
                f"{hazard_table.columns.tolist()}"
            )
    logger.debug(f"Hazard‐table columns after  rename: {hazard_table.columns.tolist()}")

    # Standardize the role column using EMP_ROLE from columns.py
    from cost_model.utils.columns import EMP_ROLE
    if 'role' not in hazard_table.columns:
        if EMP_ROLE in hazard_table.columns:
            hazard_table = hazard_table.rename(columns={EMP_ROLE: 'role'})
            logger.debug(f"Renamed hazard_table.{EMP_ROLE} → hazard_table.role")
        else:
            raise KeyError(
                "Your hazard_table is missing both 'role' and EMP_ROLE columns; "
                f"columns present: {hazard_table.columns.tolist()}"
            )
    # Check for all required columns
    expected = {
        'simulation_year',
        'role',
        'tenure_band',
        'term_rate',
        'comp_raise_pct',
        'new_hire_termination_rate',
        'cola_pct',
        'cfg'
    }
    missing = expected - set(hazard_table.columns)
    if missing:
        raise KeyError(f"Hazard table is missing required columns: {missing}. Columns present: {hazard_table.columns.tolist()}")
    # —————————————————————————————————————————————

    # --- 1. Extract the hazard slice for this year ---
    hazard_slice = hazard_table[hazard_table['simulation_year'] == year]
    # Log unique roles and tenure_bands in hazard_slice and prev_snapshot for this year
    from cost_model.utils.columns import EMP_ROLE
    unique_hazard_roles = hazard_slice['role'].unique().tolist() if 'role' in hazard_slice.columns else []
    unique_hazard_tenure = hazard_slice['tenure_band'].unique().tolist() if 'tenure_band' in hazard_slice.columns else []
    unique_snap_roles = prev_snapshot[EMP_ROLE].unique().tolist() if EMP_ROLE in prev_snapshot.columns else []
    unique_snap_tenure = prev_snapshot['tenure_band'].unique().tolist() if 'tenure_band' in prev_snapshot.columns else []
    logger.info(f"[RUN_ONE_YEAR YR={year}] hazard_slice roles: {unique_hazard_roles}")
    logger.info(f"[RUN_ONE_YEAR YR={year}] hazard_slice tenure_bands: {unique_hazard_tenure}")
    logger.info(f"[RUN_ONE_YEAR YR={year}] snapshot roles: {unique_snap_roles}")
    logger.info(f"[RUN_ONE_YEAR YR={year}] snapshot tenure_bands: {unique_snap_tenure}")
    if hazard_slice.empty:
        logger.warning(f"[RUN_ONE_YEAR YR={year}] ⚠️ hazard_slice is EMPTY after filtering. "
                       f"Available years in table: {sorted(hazard_table['simulation_year'].unique())}")
        hazard_slice = pd.DataFrame([{
            'simulation_year': year,
            'term_rate': 0.0,
            'comp_raise_pct': 0.0,
            'new_hire_termination_rate': 0.0,
            'cfg': config
        }])

    # Log rates
    log_term_rate = hazard_slice['term_rate'].mean()
    log_comp_rate = hazard_slice['comp_raise_pct'].mean()
    log_nh_term_rate = hazard_slice['new_hire_termination_rate'].mean()
    logger.info(f"[RUN_ONE_YEAR YR={year}] Rates → term={log_term_rate:.3f}, comp={log_comp_rate:.3f}, nh_term={log_nh_term_rate:.3f}")

    # reproducible RNG per year
    year_rng = np.random.default_rng(rng_seed_offset or rng.bit_generator._seed_seq.entropy)

    as_of = pd.Timestamp(f"{year}-01-01")
    prev_as_of = pd.Timestamp(f"{year-1}-01-01")

    # --- 1b. Initialize event accumulator ---
    all_new_events: List[pd.DataFrame] = []

    # --- 1c. COLA: Cost-of-Living Adjustment events ---
    days_into_year_for_cola = getattr(global_params, 'days_into_year_for_cola', 0)
    cola_jitter_days = getattr(global_params, 'cola_jitter_days', 0)
    cola_df, = cola(prev_snapshot, hazard_slice, as_of, days_into_year=days_into_year_for_cola, jitter_days=cola_jitter_days, rng=year_rng)
    if not cola_df.empty:
        all_new_events.append(cola_df)

    # --- 1d. Promotion and Raise Events ---
    from .promotion import promote
    days_into_year_for_promotion = getattr(global_params, 'days_into_year_for_promotion', 0)
    promo_time = as_of + pd.Timedelta(days=days_into_year_for_promotion)
    # Optionally, you could stagger raise_time (e.g., 30 days after promotion)
    raise_time = promo_time  # or promo_time + pd.Timedelta(days=30)
    promotion_rules = getattr(global_params, 'promotion_rules', {}) if hasattr(global_params, 'promotion_rules') else {}
    promotions_df, raises_df = promote(prev_snapshot, promotion_rules, promo_time, raise_time=raise_time)
    if not promotions_df.empty:
        all_new_events.append(promotions_df)
    if not raises_df.empty:
        all_new_events.append(raises_df)

    # --- 2. Plan rules engines (eligibility → etc) ---
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
    # Ensure EMP_HIRE_DATE is datetime
    prev_snapshot[EMP_HIRE_DATE] = pd.to_datetime(prev_snapshot[EMP_HIRE_DATE], errors='coerce')
    # Count START-of-year experienced active (exclude any hire this year)
    # Experienced = hired before Jan 1 of this year
    as_of = pd.Timestamp(f"{year}-01-01")
    mask_exp = (
        ((prev_snapshot[EMP_TERM_DATE].isna()) | (prev_snapshot[EMP_TERM_DATE] > as_of))
        & (prev_snapshot[EMP_HIRE_DATE] < as_of)
    )
    mask_new_hire = (
        ((prev_snapshot[EMP_TERM_DATE].isna()) | (prev_snapshot[EMP_TERM_DATE] > as_of))
        & (prev_snapshot[EMP_HIRE_DATE] >= as_of)
    )
    logger.info(f"[YR={year}] SOY experienced mask: {mask_exp.sum()} | new hire mask: {mask_new_hire.sum()} | total: {len(prev_snapshot)}")
    if 'active' in prev_snapshot.columns:
        start_count = int(prev_snapshot['active'].loc[mask_exp].sum())
    else:
        start_count = int(mask_exp.sum())
    n0_exp = int(mask_exp.sum())  # For net-growth calculation
    logger.info(f"[YR={year}] SOY Experienced Active = {start_count}")

    # 3a. comp bump
    comp_events = comp.bump(prev_snapshot, hazard_slice, as_of, year_rng)
    # 3b. term events (EXPERIENCED ONLY: exclude new hires)
    term_events = term.run(prev_snapshot.loc[mask_exp], hazard_slice, year_rng, deterministic_term)
    # Integrity check: log number and EMP_IDs of term events
    n_term_events = len(term_events[0]) if term_events and len(term_events) > 0 else 0
    logger.info(f"[YR={year}] Term events generated: {n_term_events}")
    if term_events and len(term_events) > 0:
        logger.info(f"[YR={year}] Term EMP_IDs: {term_events[0][EMP_ID].tolist()}")

    # merge comp+term
    core_events = []
    for lst in (comp_events, term_events):
        for df in lst or []:
            if isinstance(df, pd.DataFrame) and not df.empty:
                core_events.append(df)
    core_df = pd.concat(core_events, ignore_index=True) if core_events else pd.DataFrame(columns=EVENT_COLS)

    # 3c. update snapshot with comp+term
    temp_snap = snapshot.update(prev_snapshot.set_index(EMP_ID, drop=False), core_df, year)
    # Integrity check: log unique EMP_IDs and assert uniqueness
    n_unique_empids = temp_snap[EMP_ID].nunique() if EMP_ID in temp_snap.columns else temp_snap.index.nunique()
    n_rows = len(temp_snap)
    logger.info(f"[YR={year}] Post-term snapshot: {n_unique_empids} unique EMP_IDs, {n_rows} rows")
    if n_unique_empids != n_rows:
        logger.error(f"[YR={year}] Duplicate EMP_IDs detected in post-term snapshot!")
        raise ValueError("Duplicate EMP_IDs in post-term snapshot!")
    _dbg("post-term snapshot", temp_snap, year)
    # 2) After comp+term but BEFORE hires
    n_active_survivors = int(temp_snap['active'].sum()) if 'active' in temp_snap.columns else len(temp_snap)
    logger.info(f"[YR={year}] Post-term Active= {n_active_survivors}")

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

    # Try to get new_hire_rate from different possible locations in the config
    nh_rate = 0.0
    # 1. Try from global_params.new_hires.new_hire_rate (nested structure)
    if hasattr(global_params, 'new_hires') and hasattr(global_params.new_hires, 'new_hire_rate'):
        nh_rate = global_params.new_hires.new_hire_rate
        logger.info(f"[RUN_ONE_YEAR YR={year}] Using new_hire_rate={nh_rate} from global_params.new_hires.new_hire_rate")
    # 2. Fall back to direct attribute on global_params
    else:
        nh_rate = getattr(global_params, 'new_hire_rate', 0.0)
        logger.info(f"[RUN_ONE_YEAR YR={year}] Using new_hire_rate={nh_rate} from global_params.new_hire_rate")
    
    # Similarly for new_hire_termination_rate
    nh_term_rate = 0.0
    # 1. Try from global_params.attrition.new_hire_termination_rate (nested structure)
    if hasattr(global_params, 'attrition') and hasattr(global_params.attrition, 'new_hire_termination_rate'):
        nh_term_rate = global_params.attrition.new_hire_termination_rate
        logger.info(f"[RUN_ONE_YEAR YR={year}] Using new_hire_termination_rate={nh_term_rate} from global_params.attrition.new_hire_termination_rate")
    # 2. Fall back to direct attribute on global_params
    else:
        nh_term_rate = getattr(global_params, 'new_hire_termination_rate', 0.0)
        logger.info(f"[RUN_ONE_YEAR YR={year}] Using new_hire_termination_rate={nh_term_rate} from global_params.new_hire_termination_rate")
    
    net_hires = int(math.ceil(start_count * nh_rate))
    gross_hires = int(math.ceil(net_hires / (1 - nh_term_rate))) if nh_term_rate < 1.0 else 0

    logger.info(
        f"[RUN_ONE_YEAR YR={year}] start={start_count}, "
        f"net_hires={net_hires} ({nh_rate*100:.1f}%), "
        f"gross_hires={gross_hires} (to allow for {nh_term_rate*100:.1f}% NH term)"
    )

    # --- 5. Generate gross_hires new-hire events + starting comp ---
    # pass the termination pool into hire.run
    terminated_events = pd.concat(term_events, ignore_index=True) if term_events is not None and len(term_events) > 0 else pd.DataFrame()
    hire_tuple = hire.run(
        temp_snap,
        gross_hires,
        hazard_slice,
        year_rng,
        census_template_path,
        global_params,
        terminated_events=terminated_events,
    )
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
    snap_with_hires = snapshot.update(temp_snap, before_nh_df, year)
    # Integrity check: log unique EMP_IDs and assert uniqueness
    n_unique_empids_hires = snap_with_hires[EMP_ID].nunique() if EMP_ID in snap_with_hires.columns else snap_with_hires.index.nunique()
    n_rows_hires = len(snap_with_hires)
    logger.info(f"[YR={year}] With hires snapshot: {n_unique_empids_hires} unique EMP_IDs, {n_rows_hires} rows")
    if n_unique_empids_hires != n_rows_hires:
        logger.error(f"[YR={year}] Duplicate EMP_IDs detected in with-hires snapshot!")
        raise ValueError("Duplicate EMP_IDs in with-hires snapshot!")
    _dbg("with hires snapshot", snap_with_hires, year)
    # 3) After applying new hires (but BEFORE NH-termination)
    hires_added = len(hires_df) if not hires_df.empty else 0
    logger.info(f"[YR={year}] After Hires     = {snap_with_hires['active'].sum()}  (added {hires_added} hires)")

    # --- 7. New-hire terminations ---
    nh_term_list = term.run_new_hires(snap_with_hires, hazard_slice, year_rng, year, deterministic_term)
    nh_term_frames = [df for df in (nh_term_list or []) if isinstance(df, pd.DataFrame) and not df.empty]
    if nh_term_frames:
        nh_term_df = pd.concat(nh_term_frames, ignore_index=True)
    else:
        nh_term_df = pd.DataFrame(columns=EVENT_COLS)
    logger.info(f"[RUN_ONE_YEAR YR={year}] New-hire terminations: {len(nh_term_df)}")
    # 4) After new-hire terminations
    nh_terms_removed = len(nh_term_df) if not nh_term_df.empty else 0

    # --- 8. Final event log for the year ---
    all_events = pd.concat([plan_rule_df, core_df, hires_df, nh_term_df], ignore_index=True)
    full_event_log_for_year = all_events.sort_values(['event_time', 'event_type'], ignore_index=True)

    # --- 9. Final snapshot update (apply NH terminations) ---
    final_snapshot = snapshot.update(snap_with_hires, nh_term_df, year)
    # Integrity check: log unique EMP_IDs and assert uniqueness
    n_unique_empids_final = final_snapshot[EMP_ID].nunique() if EMP_ID in final_snapshot.columns else final_snapshot.index.nunique()
    n_rows_final = len(final_snapshot)
    logger.info(f"[YR={year}] Final snapshot: {n_unique_empids_final} unique EMP_IDs, {n_rows_final} rows")
    if n_unique_empids_final != n_rows_final:
        logger.error(f"[YR={year}] Duplicate EMP_IDs detected in final snapshot!")
        raise ValueError("Duplicate EMP_IDs in final snapshot!")
    logger.info(f"[YR={year}] Post-NH-term    = {final_snapshot['active'].sum()}  (removed {nh_terms_removed} NH terms)")

    return full_event_log_for_year, final_snapshot