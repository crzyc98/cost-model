# cost_model/engines/run_one_year_engine.py
"""
Orchestration engine for running a complete simulation year, coordinating all workforce dynamics.
QuickStart: see docs/cost_model/engines/run_one_year_engine.md
"""

import json
import logging
from math import ceil
from types import SimpleNamespace
from typing import Tuple, List

import numpy as np
import pandas as pd

from . import term, hire
from .cola import cola
from cost_model.state.event_log import EVENT_COLS, EVT_COMP, EVT_HIRE
from cost_model.state import snapshot
from cost_model.utils.columns import (
    EMP_ID,
    EMP_LEVEL,
    EMP_HIRE_DATE,
    EMP_BIRTH_DATE,
    EMP_ROLE,
    EMP_TERM_DATE,
    EMP_GROSS_COMP,
    EMP_DEFERRAL_RATE,
    EMP_ACTIVE,
    EMP_TENURE_BAND,
    EMP_LEVEL_SOURCE,
    EMP_EXITED
)
REQUIRED_SNAPSHOT_DEFAULTS = {
    EMP_LEVEL: pd.NA,
    EMP_LEVEL_SOURCE: pd.NA,
    EMP_EXITED: False,
    EMP_ACTIVE: True,
    EMP_TENURE_BAND: pd.NA,
}

from cost_model.state.job_levels.sampling import sample_mixed_new_hires
from cost_model.state.event_log import create_event

from cost_model.plan_rules.eligibility import run as eligibility_run
from cost_model.plan_rules.eligibility_events import run as eligibility_events_run
from cost_model.plan_rules.enrollment import run as enrollment_run
from cost_model.plan_rules.contribution_increase import run as contrib_increase_run
from cost_model.plan_rules.proactive_decrease import run as proactive_decrease_run
from .compensation import update_salary
from cost_model.dynamics.compensation import apply_comp_bump
from cost_model.utils.columns import EMP_ID, EMP_GROSS_COMP
logger = logging.getLogger(__name__)

# helper functions below
def _dbg(label: str, df: pd.DataFrame, year: int):
    if df is None:
        logger.error(f"[DBG YR={year}] {label:<25} DataFrame is None!")
        return
    uniq = df[EMP_ID].nunique() if EMP_ID in df.columns else df.index.nunique()
    rows = len(df)
    act = df[EMP_ACTIVE].sum() if EMP_ACTIVE in df.columns else 'N/A'
    logger.debug(f"[DBG YR={year}] {label:<25} rows={rows:5d} uniq_ids={uniq:5d} act={act}")

def _nonempty_frames(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Return only those DataFrames which are non-empty AND not entirely NA.
    """
    good = []
    for df in dfs:
        if df.empty:
            continue
        if df.isna().all().all():
            continue
        good.append(df)
    return good

def ensure_snapshot_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing snapshot columns with safe defaults."""
    for col, default in REQUIRED_SNAPSHOT_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    return df

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

    # --- 0. (NEW) Pre-flight: guarantee incoming snapshot has the full schema
    prev_snapshot = ensure_snapshot_cols(prev_snapshot.copy())
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

    # Eligibility (skip if no config provided)
    elig_cfg = getattr(cfg, 'eligibility', None)
    if elig_cfg is not None:
        evs = eligibility_run(prev_snapshot, as_of, elig_cfg)
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
    # Enrollment (skip if no events or no config)
    enrollment_cfg = getattr(cfg, 'enrollment', None)
    if enrollment_cfg is not None and all_new_events:
        evs = enrollment_run(prev_snapshot, pd.concat(all_new_events, ignore_index=True), as_of, enrollment_cfg)
        for df in evs or []:
            if not df.empty:
                all_new_events.append(df)
    # Contribution increase (skip if no events or no config)
    ci_cfg = getattr(cfg, 'contribution_increase', None)
    if ci_cfg is not None and all_new_events:
        evs = contrib_increase_run(prev_snapshot, pd.concat(all_new_events, ignore_index=True), as_of, ci_cfg)
        for df in evs or []:
            if not df.empty:
                all_new_events.append(df)
    # Proactive decrease (skip if no events or no config)
    pd_cfg = getattr(cfg, 'proactive_decrease', None)
    if pd_cfg is not None and all_new_events:
        evs = proactive_decrease_run(prev_snapshot, pd.concat(all_new_events, ignore_index=True), as_of, pd_cfg)
        for df in evs or []:
            if not df.empty:
                all_new_events.append(df)

    # Flatten plan-rule events
    plan_rule_events = [df for df in all_new_events if isinstance(df, pd.DataFrame)]
    plan_rule_events = [df for df in plan_rule_events if not df.empty and not df.isna().all().all()]
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
    if EMP_ACTIVE in prev_snapshot.columns:
        start_count = int(prev_snapshot[EMP_ACTIVE].loc[mask_exp].sum())
    else:
        start_count = int(mask_exp.sum())
    n0_exp = int(mask_exp.sum())  # For net-growth calculation
    logger.info(f"[YR={year}] SOY Experienced Active = {start_count}")

    # 3a. comp bump (COLA → promo → merit)
    # update_salary mutates EMP_GROSS_COMP and returns EVT_RAISE events
    comp_events = [update_salary(
        prev_snapshot,
        params=global_params.compensation,
        rng=year_rng
    )]
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
    core_events = [df for df in core_events if not df.empty and not df.isna().all().all()]
    core_df = pd.concat(core_events, ignore_index=True) if core_events else pd.DataFrame(columns=EVENT_COLS)

    # 3c. update snapshot with comp+term
    temp_snap = snapshot.update(ensure_snapshot_cols(prev_snapshot).set_index(EMP_ID, drop=False), core_df, year)
    # Integrity check: log unique EMP_IDs and assert uniqueness
    n_unique_empids = temp_snap[EMP_ID].nunique() if EMP_ID in temp_snap.columns else temp_snap.index.nunique()
    n_rows = len(temp_snap)
    logger.info(f"[YR={year}] Post-term snapshot: {n_unique_empids} unique EMP_IDs, {n_rows} rows")
    if n_unique_empids != n_rows:
        logger.error(f"[YR={year}] Duplicate EMP_IDs detected in post-term snapshot!")
        raise ValueError("Duplicate EMP_IDs in post-term snapshot!")
    _dbg("post-term snapshot", temp_snap, year)
    # 2) After comp+term but BEFORE hires
    n_active_survivors = int(temp_snap[EMP_ACTIVE].sum()) if EMP_ACTIVE in temp_snap.columns else len(temp_snap)
    logger.info(f"[YR={year}] Post-term Active= {n_active_survivors}")

    # count survivors
    survivors = temp_snap.loc[
        temp_snap[EMP_TERM_DATE].isna() | (temp_snap[EMP_TERM_DATE] > as_of)
    ]
    n_active_survivors = int(survivors[EMP_ACTIVE].sum()) if EMP_ACTIVE in survivors.columns else len(survivors)
    n_terminated = start_count - n_active_survivors
    logger.info(f"[RUN_ONE_YEAR YR={year}] Terminations: {n_terminated}, survivors active: {n_active_survivors}")

    # --- 4. Gross-up new-hire rate logic ---
    import math
    # Capture start-of-year headcount
    if EMP_ACTIVE in prev_snapshot.columns:
        start_count = int(prev_snapshot[EMP_ACTIVE].sum())
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

    # --- 5. Generate gross_hires new-hire events ---
    hire_events = hire.run(
        temp_snap,
        gross_hires,
        hazard_slice,
        year_rng,
        census_template_path,
        global_params,
        terminated_events=pd.concat(term_events, ignore_index=True) if term_events is not None and len(term_events) > 0 else pd.DataFrame()
    ) or []

    # --- 5b. Sample compensation for hires using band-aware sampler ---
    if gross_hires > 0:
        # derive distribution of levels from previous snapshot
        curr_dist = prev_snapshot[EMP_LEVEL].value_counts(normalize=True).to_dict()
        level_counts = {lvl: int(dist * gross_hires) for lvl, dist in curr_dist.items()}
        assigned = sum(level_counts.values())
        if assigned < gross_hires and level_counts:
            level_counts[min(level_counts.keys())] += (gross_hires - assigned)
        # sample new hires with levels & compensation
        new_hires = sample_mixed_new_hires(level_counts, age_range=(25, 55), random_state=year_rng)
        new_hires = new_hires.rename(columns={'level_id': EMP_LEVEL, 'compensation': EMP_GROSS_COMP})
        new_hires['job_level_source'] = 'hire'
        # align hire IDs from hire events
        hire_events_df = pd.concat(hire_events, ignore_index=True) if hire_events else pd.DataFrame(columns=EVENT_COLS)
        new_hires[EMP_ID] = hire_events_df[EMP_ID].reset_index(drop=True)
        # build compensation events - ensure values are properly set with no empty rows
        comp_events = []
        logger.info(f"[YR={year}] Processing {len(new_hires)} new hires for compensation events")
        
        for idx, r in new_hires.iterrows():
            # Get EMP_ID and ensure it's valid
            emp_id = r.get(EMP_ID)
            if pd.isna(emp_id) or emp_id == "":
                logger.warning(f"[YR={year}] Row {idx}: Skipping EVT_COMP creation - missing employee ID")
                continue
                
            # Extract compensation and validate it
            # First make sure the column exists
            if EMP_GROSS_COMP not in r:
                logger.warning(f"[YR={year}] Row {idx}: Employee {emp_id} missing EMP_GROSS_COMP column")
                # Try to get salary from other fields if possible
                if 'compensation' in r:
                    comp_value = r['compensation']
                    logger.info(f"[YR={year}] Using 'compensation' column instead for {emp_id}: {comp_value}")
                else:
                    logger.error(f"[YR={year}] No compensation found for {emp_id} - cannot create EVT_COMP")
                    continue
            else:
                comp_value = r[EMP_GROSS_COMP]
            
            # Handle NaN values
            if pd.isna(comp_value):
                logger.warning(f"[YR={year}] NaN compensation for {emp_id} - trying to fix")
                # Try to get compensation another way - use average compensation for their role
                role = r.get('employee_role', None)
                if role is not None and not pd.isna(role):
                    role_avg_comp = new_hires.loc[~pd.isna(new_hires[EMP_GROSS_COMP]) & 
                                                 (new_hires['employee_role'] == role), 
                                                 EMP_GROSS_COMP].mean()
                    if not pd.isna(role_avg_comp) and role_avg_comp > 0:
                        comp_value = role_avg_comp
                        logger.info(f"[YR={year}] Using role average comp for {emp_id}: {comp_value}")
                    else:
                        # If no average available, use a default value
                        comp_value = 75000.0  # Default compensation as fallback
                        logger.warning(f"[YR={year}] Using default comp value for {emp_id}: {comp_value}")
                else:
                    # If no role info, use a default value
                    comp_value = 75000.0  # Default compensation as fallback
                    logger.warning(f"[YR={year}] Using default comp value for {emp_id}: {comp_value}")
            
            # Try to convert to float and validate
            try:
                comp_value = float(comp_value)
                if comp_value <= 0:
                    logger.warning(f"[YR={year}] Non-positive compensation ({comp_value}) for {emp_id} - using default")
                    comp_value = 75000.0
            except (ValueError, TypeError):
                logger.warning(f"[YR={year}] Invalid compensation type for {emp_id} - using default")
                comp_value = 75000.0
                
            # Create the compensation event
            comp_events.append(create_event(
                event_time=as_of,
                employee_id=emp_id,
                event_type=EVT_COMP,
                value_num=comp_value,  # Ensure this is a valid float
                meta="New-hire starting compensation"
            ))
            
        # Only create DataFrame if we have valid events
        if comp_events:
            logger.info(f"[YR={year}] Created {len(comp_events)} compensation events for new hires")
            hire_comp_events = [pd.DataFrame(comp_events)]
        else:
            logger.warning(f"[YR={year}] No valid compensation events created for new hires")
            hire_comp_events = []
    else:
        hire_comp_events = []

    hire_dfs = [df for df in hire_events + hire_comp_events if isinstance(df, pd.DataFrame)]
    # filter non-empty/all-NA frames
    parts_hires = _nonempty_frames(hire_dfs)
    hires_df = pd.concat(parts_hires, ignore_index=True) if parts_hires else pd.DataFrame(columns=EVENT_COLS)
    logger.info(f"[RUN_ONE_YEAR YR={year}] Generated {len(hires_df)} new-hire events")

    # --- 6. Apply hires to snapshot (pre NH-term) ---
    # --- before new hires snapshot merge ---
    parts = _nonempty_frames([core_df, hires_df])
    before_nh_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=EVENT_COLS)
    snap_with_hires = snapshot.update(ensure_snapshot_cols(temp_snap), before_nh_df, year)
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
    logger.info(f"[YR={year}] After Hires     = {snap_with_hires[EMP_ACTIVE].sum()}  (added {hires_added} hires)")

    # --- 6b. Apply annual comp bump only to experienced employees ---
    # 1) Identify all new-hire IDs so we can exclude them
    new_hire_ids = set(hires_df[EMP_ID]) if not hires_df.empty else set()

    # 2) Build mask of those present at year-start (i.e. not new hires)
    experienced_mask = ~snap_with_hires[EMP_ID].isin(new_hire_ids)
    experienced = snap_with_hires.loc[experienced_mask]
    # Debug: check experienced slice before bump
    logger.debug(
        f"[YR={year}] Experienced slice: {len(experienced)} rows; sample IDs: {experienced[EMP_ID].tolist()[:5]}"
    )

    # 3) Determine bump rate from hazard table
    rate = float(hazard_slice['comp_raise_pct'].iloc[0])

    # 4) Run the bump on experienced headcount
    bumped_experienced = apply_comp_bump(
        experienced,
        EMP_GROSS_COMP,
        rate,
        year_rng,
        logger
    )

    # 5) Merge their updated salaries back into the full snapshot
    snap_with_hires.loc[experienced_mask, EMP_GROSS_COMP] = bumped_experienced[EMP_GROSS_COMP]

    # 6) Create and append EVT_COMP events for the compensation bump
    comp_events = []
    # Get original vs updated compensation values
    before_comps = experienced[EMP_GROSS_COMP].to_dict()
    after_comps = bumped_experienced[EMP_GROSS_COMP].to_dict()
    
    # Create event logs for experienced employees
    for emp_id, after_comp in after_comps.items():
        before_comp = before_comps.get(emp_id, 0)
        if pd.isna(before_comp):
            before_comp = 0.0
            logger.warning(f"[YR={year}] Fixed NA before_comp value for {emp_id}: using 0.0")
        if pd.isna(after_comp):
            # Skip if after_comp is still NA
            logger.warning(f"[YR={year}] Skipping - NA after_comp value for {emp_id}")
            continue
            
        # Calculate absolute delta (always log the comp event regardless of size)
        delta = float(after_comp - before_comp)
        comp_events.append(create_event(
            pd.Timestamp(f"{year}-01-01"),
            emp_id,
            EVT_COMP,
            value_num=delta,  # Even if delta is very small, still log it
            meta=f"Annual compensation increase of {rate:.1%}"
        ))
    
    # Add the comp events to the event log
    if comp_events:
        logger.info(f"[YR={year}] Adding {len(comp_events)} compensation bump events to log")
        comp_events_df = pd.DataFrame(comp_events)
        event_log = pd.concat([event_log, comp_events_df], ignore_index=True)

    # --- 7. New-hire terminations ---
    nh_term_list = term.run_new_hires(snap_with_hires, hazard_slice, year_rng, year, deterministic_term)
    nh_term_frames = [df for df in (nh_term_list or []) if isinstance(df, pd.DataFrame) and not df.empty]
    nh_term_frames = [df for df in nh_term_frames if not df.empty and not df.isna().all().all()]
    nh_term_df = pd.concat(nh_term_frames, ignore_index=True) if nh_term_frames else pd.DataFrame(columns=EVENT_COLS)
    logger.info(f"[RUN_ONE_YEAR YR={year}] New-hire terminations: {len(nh_term_df)}")
    # 4) After new-hire terminations
    nh_terms_removed = len(nh_term_df) if not nh_term_df.empty else 0

    # --- 8. Final event log for the year ---
    all_bits = [plan_rule_df, core_df, hires_df, nh_term_df]
    all_bits = [df for df in all_bits if not df.empty and not df.isna().all().all()]
    full_event_log_for_year = pd.concat(all_bits, ignore_index=True) if all_bits else pd.DataFrame(columns=EVENT_COLS)
    full_event_log_for_year = full_event_log_for_year.sort_values(['event_time', 'event_type'], ignore_index=True)

    # --- 9. Final snapshot update (apply NH terminations) ---
    final_snapshot = snapshot.update(ensure_snapshot_cols(snap_with_hires), nh_term_df, year)
    # Integrity check: log unique EMP_IDs and assert uniqueness
    n_unique_empids_final = final_snapshot[EMP_ID].nunique() if EMP_ID in final_snapshot.columns else final_snapshot.index.nunique()
    n_rows_final = len(final_snapshot)
    logger.info(f"[YR={year}] Final snapshot: {n_unique_empids_final} unique EMP_IDs, {n_rows_final} rows")
    if n_unique_empids_final != n_rows_final:
        logger.error(f"[YR={year}] Duplicate EMP_IDs detected in final snapshot!")
        raise ValueError("Duplicate EMP_IDs in final snapshot!")
    logger.info(f"[YR={year}] Post-NH-term    = {final_snapshot[EMP_ACTIVE].sum()}  (removed {nh_terms_removed} NH terms)")
    # Propagate job_level_source for new hires
    if 'job_level_source' in final_snapshot.columns:
        # Add 'hire' category if column is categorical
        if pd.api.types.is_categorical_dtype(final_snapshot['job_level_source']):
            final_snapshot['job_level_source'] = final_snapshot['job_level_source'].cat.add_categories(['hire'])
        final_snapshot['job_level_source'] = final_snapshot['job_level_source'].fillna('hire')
    # Propagate employee levels for existing and new hires
    if EMP_LEVEL not in final_snapshot.columns:
        # map levels from original snapshot
        level_map = prev_snapshot.set_index(EMP_ID)[EMP_LEVEL].to_dict()
        # include new hire levels if any
        if gross_hires > 0:
            level_map.update(new_hires.set_index(EMP_ID)[EMP_LEVEL].to_dict())
        final_snapshot[EMP_LEVEL] = final_snapshot[EMP_ID].map(level_map)
    return full_event_log_for_year, final_snapshot