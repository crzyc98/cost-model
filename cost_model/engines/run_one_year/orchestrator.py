"""
Orchestrator module for run_one_year package.

Coordinates the execution of all simulation steps for a single year.
"""
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np

from cost_model.engines import hire
from cost_model.state.snapshot import update as snapshot_update
from cost_model.state.schema import EMP_ID

# Import submodules
from .validation import ensure_snapshot_cols, validate_and_extract_hazard_slice, validate_eoy_snapshot
from .utils import compute_headcount_targets, dbg
from cost_model.engines.markov_promotion import apply_markov_promotions
from cost_model.engines import term
from cost_model.engines.nh_termination import run_new_hires


def run_one_year(
    event_log: pd.DataFrame,
    prev_snapshot: pd.DataFrame,
    year: int,
    global_params: Any,
    plan_rules: Dict[str, Any],
    hazard_table: pd.DataFrame,
    rng: Any,
    census_template_path: Optional[str] = None,
    rng_seed_offset: int = 0,
    deterministic_term: bool = False,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orchestrates simulation for a single year, following the new hiring/termination flow:
      1. Markov promotions/exits (experienced only)
      2. Hazard-based terminations (experienced only)
      3. Update snapshot to survivors
      4. Compute headcount targets (gross/net)
      5. Generate/apply hires
      6. Deterministic new-hire terminations
      7. Final snapshot + validation
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[RUN_ONE_YEAR] Simulating year {year}")

    # --- 1. Initialization and validation ---
    as_of = pd.Timestamp(f"{year}-01-01")
    prev_snapshot = ensure_snapshot_cols(prev_snapshot)
    hazard_slice = hazard_table
    year_rng = rng
    census_template_path = getattr(global_params, "census_template_path", census_template_path)

    # --- 2. Markov promotions & exits ---
    logger.info("[STEP] Markov promotions/exits (experienced only)")
    promo_time = as_of  # Promotions at SOY
    promotion_raise_config = getattr(global_params, 'promotion_raise_config', {})
    promotions_df, raises_df, exited_df, survivors_after_markov = apply_markov_promotions(
        snapshot=prev_snapshot,
        rng=year_rng,
        simulation_year=year,
        promo_time=promo_time,
        promotion_raise_config=promotion_raise_config
    )
    logger.info(f"[MARKOV] Promotions: {len(promotions_df)}, Raises: {len(raises_df)}, Exits: {len(exited_df)}")

    # --- 3. Hazard-based terminations (experienced only) ---
    logger.info("[STEP] Hazard-based terminations (experienced only)")
    # Only process experienced employees (not new hires)
    experienced_mask = survivors_after_markov['EMP_HIRE_DATE'] < pd.Timestamp(f"{year}-01-01")
    experienced = survivors_after_markov[experienced_mask].copy()
    # Run hazard-based terminations
    term_event_dfs = term.run(
        snapshot=experienced,
        hazard_slice=hazard_slice,
        rng=year_rng,
        deterministic=False
    )
    term_events = term_event_dfs[0] if term_event_dfs else pd.DataFrame()
    comp_events = term_event_dfs[1] if len(term_event_dfs) > 1 else pd.DataFrame()
    logger.info(f"[TERM] Terminations: {len(term_events)}, Prorated comp events: {len(comp_events)}")
    # Remove terminated employees from survivors
    terminated_ids = set(term_events['employee_id']) if not term_events.empty else set()
    survivors_after_term = survivors_after_markov[~survivors_after_markov[EMP_ID].isin(terminated_ids)].copy()

    # --- 4. Update snapshot to survivors ---
    logger.info("[STEP] Update snapshot to survivors (post-terminations)")
    snapshot_survivors = survivors_after_term.copy()

    # --- 5. Compute headcount targets ---
    start_count = snapshot_survivors['active'].sum() if 'active' in snapshot_survivors.columns else len(snapshot_survivors)
    target_growth = getattr(global_params, 'target_growth', 0.0)
    nh_term_rate = getattr(global_params, 'new_hire_termination_rate', 0.0)
    target_eoy, net_hires, gross_hires = compute_headcount_targets(start_count, start_count, target_growth, nh_term_rate)
    logger.info(f"[DEBUG-HIRE] Start: {start_count}, Net Hires: {net_hires}, Gross Hires: {gross_hires}, Target EOY: {target_eoy}")

    # --- 6. Generate/apply hires ---
    logger.info("[STEP] Generate/apply hires")
    hires_events, snapshot_with_hires = hire.process_new_hires(
        temp_snapshot=snapshot_survivors,
        gross_hires=gross_hires,
        hazard_slice=hazard_slice,
        year_rng=year_rng,
        census_template_path=census_template_path,
        global_params=global_params,
        term_events=term_events,
        year=year
    )

    # --- 7. Deterministic new-hire terminations ---
    logger.info("[STEP] Deterministic new-hire terminations")
    nh_term_event_dfs = run_new_hires(
        snapshot=snapshot_with_hires,
        hazard_slice=hazard_slice,
        rng=year_rng,
        year=year,
        deterministic=True
    )
    nh_term_events = nh_term_event_dfs[0] if nh_term_event_dfs else pd.DataFrame()
    nh_comp_events = nh_term_event_dfs[1] if len(nh_term_event_dfs) > 1 else pd.DataFrame()
    logger.info(f"[NH_TERM] New-hire terminations: {len(nh_term_events)}, Prorated comp events: {len(nh_comp_events)}")
    # Remove terminated new hires from snapshot
    nh_terminated_ids = set(nh_term_events['employee_id']) if not nh_term_events.empty else set()
    final_snapshot = snapshot_with_hires[~snapshot_with_hires[EMP_ID].isin(nh_terminated_ids)].copy()

    # --- 8. Final snapshot + validation ---
    logger.info("[STEP] Final snapshot + validation")
    validate_eoy_snapshot(final_snapshot, target_eoy)
    # Assert no duplicate EMP_IDs
    assert not final_snapshot[EMP_ID].duplicated().any(), "Duplicate EMP_IDs detected!"

    # --- 9. Aggregate event log ---
    logger.info("[STEP] Build event log")
    event_frames = [
        promotions_df, raises_df, exited_df,
        term_events, comp_events,
        nh_term_events, nh_comp_events
    ]
    event_log = pd.concat([df for df in event_frames if df is not None and not df.empty], ignore_index=True, sort=False)
    logger.info(f"[EVENT_LOG] Total events: {len(event_log)}")

    logger.info(f"[RESULT] EOY={final_snapshot['active'].sum() if 'active' in final_snapshot.columns else 'unknown'} (target={target_eoy})")
    return event_log, final_snapshot
