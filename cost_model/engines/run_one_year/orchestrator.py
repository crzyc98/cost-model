"""
Orchestrator module for run_one_year package.

Coordinates the execution of all simulation steps for a single year.
"""
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np

from cost_model.engines import cola, promotion, hire, term
from cost_model.state.snapshot import update as snapshot_update
from cost_model.state.schema import EMP_ID

# Import submodules
from cost_model.engines.run_one_year_engine.validation import ensure_snapshot_cols, validate_and_extract_hazard_slice
from cost_model.engines.run_one_year_engine.plan_rules import run_all_plan_rules
from cost_model.engines.run_one_year_engine.comp_term import apply_compensation_and_terminations
from cost_model.engines.run_one_year_engine.hires import compute_hire_counts, process_new_hires
from cost_model.engines.run_one_year_engine.finalize import apply_new_hire_terminations, build_full_event_log, finalize_snapshot
from cost_model.engines.run_one_year_engine.utils import dbg


def run_one_year(
    prev_snapshot: pd.DataFrame,
    hazard_table: pd.DataFrame,
    year: int,
    global_params: Any,
    as_of: Optional[pd.Timestamp] = None,
    prev_as_of: Optional[pd.Timestamp] = None,
    prev_events: Optional[List[pd.DataFrame]] = None,
    deterministic_term: bool = False,
    random_seed: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orchestrates simulation for a single year, coordinating:
    - Compensation changes for experienced employees
    - Terminations
    - New hires and their compensation
    - New hire terminations
    - Plan rule processing (eligibility, enrollment, contributions)
    
    Args:
        prev_snapshot: Snapshot DataFrame from the previous year
        hazard_table: Hazard table with demographic transition probabilities
        year: Current simulation year
        global_params: Parameters for the simulation
        as_of: Current simulation date (default: Jan 1 of current year)
        prev_as_of: Previous simulation date (default: Jan 1 of previous year)
        prev_events: List of event DataFrames from previous years
        deterministic_term: Whether to use deterministic terminations
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple containing (event_log_df, final_snapshot_df)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[RUN_ONE_YEAR] Simulating year {year}")
    
    # --- 1. Initialization and validation ---
    # Set default timestamps if not provided
    if as_of is None:
        as_of = pd.Timestamp(f"{year}-01-01")
    if prev_as_of is None:
        prev_as_of = pd.Timestamp(f"{year-1}-01-01")
    
    # Initialize random number generator
    year_rng = np.random.default_rng(random_seed if random_seed is not None else year)
    
    # Validate inputs
    prev_snapshot = ensure_snapshot_cols(prev_snapshot)
    hazard_slice = validate_and_extract_hazard_slice(hazard_table, year)
    
    # Get config params from hazard slice
    census_template_path = getattr(global_params, "census_template_path", None)
    
    # --- 2. Run plan rules ---
    plan_rule_events = run_all_plan_rules(
        prev_snapshot=prev_snapshot,
        all_events=prev_events or [],
        hazard_cfg=hazard_slice,
        as_of=as_of,
        prev_as_of=prev_as_of,
        year=year
    )
    
    # --- 3. Run COLA and promotion modules ---
    cola_events = None
    if hasattr(global_params, "days_into_year_for_cola") and global_params.days_into_year_for_cola > 0:
        cola_events = cola.run(
            snapshot=prev_snapshot,
            year=year,
            global_params=global_params
        )
    
    promo_events = None
    if hasattr(global_params, "days_into_year_for_promotion") and global_params.days_into_year_for_promotion > 0:
        promo_events = promotion.run(
            snapshot=prev_snapshot,
            year=year,
            global_params=global_params
        )
    
    # --- 4. Apply compensation bumps and terminations ---
    comp_term_events, updated_snapshot = apply_compensation_and_terminations(
        prev_snapshot=prev_snapshot,
        hazard_slice=hazard_slice,
        global_params=global_params,
        year_rng=year_rng,
        deterministic_term=deterministic_term,
        year=year
    )
    
    # --- 5. Process new hires ---
    # Calculate number of hires
    start_count = (prev_snapshot["active"] == True).sum() if "active" in prev_snapshot.columns else len(prev_snapshot)
    net_hires, gross_hires = compute_hire_counts(
        start_count=start_count,
        global_params=global_params,
        hazard_slice=hazard_slice,
        year=year
    )
    
    # Process new hires if needed
    hires_events, snapshot_with_hires = process_new_hires(
        temp_snapshot=updated_snapshot,
        gross_hires=gross_hires,
        hazard_slice=hazard_slice,
        year_rng=year_rng,
        census_template_path=census_template_path,
        global_params=global_params,
        term_events=comp_term_events,
        year=year
    )
    
    # --- 6. Apply new hire terminations ---
    nh_term_events, final_snapshot = apply_new_hire_terminations(
        snap_with_hires=snapshot_with_hires,
        hazard_slice=hazard_slice,
        year_rng=year_rng,
        year=year
    )
    
    # --- 7. Finalize snapshot ---
    finalized_snapshot = finalize_snapshot(
        snapshot=final_snapshot,
        year=year
    )
    
    # --- 8. Build complete event log ---
    event_log = build_full_event_log(
        plan_rule_events=plan_rule_events,
        comp_term_events=comp_term_events,
        hires_events=hires_events,
        nh_term_events=nh_term_events,
        year=year
    )
    
    logger.info(f"[RUN_ONE_YEAR] Year {year} simulation complete: {len(event_log)} events, {len(finalized_snapshot)} employees ({finalized_snapshot['active'].sum() if 'active' in finalized_snapshot.columns else 'unknown'} active)")
    
    return event_log, finalized_snapshot
