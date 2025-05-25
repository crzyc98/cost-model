"""
Orchestrator module for run_one_year package.

Coordinates the execution of all simulation steps for a single year.
"""
import json
import logging
import uuid
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np

from cost_model.state.event_log import EVENT_COLS, EVENT_PANDAS_DTYPES

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
    promotions_df, raises_df, exited_df = apply_markov_promotions(
        snapshot=prev_snapshot,
        promo_time=promo_time,
        rng=year_rng,
        promotion_raise_config=promotion_raise_config,
        simulation_year=year,
        global_params=global_params        # <-- pass through so dev_mode is honoured
    )
    logger.info(f"[MARKOV] Promotions: {len(promotions_df)}, Raises: {len(raises_df)}, Exits: {len(exited_df)}")
    
    # Get survivors after markov promotions/exits
    exited_emp_ids = set(exited_df['employee_id'].unique())
    survivors_after_markov = prev_snapshot[~prev_snapshot['employee_id'].isin(exited_emp_ids)].copy()

    # --- 3. Hazard-based terminations (experienced only) ---
    logger.info("[STEP] Hazard-based terminations (experienced only)")
    # Only process experienced employees (not new hires)
    experienced_mask = survivors_after_markov['employee_hire_date'] < pd.Timestamp(f"{year}-01-01")
    experienced = survivors_after_markov[experienced_mask].copy()
    
    # Filter hazard table by year for termination engines
    from cost_model.state.schema import EMP_LEVEL, EMP_TENURE_BAND
    hz_slice = hazard_table[hazard_table['simulation_year'] == year].drop_duplicates([EMP_LEVEL, EMP_TENURE_BAND])
    
    # Run hazard-based terminations
    term_event_dfs = term.run(
        snapshot=experienced,
        hazard_slice=hz_slice,
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
    start_count = prev_snapshot['active'].sum() if 'active' in prev_snapshot.columns else len(prev_snapshot)
    survivor_count = survivors_after_term['active'].sum() if 'active' in survivors_after_term.columns else len(survivors_after_term)
    target_growth = getattr(global_params, 'target_growth', 0.0)
    nh_term_rate = getattr(global_params, 'new_hire_termination_rate', 0.0)
    target_eoy, net_hires, gross_hires = compute_headcount_targets(start_count, survivor_count, target_growth, nh_term_rate)
    logger.info(f"[DEBUG-HIRE] Start: {start_count}, Survivors: {survivor_count}, Net Hires: {net_hires}, Gross Hires: {gross_hires}, Target EOY: {target_eoy}")

    # --- 6. Generate/apply hires ---
    logger.info("[STEP] Generate/apply hires")
    # Filter hazard table by year to ensure correct matching
    from cost_model.state.schema import EMP_LEVEL, EMP_TENURE_BAND
    hz_slice = hazard_table[hazard_table['simulation_year'] == year].drop_duplicates([EMP_LEVEL, EMP_TENURE_BAND])
    hires_result = hire.run(
        snapshot=survivors_after_term,
        hires_to_make=gross_hires,  # Use gross_hires to account for expected new hire terminations
        hazard_slice=hz_slice,
        rng=year_rng,
        census_template_path=census_template_path,
        global_params=global_params,
        terminated_events=term_events
    )
    
    # The run function returns a list with hire events and comp events
    hires_events = hires_result[0]  # Get the hire events
    logger.info(f"[HIRES] Added {len(hires_events)} new hires")

    # --- 7. Update snapshot with hires ---
    # Since the run function doesn't return the updated snapshot, we'll need to create it
    # by combining the survivors with the new hires
    # This is a simplified approach - you might need to adjust based on your actual data structure
    if not hires_events.empty:
        def safe_get_meta(meta_str, key, default=None):
            """Safely get a value from the meta JSON string."""
            if pd.isna(meta_str) or not meta_str:
                return default
            try:
                meta = json.loads(meta_str)
                return meta.get(key, default)
            except (json.JSONDecodeError, TypeError):
                return default

        # Create a DataFrame with the new hires
        new_hires = pd.DataFrame({
            'employee_id': hires_events['employee_id'],
            'employee_hire_date': pd.to_datetime(hires_events['event_time']),
            'employee_birth_date': pd.to_datetime('1990-01-01'),  # Default date
            'employee_role': 'UNKNOWN',  # Default role
            'employee_gross_comp': hires_events['value_num'],
            'employee_termination_date': pd.NaT,
            'active': True,
            'employee_deferral_rate': 0.0,  # Default value, adjust as needed
            'employee_tenure_band': '0-1yr',  # New hires have 0-1 year tenure
            'employee_tenure': 0.0,  # New hires have 0 years tenure
            'employee_level': 1,  # Default level for new hires, adjust as needed
            'job_level_source': 'new_hire',
            'exited': False,
            'simulation_year': year
        })
        
        # Try to get birth date and role from meta if available
        if 'meta' in hires_events.columns:
            new_hires['employee_birth_date'] = hires_events['meta'].apply(
                lambda x: pd.to_datetime(safe_get_meta(x, 'birth_date', '1990-01-01'))
            )
            new_hires['employee_role'] = hires_events['meta'].apply(
                lambda x: safe_get_meta(x, 'role', 'UNKNOWN')
            )
        
        # Set the index to employee_id to match the snapshot
        new_hires.set_index('employee_id', inplace=True)
        
        # Combine the survivors with the new hires
        snapshot_with_hires = pd.concat([survivors_after_term, new_hires])
    else:
        snapshot_with_hires = survivors_after_term

    # --- 8. Deterministic new-hire terminations ---
    logger.info("[STEP] Deterministic new-hire terminations")
    from cost_model.engines.nh_termination import run_new_hires
    nh_term_events, nh_term_comp_events = run_new_hires(snapshot_with_hires, hz_slice, year_rng, year, deterministic=True)
    terminated_ids = set(nh_term_events['employee_id']) if not nh_term_events.empty else set()
    final_snapshot = snapshot_with_hires.loc[~snapshot_with_hires.index.isin(terminated_ids)]
    logger.info(f"[NH-TERM] Terminated {len(terminated_ids)} new hires")

    # --- 9. Final snapshot + validation ---
    logger.info("[STEP] Final snapshot + validation")
    logger.info(f"[FINAL] Final headcount: {len(final_snapshot)}")

    # --- 9. Aggregate event log ---
    logger.info("[STEP] Build event log")
    
    # Initialize a list to collect all events
    events_to_concat = []
    
    # Add each event type if it's not empty
    if not promotions_df.empty:
        logger.info(f"Adding {len(promotions_df)} promotion events")
        events_to_concat.append(promotions_df)
    if not raises_df.empty:
        logger.info(f"Adding {len(raises_df)} raise events")
        events_to_concat.append(raises_df)
    if not exited_df.empty:
        logger.info(f"Adding {len(exited_df)} exit events")
        events_to_concat.append(exited_df)
    if not term_events.empty:
        logger.info(f"Adding {len(term_events)} termination events")
        events_to_concat.append(term_events)
    if not hires_events.empty:
        logger.info(f"Adding {len(hires_events)} hire events")
        events_to_concat.append(hires_events)
    if not nh_term_events.empty:
        logger.info(f"Adding {len(nh_term_events)} new hire termination events")
        events_to_concat.append(nh_term_events)
    if not comp_events.empty:
        logger.info(f"Adding {len(comp_events)} compensation events")
        events_to_concat.append(comp_events)
    if not nh_term_comp_events.empty:
        logger.info(f"Adding {len(nh_term_comp_events)} new hire compensation events")
        events_to_concat.append(nh_term_comp_events)
    if not nh_term_comp_events.empty:
        events_to_concat.append(nh_term_comp_events)
    
    # 1. Create this year's new events
    if events_to_concat:
        logger.info(f"Concatenating {len(events_to_concat)} event DataFrames for year {year}")
        new_events = pd.concat(events_to_concat, ignore_index=True)
        logger.info(f"Created {len(new_events)} new events for year {year}")
    else:
        logger.warning(f"No new events to concatenate for year {year}")
        new_events = pd.DataFrame(columns=EVENT_COLS)
    
    # 2. Ensure all required columns are present in the new events
    for col in EVENT_COLS:
        if col not in new_events.columns:
            new_events[col] = None
    
    # 3. Ensure proper data types for new events
    for col, dtype in EVENT_PANDAS_DTYPES.items():
        if col in new_events.columns:
            try:
                new_events[col] = new_events[col].astype(dtype)
            except Exception as e:
                logger.error(f"Failed to cast {col} to {dtype} for new events: {e}")
    
    # 4. Ensure event_time is properly set for new events
    if 'event_time' in new_events.columns and new_events['event_time'].isna().any():
        logger.warning(f"Found NA values in event_time for new events, filling with current year {year}")
        new_events['event_time'] = new_events['event_time'].fillna(pd.Timestamp(f"{year}-01-01"))
    
    # 5. Ensure event_type is properly set for new events
    if 'event_type' in new_events.columns and new_events['event_type'].isna().any():
        logger.warning("Found NA values in event_type for new events, dropping these events")
        new_events = new_events.dropna(subset=['event_type'])
    
    # 6. Ensure employee_id is present for all new events
    if 'employee_id' in new_events.columns and new_events['employee_id'].isna().any():
        logger.warning("Found NA values in employee_id for new events, dropping these events")
        new_events = new_events.dropna(subset=['employee_id'])
    
    # 7. Ensure event_id is unique and present for all new events
    if 'event_id' in new_events.columns:
        if new_events['event_id'].isna().any():
            logger.warning("Generating missing event_ids for new events")
            mask = new_events['event_id'].isna()
            new_events.loc[mask, 'event_id'] = [str(uuid.uuid4()) for _ in range(mask.sum())]
    else:
        logger.warning("event_id column missing in new events, generating new event_ids")
        new_events['event_id'] = [str(uuid.uuid4()) for _ in range(len(new_events))]
    
    # 8. Ensure new events have all required columns in the correct order
    if not new_events.empty:
        new_events = new_events[EVENT_COLS]
    
    # 9. Append new events to the incoming event log
    logger.info(f"Appending {len(new_events)} new events to the cumulative event log")
    if not event_log.empty:
        logger.info(f"Incoming event log has {len(event_log)} events")
        cumulative_events = pd.concat([event_log, new_events], ignore_index=True)
    else:
        logger.info("No incoming event log, using new events as cumulative")
        cumulative_events = new_events
    
    # 10. Sort all events by timestamp for chronological order
    if 'event_time' in cumulative_events.columns and not cumulative_events.empty:
        logger.info("Sorting events chronologically")
        cumulative_events['event_time'] = pd.to_datetime(cumulative_events['event_time'], errors='coerce')
        cumulative_events = cumulative_events.sort_values('event_time', ignore_index=True)
    
    logger.info(f"[RUN_ONE_YEAR] Year {year} complete. Final headcount: {len(final_snapshot)}, "
                f"New events: {len(new_events)}, Total events: {len(cumulative_events)}")
    
    # Log some statistics about the new events
    if not new_events.empty:
        new_event_counts = new_events['event_type'].value_counts()
        logger.info(f"New event type counts for year {year}:\n{new_event_counts.to_string()}")
    
    # Log some statistics about the cumulative events
    if not cumulative_events.empty:
        cumulative_event_counts = cumulative_events['event_type'].value_counts()
        logger.info(f"Cumulative event type counts (all years):\n{cumulative_event_counts.to_string()}")
    
    logger.info(f"[RESULT] EOY={final_snapshot['active'].sum() if 'active' in final_snapshot.columns else 'unknown'} (target={target_eoy})")
    
    # Return the cumulative event log and final snapshot
    return cumulative_events, final_snapshot
