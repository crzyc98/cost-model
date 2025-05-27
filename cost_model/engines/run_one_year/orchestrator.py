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
from cost_model.state.schema import EMP_ID, SIMULATION_YEAR, EMP_LEVEL, EMP_TENURE_BAND, EMP_GROSS_COMP

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
    all_new_events_list = []  # Initialize list to collect event DataFrames

    # --- 1. Initialization and validation ---
    as_of = pd.Timestamp(f"{year}-01-01")
    prev_snapshot = ensure_snapshot_cols(prev_snapshot)

    # Validate EMP_ID in prev_snapshot
    if EMP_ID in prev_snapshot.columns and prev_snapshot[EMP_ID].isna().any():
        na_count = prev_snapshot[EMP_ID].isna().sum()
        logger.warning(
            f"[RUN_ONE_YEAR] Year {year}: Input snapshot (prev_snapshot) contains {na_count} "
            f"records with NA {EMP_ID}. These records will be dropped."
        )
        prev_snapshot = prev_snapshot.dropna(subset=[EMP_ID]).copy()
        if prev_snapshot.empty:
            logger.error(
                f"[RUN_ONE_YEAR] Year {year}: prev_snapshot is empty after dropping NA {EMP_ID}s. "
                "This may lead to issues downstream or indicate a significant data problem."
            )
            # Depending on desired behavior, could raise an error or return empty results.
            # For now, allowing it to proceed with an empty snapshot.
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
    exited_emp_ids = set(exited_df[EMP_ID].unique())
    survivors_after_markov = prev_snapshot[~prev_snapshot[EMP_ID].isin(exited_emp_ids)].copy()

    # --- 3. Hazard-based terminations (experienced only) ---
    logger.info("[STEP] Hazard-based terminations (experienced only)")
    # Only process experienced employees (not new hires)
    experienced_mask = survivors_after_markov['employee_hire_date'] < pd.Timestamp(f"{year}-01-01")
    experienced = survivors_after_markov[experienced_mask].copy()
    
    # Filter hazard table by year for termination engines
    hz_slice = hazard_table[hazard_table['simulation_year'] == year].drop_duplicates([EMP_LEVEL, EMP_TENURE_BAND])
    
    # Run hazard-based terminations
    term_event_dfs = term.run(
        snapshot=experienced,
        hazard_slice=hz_slice,
        rng=year_rng,
        deterministic=False
    )
    term_events = term_event_dfs[0] if term_event_dfs and not term_event_dfs[0].empty else pd.DataFrame()
    comp_events = term_event_dfs[1] if len(term_event_dfs) > 1 and not term_event_dfs[1].empty else pd.DataFrame()

    if not term_events.empty and SIMULATION_YEAR not in term_events.columns:
        term_events[SIMULATION_YEAR] = year
    if not comp_events.empty and SIMULATION_YEAR not in comp_events.columns:
        comp_events[SIMULATION_YEAR] = year
    
    logger.info(f"[TERM] Terminations: {len(term_events)}, Prorated comp events: {len(comp_events)}")
    # Remove terminated employees from survivors
    # Ensure EMP_ID column exists before trying to access it
    terminated_ids = set(term_events[EMP_ID]) if not term_events.empty and EMP_ID in term_events.columns else set()
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
    hires_events = hires_result[0] if hires_result and not hires_result[0].empty else pd.DataFrame()
    hires_comp_events = hires_result[1] if len(hires_result) > 1 and not hires_result[1].empty else pd.DataFrame()

    if not hires_events.empty:
        if EMP_ID not in hires_events.columns:
            logger.error(f"Critical: '{EMP_ID}' column missing in hires_events. Cannot process hires.")
            hires_events = pd.DataFrame() # Treat as empty to prevent downstream errors
        else:
            original_hire_count = len(hires_events)
            hires_events.dropna(subset=[EMP_ID], inplace=True)
            if len(hires_events) < original_hire_count:
                logger.warning(
                    f"Dropped {original_hire_count - len(hires_events)} hire events with NA '{EMP_ID}'."
                )
        
        if not hires_events.empty and SIMULATION_YEAR not in hires_events.columns:
            hires_events[SIMULATION_YEAR] = year
            
    if not hires_comp_events.empty and SIMULATION_YEAR not in hires_comp_events.columns:
        hires_comp_events[SIMULATION_YEAR] = year
        
    logger.info(f"[HIRES] Processed {len(hires_events)} new hires after validation. Corresponding comp events: {len(hires_comp_events)}")

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
        logger.info(f"Generated {len(hires_events)} new hire events for year {year}")
        
        # Ensure employee_id from hires_events is explicitly handled as strings
        employee_ids_for_new_hires = hires_events[EMP_ID]
        if employee_ids_for_new_hires.dtype == 'object': # If it's object, it might contain Nones
            # Convert actual None to string "None", pd.NA to string "<NA>"
            employee_ids_for_new_hires = employee_ids_for_new_hires.apply(
                lambda x: "<NA>" if pd.isna(x) else ("None" if x is None else str(x))
            )
        elif pd.api.types.is_string_dtype(employee_ids_for_new_hires) and employee_ids_for_new_hires.isna().any():
            # If it's already a Pandas StringDtype (like pd.StringDtype()) and has pd.NA, convert pd.NA to "<NA>" string.
            employee_ids_for_new_hires = employee_ids_for_new_hires.fillna("<NA>")
        else: # For other dtypes, just ensure string conversion
            employee_ids_for_new_hires = employee_ids_for_new_hires.astype(str)
        
        # Extract compensation from value_json if available
        compensation_values = []
        for idx, row in hires_events.iterrows():
            # First try to get compensation from value_num (which might be None)
            comp_value = row.get('value_num')
            
            # If value_num is None, try to extract compensation from value_json
            if pd.isna(comp_value) and 'value_json' in row:
                try:
                    # Parse the JSON string if it's a string
                    if isinstance(row['value_json'], str):
                        value_json = json.loads(row['value_json'])
                    else:
                        value_json = row['value_json']  # Might already be a dict
                        
                    # Extract compensation from the parsed JSON
                    if isinstance(value_json, dict) and 'compensation' in value_json:
                        comp_value = float(value_json['compensation'])
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    logger.warning(f"Failed to extract compensation from value_json for hire event {row.get(EMP_ID, 'unknown')}: {str(e)}")
            
            compensation_values.append(comp_value)
        
        # Construct the new_hires DataFrame for the snapshot
        new_hires_data = {
            EMP_ID: employee_ids_for_new_hires, # Use the sanitized Series
            'employee_hire_date': pd.to_datetime(hires_events['event_time']),
            'employee_birth_date': pd.to_datetime('1990-01-01'),  # Default date
            'employee_role': 'UNKNOWN',  # Default role
            EMP_GROSS_COMP: compensation_values,  # Use the extracted compensation values
            'employee_termination_date': pd.NaT,
            'active': True,
            'employee_deferral_rate': 0.0,  # Default value, adjust as needed
            'employee_tenure_band': '0-1yr',  # New hires have 0-1 year tenure
            'employee_tenure': 0.0,  # New hires have 0 years tenure
            'employee_level': 1,  # Default level for new hires, adjust as needed
            'job_level_source': 'new_hire',
            'exited': False,
            'simulation_year': year
        }
        
        # Try to get birth date and role from meta if available
        if 'meta' in hires_events.columns:
            new_hires_data['employee_birth_date'] = hires_events['meta'].apply(
                lambda x: pd.to_datetime(safe_get_meta(x, 'birth_date', '1990-01-01'))
            )
            new_hires_data['employee_role'] = hires_events['meta'].apply(
                lambda x: safe_get_meta(x, 'role', 'UNKNOWN')
            )
        
        # Create the new_hires DataFrame and set the index to employee_id while keeping it as a column
        new_hires = pd.DataFrame(new_hires_data)
        
        # Check for and fix missing compensation values
        missing_comp_mask = new_hires[EMP_GROSS_COMP].isna()
        if missing_comp_mask.any():
            missing_count = missing_comp_mask.sum()
            logger.warning(f"Found {missing_count} new hires with missing compensation. Assigning default compensation values.")
            
            # Set default compensation based on role if available, otherwise use a reasonable default
            role_comp_defaults = getattr(global_params, 'compensation', {})
            if hasattr(role_comp_defaults, 'new_hire') and hasattr(role_comp_defaults.new_hire, 'comp_base_salary'):
                default_comp = role_comp_defaults.new_hire.comp_base_salary
            else:
                default_comp = 50000.0  # Fallback default
                
            logger.info(f"Using default compensation of {default_comp} for new hires with missing values")
            new_hires.loc[missing_comp_mask, EMP_GROSS_COMP] = default_comp
        
        # Don't drop employee_id column when setting index (key difference!)
        new_hires = new_hires.set_index(EMP_ID, drop=False)
        
        # Concatenate new hires with survivors
        # Ensure indices are compatible (both should be EMP_ID)
        # If survivors_after_term.index.name is not EMP_ID, it needs to be set.
        if survivors_after_term.index.name != EMP_ID and EMP_ID in survivors_after_term.columns:
            logger.warning(f"Setting index of survivors_after_term to '{EMP_ID}' before concat.")
            survivors_after_term = survivors_after_term.set_index(EMP_ID)
        elif survivors_after_term.index.name != EMP_ID and EMP_ID not in survivors_after_term.columns:
            logger.error(f"CRITICAL: Cannot set index for survivors_after_term, '{EMP_ID}' not in columns or index.")
            # Handle this error case, perhaps by raising an exception or returning

        # Before concatenating, ensure no duplicate EMP_IDs exist between survivors and new_hires
        # This should be guaranteed by hire.run's existing_ids check, but good to be defensive
        common_ids = survivors_after_term.index.intersection(new_hires.index)
        if not common_ids.empty:
            logger.warning(f"Found {len(common_ids)} duplicate EMP_IDs between survivors and new hires. New hires will overwrite. IDs: {common_ids.tolist()}")
            # This implies hire.run might not have perfectly unique IDs or existing_ids was incomplete.
            # For now, new_hires will overwrite, which might be desired if re-hiring.

        snapshot_with_hires = pd.concat([survivors_after_term, new_hires], sort=False) # sort=False is typical
        
        # Add new hire events to the list of all events for the year
        all_new_events_list.append(hires_events)
        
        # Diagnostic logging for new_hires DataFrame
        if not new_hires.empty and EMP_ID in new_hires.columns:
            logger.info(f"[ORCHESTRATOR DIAGNOSTIC YR={year}] After new_hires DataFrame creation:")
            logger.info(f"  new_hires['{EMP_ID}'] dtype: {new_hires[EMP_ID].dtype}")
            na_sum_new_hires = new_hires[EMP_ID].isna().sum()
            logger.info(f"  new_hires['{EMP_ID}'] NA sum: {na_sum_new_hires}")
            if na_sum_new_hires > 0:
                logger.info(f"  Sample of NA IDs in new_hires['{EMP_ID}']: {new_hires[new_hires[EMP_ID].isna()][EMP_ID].head().tolist()}")
            if new_hires[EMP_ID].dtype == 'object':
                none_sum_new_hires = new_hires[EMP_ID].apply(lambda x: x is None).sum()
                if none_sum_new_hires > 0:
                    logger.info(f"  new_hires['{EMP_ID}'] Python None sum: {none_sum_new_hires}")
                    logger.info(f"  Sample of None IDs in new_hires['{EMP_ID}']: {new_hires[new_hires[EMP_ID].apply(lambda x: x is None)][EMP_ID].head().tolist()}")
        elif new_hires.empty:
            logger.info(f"[ORCHESTRATOR DIAGNOSTIC YR={year}] After new_hires DataFrame creation: new_hires is empty.")
        else: # Not empty, but EMP_ID column missing
            logger.warning(f"[ORCHESTRATOR DIAGNOSTIC YR={year}] After new_hires DataFrame creation: new_hires is NOT empty, but MISSING '{EMP_ID}' column. Columns: {new_hires.columns.tolist()}")
    else:
        snapshot_with_hires = survivors_after_term

    # Ensure we don't have any NA or invalid employee IDs in the snapshot
    def is_valid_employee_id(emp_id):
        try:
            return emp_id is not None and not pd.isna(emp_id) and str(emp_id).strip() != ''
        except Exception:
            return False
            
    # Ensure EMP_ID is a column and not part of the index
    if EMP_ID in snapshot_with_hires.index.names and EMP_ID not in snapshot_with_hires.columns:
        snapshot_with_hires = snapshot_with_hires.reset_index(level=EMP_ID)
    
    # Create a mask for valid employee IDs
    valid_mask = snapshot_with_hires[EMP_ID].apply(is_valid_employee_id)
    invalid_count = len(snapshot_with_hires) - sum(valid_mask)
    
    if invalid_count > 0:
        logger.warning(
            f"Found {invalid_count} records with invalid employee IDs in snapshot_with_hires. "
            f"Filtering them out. Sample of invalid IDs: {snapshot_with_hires[~valid_mask][EMP_ID].head(5).tolist()}"
        )
        snapshot_with_hires = snapshot_with_hires[valid_mask].copy()

    if snapshot_with_hires.empty:
        logger.error("No valid employee records left after filtering invalid employee IDs")
        raise ValueError("No valid employee records with valid employee IDs found after filtering")

    # --- 7b. Ensure all employees have valid compensation values ---
    # Check for missing compensation in the entire snapshot and handle it
    missing_comp_mask = snapshot_with_hires[EMP_GROSS_COMP].isna()
    if missing_comp_mask.any():
        missing_count = missing_comp_mask.sum()
        logger.warning(f"Found {missing_count} employees with missing compensation in final snapshot. Will assign default values.")
        
        # Get the relevant employees and log some details for debugging
        missing_comp_df = snapshot_with_hires.loc[missing_comp_mask, [EMP_ID, EMP_HIRE_DATE, EMP_ROLE]].copy()
        if not missing_comp_df.empty:
            missing_comp_df[EMP_HIRE_DATE] = missing_comp_df[EMP_HIRE_DATE].dt.strftime('%Y-%m-%d')
            for _, row in missing_comp_df.head(5).iterrows():
                logger.warning(f"Employee {row[EMP_ID]} (hired {row[EMP_HIRE_DATE]}, role {row[EMP_ROLE]}) is missing compensation data")
            
            if len(missing_comp_df) > 5:
                logger.warning(f"... and {len(missing_comp_df) - 5} more employees with missing compensation")
        
        # Assign default compensation based on role if available, otherwise use a reasonable default
        role_comp_defaults = getattr(global_params, 'compensation', {})
        if hasattr(role_comp_defaults, 'new_hire') and hasattr(role_comp_defaults.new_hire, 'comp_base_salary'):
            default_comp = role_comp_defaults.new_hire.comp_base_salary
        else:
            default_comp = 50000.0  # Fallback default
            
        logger.info(f"Using default compensation of {default_comp} for employees with missing values")
        snapshot_with_hires.loc[missing_comp_mask, EMP_GROSS_COMP] = default_comp

    # --- 8. Run new-hire termination ---
    logger.info("[STEP] Deterministic new-hire terminations")
    from cost_model.engines.nh_termination import run_new_hires
    nh_term_events, nh_term_comp_events = run_new_hires(snapshot_with_hires, hz_slice, year_rng, year, deterministic=True)
    terminated_ids = set(nh_term_events[EMP_ID]) if not nh_term_events.empty else set()
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
    
    # 1. Filter out empty DataFrames before concatenation
    non_empty_events = [df for df in events_to_concat if not df.empty]
    
    # 2. Create this year's new events
    if non_empty_events:
        logger.info(f"Concatenating {len(non_empty_events)} non-empty event DataFrames for year {year}")
        try:
            new_events = pd.concat(non_empty_events, ignore_index=True)
            logger.info(f"Successfully created {len(new_events)} new events for year {year}")
        except Exception as e:
            logger.error(f"Error concatenating events for year {year}: {e}")
            logger.debug(f"Event DataFrames being concatenated: {[df.shape for df in non_empty_events]}")
            raise
    else:
        logger.warning(f"No non-empty event DataFrames to concatenate for year {year}")
        # Create an empty DataFrame with the correct columns and dtypes
        new_events = pd.DataFrame({col: pd.Series(dtype=str(t) if t != 'object' else object) 
                                 for col, t in EVENT_PANDAS_DTYPES.items()})
    
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
    if EMP_ID in new_events.columns and new_events[EMP_ID].isna().any():
        logger.warning(f"Found NA values in {EMP_ID} for new events, dropping these events")
        new_events = new_events.dropna(subset=[EMP_ID])
    
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
