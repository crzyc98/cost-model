"""
Orchestrator module for run_one_year package.

Coordinates the execution of all simulation steps for a single year.
"""
import json
import logging  # For type hinting and legacy usage
from logging_config import get_logger, get_diagnostic_logger
import uuid
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np

from cost_model.state.event_log import EVENT_COLS, EVENT_PANDAS_DTYPES

from cost_model.engines import hire
from cost_model.state.snapshot import update as snapshot_update
from cost_model.state.schema import EMP_ID, SIMULATION_YEAR, EMP_LEVEL, EMP_TENURE_BAND, EMP_GROSS_COMP
from cost_model.utils.tenure_utils import standardize_tenure_band

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
    logger = get_logger(__name__)
    logger.info(f"[RUN_ONE_YEAR] Simulating year {year}")
    all_new_events_list = []  # Initialize list to collect event DataFrames

    # --- 1. Initialization and validation ---
    as_of = pd.Timestamp(f"{year}-01-01")
    
    # Get diagnostic logger for non-critical messages
    diag_logger = get_diagnostic_logger(__name__)
    
    # Add diagnostics for EMP_LEVEL before ensure_snapshot_cols
    diag_logger.debug(f"[ORCHESTRATOR DIAGNOSTIC YR={year}] Before ensure_snapshot_cols:")
    if EMP_LEVEL in prev_snapshot.columns:
        diag_logger.debug(f"  pre-ensure_snapshot_cols['{EMP_LEVEL}'] NA count: {prev_snapshot[EMP_LEVEL].isna().sum()}")
        diag_logger.debug(f"  pre-ensure_snapshot_cols['{EMP_LEVEL}'] dtype: {prev_snapshot[EMP_LEVEL].dtype}")
        # Log level distribution if there are no NaN values
        if prev_snapshot[EMP_LEVEL].isna().sum() == 0:
            level_counts = prev_snapshot[EMP_LEVEL].value_counts().to_dict()
            diag_logger.debug(f"  Level distribution before ensure_snapshot_cols: {level_counts}")
    else:
        diag_logger.warning(f"{EMP_LEVEL} column not found in prev_snapshot before ensure_snapshot_cols")
    
    prev_snapshot = ensure_snapshot_cols(prev_snapshot)
    
    # Add diagnostics for EMP_LEVEL after ensure_snapshot_cols
    diag_logger.debug(f"[ORCHESTRATOR DIAGNOSTIC YR={year}] After ensure_snapshot_cols:")
    diag_logger.debug(f"  post-ensure_snapshot_cols['{EMP_LEVEL}'] NA count: {prev_snapshot[EMP_LEVEL].isna().sum()}")
    diag_logger.debug(f"  post-ensure_snapshot_cols['{EMP_LEVEL}'] dtype: {prev_snapshot[EMP_LEVEL].dtype}")
    # Log level distribution if there are no NaN values
    if prev_snapshot[EMP_LEVEL].isna().sum() == 0:
        level_counts = prev_snapshot[EMP_LEVEL].value_counts().to_dict()
        diag_logger.debug(f"  Level distribution after ensure_snapshot_cols: {level_counts}")

    # Validate EMP_ID in prev_snapshot
    if EMP_ID in prev_snapshot.columns and prev_snapshot[EMP_ID].isna().any():
        na_count = prev_snapshot[EMP_ID].isna().sum()
        diag_logger.warning(
            f"Year {year}: Input snapshot (prev_snapshot) contains {na_count} "
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
    
    # CRITICAL FIX: Standardize tenure band formats to ensure consistent matching
    # Standardize both experienced employees and hazard slice tenure bands
    if EMP_TENURE_BAND in experienced.columns:
        experienced[EMP_TENURE_BAND] = experienced[EMP_TENURE_BAND].apply(standardize_tenure_band)
        logger.info(f"[TERM STANDARDIZATION] Standardized employee tenure bands: {experienced[EMP_TENURE_BAND].unique().tolist()}")
    
    if EMP_TENURE_BAND in hz_slice.columns:
        hz_slice[EMP_TENURE_BAND] = hz_slice[EMP_TENURE_BAND].apply(standardize_tenure_band)
        logger.info(f"[TERM STANDARDIZATION] Standardized hazard slice tenure bands: {hz_slice[EMP_TENURE_BAND].unique().tolist()}")
    
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
    
    # CRITICAL FIX: Standardize tenure band formats for hiring, just as we did for terminations
    # Standardize both survivor snapshot and hazard slice tenure bands
    if EMP_TENURE_BAND in survivors_after_term.columns:
        survivors_after_term[EMP_TENURE_BAND] = survivors_after_term[EMP_TENURE_BAND].apply(standardize_tenure_band)
        logger.info(f"[HIRE STANDARDIZATION] Standardized survivor tenure bands: {survivors_after_term[EMP_TENURE_BAND].unique().tolist()}")
    
    if EMP_TENURE_BAND in hz_slice.columns:
        hz_slice[EMP_TENURE_BAND] = hz_slice[EMP_TENURE_BAND].apply(standardize_tenure_band)
        logger.info(f"[HIRE STANDARDIZATION] Standardized hazard slice tenure bands: {hz_slice[EMP_TENURE_BAND].unique().tolist()}")
    
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
        
        # Define a helper function to extract level from value_json
        def extract_level_from_hire_event(row):
            """Extract the employee level from a hire event row's value_json."""
            try:
                if 'value_json' in row:
                    # Parse JSON if it's a string
                    if isinstance(row['value_json'], str):
                        try:
                            value_json = json.loads(row['value_json'])
                        except (json.JSONDecodeError, TypeError):
                            return None
                    else:
                        value_json = row['value_json']  # Already a dict
                        
                    # Extract level from 'role' field (this is where hire.py stores the level)
                    if isinstance(value_json, dict) and 'role' in value_json:
                        level = value_json['role']
                        # Convert to int if possible
                        if isinstance(level, (int, float)):
                            return int(level)
                        elif isinstance(level, str) and level.isdigit():
                            return int(level)
            except Exception as e:
                logger.debug(f"Error extracting level for employee {row.get(EMP_ID, 'unknown')}: {str(e)}")
            return None

        # Enhanced diagnostic logging
        diag_logger.info(f"[ORCHESTRATOR YR={year}] Entering level extraction block for {len(hires_events)} hire events")
        if not hires_events.empty:
            # Sample the first few hire events to understand their structure
            sample_row = hires_events.iloc[0]
            diag_logger.debug(f"[ORCHESTRATOR YR={year}] Sample hire event columns: {sample_row.index.tolist()}")
            
            # Examine value_json content
            if 'value_json' in sample_row:
                try:
                    if isinstance(sample_row['value_json'], str):
                        value_json = json.loads(sample_row['value_json'])
                    else:
                        value_json = sample_row['value_json']
                    diag_logger.debug(f"[ORCHESTRATOR YR={year}] Sample value_json keys: {list(value_json.keys()) if isinstance(value_json, dict) else 'Not a dict'}")
                    if isinstance(value_json, dict) and 'role' in value_json:
                        diag_logger.debug(f"[ORCHESTRATOR YR={year}] Sample role value: {value_json['role']}, type: {type(value_json['role'])}")
                except Exception as e:
                    logger.warning(f"[ORCHESTRATOR YR={year}] Error examining value_json: {str(e)}")

        # Extract levels using apply (vectorized)
        extracted_levels = hires_events.apply(extract_level_from_hire_event, axis=1)

        # Log statistics about extracted levels with enhanced visibility
        extracted_count = extracted_levels.notna().sum()
        diag_logger.info(f"[ORCHESTRATOR YR={year}] Successfully extracted levels for {extracted_count} out of {len(hires_events)} new hires")
        if extracted_count < len(hires_events):
            logger.warning(f"[ORCHESTRATOR YR={year}] Using default level 1 for {len(hires_events) - extracted_count} new hires without extractable levels")
            
        if not hires_events.empty:
            # Show sample of extracted values
            sample_levels = extracted_levels.head(3).tolist()
            diag_logger.debug(f"[ORCHESTRATOR YR={year}] Sample of extracted levels: {sample_levels}")
        
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
            'employee_tenure_band': '0-1',  # New hires have 0-1 year tenure
            'employee_tenure': 0.0,  # New hires have 0 years tenure
            'employee_level': extracted_levels.fillna(1).astype('Int64'),  # Use extracted levels, default to 1 where missing
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
        
        # Diagnostic logging for snapshot_with_hires right after concatenation
        if not snapshot_with_hires.empty:
            diag_logger.info(f"[ORCHESTRATOR DIAGNOSTIC YR={year}] After snapshot_with_hires creation (concat):")
            if 'employee_level' in snapshot_with_hires.columns:
                level_na_count = snapshot_with_hires['employee_level'].isna().sum()
                diag_logger.debug(f"  snapshot_with_hires['employee_level'] NA count: {level_na_count}")
                diag_logger.debug(f"  snapshot_with_hires['employee_level'] dtype: {snapshot_with_hires['employee_level'].dtype}")
                if level_na_count > 0:
                    # Check if there are any new hires with NA levels
                    if 'job_level_source' in snapshot_with_hires.columns:
                        new_hire_level_na = snapshot_with_hires[
                            (snapshot_with_hires['employee_level'].isna()) & 
                            (snapshot_with_hires['job_level_source'] == 'new_hire')
                        ]
                        logger.warning(f"  Number of new hires with NA levels: {len(new_hire_level_na)}")
                        if not new_hire_level_na.empty:
                            logger.warning(f"  Sample of new hire employee IDs with NA levels: {new_hire_level_na[EMP_ID].head().tolist()}")
                    # Show all entries with NA levels
                    logger.warning(f"  Sample of ALL employee IDs with NA levels: {snapshot_with_hires[snapshot_with_hires['employee_level'].isna()][EMP_ID].head().tolist()}")
                # Show a frequency table of the levels
                level_counts = snapshot_with_hires['employee_level'].value_counts(dropna=False).to_dict()
                diag_logger.debug(f"  Level distribution in snapshot_with_hires: {level_counts}")
            else:
                logger.warning(f"  'employee_level' column not found in snapshot_with_hires! Columns: {snapshot_with_hires.columns.tolist()}")
        
        # Add new hire events to the list of all events for the year
        all_new_events_list.append(hires_events)
        
        # Diagnostic logging for new_hires DataFrame
        if not new_hires.empty:
            diag_logger.info(f"[ORCHESTRATOR DIAGNOSTIC YR={year}] After new_hires DataFrame creation:")
            # Check EMP_LEVEL column specifically
            if 'employee_level' in new_hires.columns:
                level_na_count = new_hires['employee_level'].isna().sum()
                diag_logger.debug(f"  new_hires['employee_level'] NA count: {level_na_count}")
                diag_logger.debug(f"  new_hires['employee_level'] dtype: {new_hires['employee_level'].dtype}")
                if level_na_count > 0:
                    logger.warning(f"  Sample of employee IDs with NA levels: {new_hires[new_hires['employee_level'].isna()][EMP_ID].head().tolist()}")
                # Show a frequency table of the levels
                level_counts = new_hires['employee_level'].value_counts(dropna=False).to_dict()
                diag_logger.debug(f"  Level distribution in new_hires: {level_counts}")
            else:
                logger.warning(f"  'employee_level' column not found in new_hires! Columns: {new_hires.columns.tolist()}")
                
            # Also check EMP_ID column
            if EMP_ID in new_hires.columns:
                na_sum_new_hires = new_hires[EMP_ID].isna().sum()
                diag_logger.debug(f"  new_hires['{EMP_ID}'] dtype: {new_hires[EMP_ID].dtype}")
                diag_logger.debug(f"  new_hires['{EMP_ID}'] NA sum: {na_sum_new_hires}")
                if na_sum_new_hires > 0:
                    logger.warning(f"  Sample of NA IDs in new_hires['{EMP_ID}']: {new_hires[new_hires[EMP_ID].isna()][EMP_ID].head().tolist()}")
            else:
                logger.warning(f"  '{EMP_ID}' column not found in new_hires! Columns: {new_hires.columns.tolist()}")
        else:
            logger.warning(f"[ORCHESTRATOR DIAGNOSTIC YR={year}] After new_hires DataFrame creation: new_hires is empty.")
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
    
    # Diagnostic logging before new hire termination
    if not snapshot_with_hires.empty:
        diag_logger.info(f"[ORCHESTRATOR DIAGNOSTIC YR={year}] Before new-hire termination:")
        if 'employee_level' in snapshot_with_hires.columns:
            level_na_count = snapshot_with_hires['employee_level'].isna().sum()
            diag_logger.debug(f"  Pre-termination snapshot['employee_level'] NA count: {level_na_count}")
            diag_logger.debug(f"  Pre-termination snapshot['employee_level'] dtype: {snapshot_with_hires['employee_level'].dtype}")
            # More detailed analysis of where NA values are
            if level_na_count > 0:
                # Check if these are new hires
                if 'job_level_source' in snapshot_with_hires.columns:
                    level_na_by_source = snapshot_with_hires[snapshot_with_hires['employee_level'].isna()]\
                        .groupby('job_level_source', dropna=False).size().to_dict()
                    logger.warning(f"  NA levels by job_level_source: {level_na_by_source}")
                    
                # Check tenure of employees with NA levels
                if 'employee_tenure_band' in snapshot_with_hires.columns:
                    level_na_by_tenure = snapshot_with_hires[snapshot_with_hires['employee_level'].isna()]\
                        .groupby('employee_tenure_band', dropna=False).size().to_dict()
                    logger.warning(f"  NA levels by tenure band: {level_na_by_tenure}")
        else:
            logger.warning(f"  'employee_level' column not found in pre-termination snapshot! Columns: {snapshot_with_hires.columns.tolist()}")
    
    from cost_model.engines.nh_termination import run_new_hires
    nh_term_events, nh_term_comp_events = run_new_hires(snapshot_with_hires, hz_slice, year_rng, year, deterministic=True)
    terminated_ids = set(nh_term_events[EMP_ID]) if not nh_term_events.empty else set()
    final_snapshot = snapshot_with_hires.loc[~snapshot_with_hires.index.isin(terminated_ids)]
    logger.info(f"[NH-TERM] Terminated {len(terminated_ids)} new hires")

    # --- 9. Final snapshot + validation ---
    logger.info("[STEP] Final snapshot + validation")
    logger.info(f"[FINAL] Final headcount: {len(final_snapshot)}")
    
    # Comprehensive diagnostic logging for final snapshot
    if not final_snapshot.empty:
        diag_logger.info(f"[ORCHESTRATOR DIAGNOSTIC YR={year}] Final snapshot analysis:")
        if 'employee_level' in final_snapshot.columns:
            level_na_count = final_snapshot['employee_level'].isna().sum()
            diag_logger.debug(f"  Final snapshot['employee_level'] NA count: {level_na_count}")
            diag_logger.debug(f"  Final snapshot['employee_level'] dtype: {final_snapshot['employee_level'].dtype}")
            
            # Detailed analysis if NA values exist
            if level_na_count > 0:
                # Get sample of employees with NA levels
                na_sample = final_snapshot[final_snapshot['employee_level'].isna()]
                if not na_sample.empty:
                    logger.warning(f"  Sample of employee IDs with NA levels: {na_sample[EMP_ID].head().tolist()}")
                    
                    # Check other important columns for these employees
                    if 'employee_hire_date' in na_sample.columns:
                        hire_year_counts = na_sample['employee_hire_date'].dt.year.value_counts().to_dict()
                        logger.warning(f"  Hire years for employees with NA levels: {hire_year_counts}")
                        
                    if 'simulation_year' in na_sample.columns:
                        sim_year_counts = na_sample['simulation_year'].value_counts().to_dict()
                        logger.warning(f"  Simulation years for employees with NA levels: {sim_year_counts}")
                    
                    if 'job_level_source' in na_sample.columns:
                        source_counts = na_sample['job_level_source'].value_counts(dropna=False).to_dict()
                        logger.warning(f"  Job level sources for employees with NA levels: {source_counts}")
            
            # Show distribution of level values
            level_counts = final_snapshot['employee_level'].value_counts(dropna=False).to_dict()
            diag_logger.info(f"  Level distribution in final snapshot: {level_counts}")
            
            # Check for any level=0 employees (possibly from demotions)
            if 0 in level_counts:
                level0_count = level_counts[0]
                diag_logger.info(f"  Found {level0_count} employees with level=0 in final snapshot")
        else:
            logger.warning(f"  'employee_level' column not found in final snapshot! Columns: {final_snapshot.columns.tolist()}")

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
    # Removed duplicate append of nh_term_comp_events
    
    # 1. Filter out empty DataFrames before concatenation
    non_empty_events = [df for df in events_to_concat if not df.empty]
    
    # 1b. Pre-validation: Ensure all event DataFrames have required columns with proper types
    required_cols = ['event_time', 'event_type', EMP_ID]
    validated_events = []
    
    for i, df in enumerate(non_empty_events):
        # Only proceed with validation if DataFrame has events
        if not df.empty:
            is_valid = True
            # Check for required columns
            for col in required_cols:
                if col not in df.columns:
                    logger.warning(f"DataFrame {i} is missing required column '{col}'. Skipping.")
                    is_valid = False
                    break
                    
            # Check for NAs in required columns
            if is_valid:
                for col in required_cols:
                    if df[col].isna().any():
                        na_count = df[col].isna().sum()
                        logger.warning(f"DataFrame {i} has {na_count} NA values in required column '{col}'. Fixing.")
                        
                        # For event_time, fill with year timestamp
                        if col == 'event_time' and na_count > 0:
                            df[col] = df[col].fillna(pd.Timestamp(f"{year}-01-01"))
                            
                        # For event_type and employee_id, drop rows with NA values
                        if (col == 'event_type' or col == EMP_ID) and na_count > 0:
                            df = df.dropna(subset=[col])
                
                # Only add valid, non-empty DataFrames
                if not df.empty:
                    validated_events.append(df)
            
    # 2. Create this year's new events
    if validated_events:
        logger.info(f"Concatenating {len(validated_events)} validated event DataFrames for year {year}")
        try:
            new_events = pd.concat(validated_events, ignore_index=True)
            logger.info(f"Successfully created {len(new_events)} new events for year {year}")
        except Exception as e:
            logger.error(f"Error concatenating events for year {year}: {e}")
            logger.debug(f"Event DataFrames being concatenated: {[df.shape for df in validated_events]}")
            raise
    else:
        logger.warning(f"No valid event DataFrames to concatenate for year {year}")
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
