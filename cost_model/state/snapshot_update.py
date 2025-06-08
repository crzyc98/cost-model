"""
Incrementally update a workforce snapshot given new events.
QuickStart: see docs/cost_model/state/snapshot_update.md
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

from cost_model.state.schema import (
    EVT_HIRE,
    EVT_COMP,
    EVT_TERM,
    EVT_COLA,
    EVT_PROMOTION,
    EVT_RAISE,
    EVT_CONTRIB,
    EVT_NEW_HIRE_TERM,
    EMP_ID,
    EMP_HIRE_DATE,
    EMP_BIRTH_DATE,
    EMP_GROSS_COMP,
    EMP_TERM_DATE,
    EMP_DEFERRAL_RATE,
    EMP_ACTIVE,
    EMP_TENURE_BAND,
    EMP_LEVEL,
    EMP_LEVEL_SOURCE,
    EMP_EXITED,
    EMP_TENURE,
    EMP_STATUS_EOY,
    EMP_CONTR,
    EMPLOYER_CORE,
    EMPLOYER_MATCH,
    IS_ELIGIBLE,
    ACTIVE_STATUS,
    SIMULATION_YEAR,
    SNAPSHOT_COLS,
    SNAPSHOT_DTYPES,
    EVENT_COLS,
    TERM_RATE,
)
from cost_model.state.snapshot_utils import (
    get_first_event,
    get_last_event,
    extract_hire_details,
    ensure_columns_and_types,
)
from cost_model.state.tenure import apply_tenure
from cost_model.utils.tenure_utils import standardize_tenure_band

logger = logging.getLogger(__name__)

__all__: List[str] = ["update"]

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

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

def _apply_new_hires(current: pd.DataFrame, new_events: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Apply new hire events to the current snapshot.

    Args:
        current: Current snapshot DataFrame
        new_events: DataFrame containing new events
        year: Current simulation year

    Returns:
        Updated DataFrame with new hires included
    """
    logger.debug("Starting _apply_new_hires for year %d", year)

    # Get all unique new hire IDs that aren't already in current snapshot
    hires = get_first_event(new_events, EVT_HIRE)
    if hires.empty:
        logger.debug("No hire events found in new_events")
        return current

    new_ids = hires[~hires[EMP_ID].isin(current.index)][EMP_ID].unique()
    if len(new_ids) == 0:
        logger.debug("No new hires to add (all IDs already exist in current snapshot)")
        return current

    logger.info("Adding %d new hires to snapshot for year %d", len(new_ids), year)

    # Process new hires
    batch = new_events[new_events[EMP_ID].isin(new_ids)]
    first_hire = get_first_event(batch, EVT_HIRE)
    details = extract_hire_details(first_hire)

    # Get last compensation and termination events for new hires
    last_comp = get_last_event(batch, EVT_COMP).set_index(EMP_ID)["value_num"].rename(EMP_GROSS_COMP)
    last_term = get_last_event(batch, [EVT_TERM, EVT_NEW_HIRE_TERM]).set_index(EMP_ID)["event_time"].rename(EMP_TERM_DATE)

    # Initialize new hires DataFrame
    new_df = pd.DataFrame(index=pd.Index(new_ids, name=EMP_ID))

    # Merge hire date and level source
    new_df = new_df.merge(
        first_hire.set_index(EMP_ID)["event_time"].rename(EMP_HIRE_DATE),
        left_index=True,
        right_index=True,
        how="left"
    )

    # Handle EMP_LEVEL_SOURCE
    if EMP_LEVEL_SOURCE in first_hire.columns:
        new_df = new_df.merge(
            first_hire.set_index(EMP_ID)[[EMP_LEVEL_SOURCE]],
            left_index=True,
            right_index=True,
            how="left"
        )
        logger.debug("Successfully merged EMP_LEVEL_SOURCE from first_hire events.")
    else:
        logger.warning("'%s' column not found in first_hire events. It will be NaN in new_df.", EMP_LEVEL_SOURCE)
        new_df[EMP_LEVEL_SOURCE] = pd.NA  # Ensure column exists for schema consistency

    # Merge other details
    new_df = new_df.merge(details, left_index=True, right_index=True, how="left")
    new_df = new_df.merge(last_comp, left_index=True, right_index=True, how="left")
    new_df = new_df.merge(last_term, left_index=True, right_index=True, how="left")

    # Set active status and simulation year
    new_df["active"] = new_df[EMP_TERM_DATE].isna()
    new_df[SIMULATION_YEAR] = year
    logger.debug("Set simulation_year=%d for %d new hires", year, len(new_df))
    logger.debug(f"P1_PRE_CHECK: EMP_LEVEL_SOURCE in new_df.columns: {EMP_LEVEL_SOURCE in new_df.columns}")
    logger.debug(f"P1_PRE_CHECK: new_df.empty: {new_df.empty}")
    if not new_df.empty:
        logger.debug(f"P1_PRE_CHECK: new_df columns: {new_df.columns.tolist()}")
    # --- Start of new P1 logging block ---
    p1_unique_values_output = "NOT_COMPUTED"
    if EMP_LEVEL_SOURCE in new_df.columns and not new_df.empty:
        try:
            p1_unique_values_output = str(new_df[EMP_LEVEL_SOURCE].unique().tolist())
        except Exception as e:
            p1_unique_values_output = f"ERROR_CALCULATING_UNIQUE: {type(e).__name__} - {e}"
            logger.error(f"SnapshotUpdate P1: Exception during .unique().tolist() for EMP_LEVEL_SOURCE: {e}", exc_info=True)
    elif EMP_LEVEL_SOURCE not in new_df.columns:
        p1_unique_values_output = "COLUMN_MISSING"
    elif new_df.empty:
        p1_unique_values_output = "DATAFRAME_EMPTY"
    logger.debug(f"SnapshotUpdate P1 (after initial merge): new_df job_level_source unique: {p1_unique_values_output}")

    if not new_df.empty:
        p1_head_output = "NOT_COMPUTED_HEAD"
        try:
            if EMP_LEVEL_SOURCE in new_df.columns:
                p1_head_output = new_df[[EMP_HIRE_DATE, EMP_LEVEL_SOURCE]].head().to_string()
                logger.debug(f"SnapshotUpdate P1 new_df head (index is EMP_ID):\n{p1_head_output}")
            else: # EMP_LEVEL_SOURCE is not in columns, but df is not empty
                p1_head_output = new_df[[EMP_HIRE_DATE]].head().to_string()
                logger.debug(f"SnapshotUpdate P1 new_df head (EMP_LEVEL_SOURCE missing, index is EMP_ID):\n{p1_head_output}")
        except Exception as e:
            p1_head_output = f"ERROR_CALCULATING_HEAD: {type(e).__name__} - {e}"
            logger.error(f"SnapshotUpdate P1: Exception during .head().to_string(): {e}", exc_info=True)
            # Log the error placeholder in the original message format if an error occurred
            if EMP_LEVEL_SOURCE in new_df.columns:
                 logger.debug(f"SnapshotUpdate P1 new_df head (index is EMP_ID):\n{p1_head_output}")
            else:
                 logger.debug(f"SnapshotUpdate P1 new_df head (EMP_LEVEL_SOURCE missing, index is EMP_ID):\n{p1_head_output}")
    else: # new_df is empty
        logger.debug("SnapshotUpdate P1: new_df is empty, skipping head display.")
    # --- End of new P1 logging block ---

    # Debug: Verify birth date types before ensure_columns_and_types
    if not pd.api.types.is_datetime64_any_dtype(new_df[EMP_BIRTH_DATE]):
        logger.warning(f"Birth dates not in datetime format: {new_df[EMP_BIRTH_DATE].dtype}")
        new_df[EMP_BIRTH_DATE] = pd.to_datetime(new_df[EMP_BIRTH_DATE], errors='coerce')

    new_df = ensure_columns_and_types(new_df)
    logger.debug(
        f"SnapshotUpdate P2 (after ensure_columns_and_types): new_df job_level_source unique: "
        f"{new_df[EMP_LEVEL_SOURCE].unique().tolist() if EMP_LEVEL_SOURCE in new_df.columns else 'COLUMN_MISSING'}"
    )
    if EMP_LEVEL_SOURCE in new_df.columns and not new_df.empty:
        logger.debug(f"SnapshotUpdate P2 new_df head (index is EMP_ID):\n{new_df[[EMP_HIRE_DATE, EMP_LEVEL_SOURCE]].head().to_string()}")
    elif not new_df.empty:
        logger.debug(f"SnapshotUpdate P2 new_df head (EMP_LEVEL_SOURCE missing, index is EMP_ID):\n{new_df[[EMP_HIRE_DATE]].head().to_string()}")

    as_of = pd.Timestamp(f"{year}-12-31")
    new_df = apply_tenure(new_df, EMP_HIRE_DATE, as_of, out_tenure_col=EMP_TENURE, out_band_col="tenure_band")

    # Ensure tenure bands are standardized
    if EMP_TENURE_BAND in new_df.columns:
        logger.debug(f"Standardizing tenure bands for {len(new_df)} new hires")
        new_df[EMP_TENURE_BAND] = new_df[EMP_TENURE_BAND].map(standardize_tenure_band)
        logger.debug(f"New hire tenure bands after standardization: {new_df[EMP_TENURE_BAND].value_counts().to_dict()}")

    new_df[EMP_ID] = new_df.index.astype(str)

    # --- Pre-concat Index Uniqueness Checks ---
    # 1. Ensure `current` (from prev_snapshot) has a unique index
    if not current.index.is_unique:
        current_dups = current.index[current.index.duplicated()].unique().tolist()
        logger.warning(f"`prev_snapshot` (now `current`) had duplicate EMP_IDs: {current_dups}. Deduplicating (keep='last').")
        current = current[~current.index.duplicated(keep='last')]

    # 2. Ensure `new_df` (new hires) has a unique index
    if not new_df.index.is_unique:
        new_df_dups = new_df.index[new_df.index.duplicated()].unique().tolist()
        logger.warning(f"`new_df` (new hires) had duplicate EMP_IDs: {new_df_dups}. Deduplicating (keep='last').")
        new_df = new_df[~new_df.index.duplicated(keep='last')]

    # 3. Assign job levels to new hires based on compensation
    from .job_levels.utils import assign_levels_to_dataframe
    new_df = assign_levels_to_dataframe(new_df, target_level_col=EMP_LEVEL)
    logger.debug(
        f"SnapshotUpdate P3 (after assign_levels): new_df job_level_source unique: "
        f"{new_df[EMP_LEVEL_SOURCE].unique().tolist() if EMP_LEVEL_SOURCE in new_df.columns else 'COLUMN_MISSING'}"
    )
    if EMP_LEVEL_SOURCE in new_df.columns and not new_df.empty:
        logger.debug(f"SnapshotUpdate P3 new_df head:\n{new_df[[EMP_ID, EMP_HIRE_DATE, EMP_LEVEL_SOURCE]].head().to_string()}")
    elif not new_df.empty:
        logger.debug(f"SnapshotUpdate P3 new_df head (EMP_LEVEL_SOURCE missing):\n{new_df[[EMP_ID, EMP_HIRE_DATE]].head().to_string()}")

    # 3. Ensure `new_df` only contains IDs not already in `current` (critical for concat)
    overlap = new_df.index.intersection(current.index)
    if not overlap.empty:
        logger.error(f"Logic error: EMP_IDs in `new_df` also in `current` after deduplication: {overlap.tolist()}. Removing from `new_df`.")
        new_df = new_df[~new_df.index.isin(current.index)]
    # --- End Pre-concat Checks ---

    # --- VERBOSE DEBUGGING PRE-CONCAT ---
    logger.info("--- Pre-concat diagnostics for _apply_new_hires ---")
    logger.info(f"current.index.is_unique: {current.index.is_unique}")
    logger.info(f"current.index.hasnans: {current.index.hasnans}")
    if len(current.index) < 20:
        logger.info(f"current.index values: {current.index.tolist()}")
    logger.info(f"new_df.index.is_unique: {new_df.index.is_unique}")
    logger.info(f"new_df.index.hasnans: {new_df.index.hasnans}")
    if len(new_df.index) < 20:
        logger.info(f"new_df.index values: {new_df.index.tolist()}")

    final_overlap_check = new_df.index.intersection(current.index)
    logger.info(f"Final overlap check (new_df.index.intersection(current.index)): {final_overlap_check.tolist()}")
    if not final_overlap_check.empty:
        logger.error("CRITICAL DEBUG: Overlap detected IMMEDIATELY before concat despite prior filtering!")
    logger.info("--- End Pre-concat diagnostics ---")
    logger.debug(
        f"SnapshotUpdate P4 (before concat): new_df job_level_source unique: "
        f"{new_df[EMP_LEVEL_SOURCE].unique().tolist() if EMP_LEVEL_SOURCE in new_df.columns else 'COLUMN_MISSING'}"
    )
    if EMP_LEVEL_SOURCE in new_df.columns and not new_df.empty:
        logger.debug(f"SnapshotUpdate P4 new_df head:\n{new_df[[EMP_ID, EMP_HIRE_DATE, EMP_LEVEL_SOURCE]].head().to_string()}")
    elif not new_df.empty:
        logger.debug(f"SnapshotUpdate P4 new_df head (EMP_LEVEL_SOURCE missing):\n{new_df[[EMP_ID, EMP_HIRE_DATE]].head().to_string()}")
    # --- END VERBOSE DEBUGGING ---

    parts = _nonempty_frames([current, new_df])
    # Filter out empty DataFrames before concatenation to avoid FutureWarning
    non_empty_parts = [df for df in parts if not df.empty]
    if non_empty_parts:
        result = pd.concat(non_empty_parts, verify_integrity=True, copy=False)
    else:
        # All parts are empty, return empty DataFrame with same structure as current
        result = current.iloc[0:0].copy() if not current.empty else new_df.iloc[0:0].copy()

    # Post-concat check (should be redundant if pre-concat checks are thorough, but good safety net)
    if not result.index.is_unique:
        # This case should ideally not be reached if pre-concat checks are correct.
        post_concat_dups = result.index[result.index.duplicated()].unique().tolist()
        logger.error(f"CRITICAL: Duplicate EMP_IDs persisted after concat: {post_concat_dups}. Deduplicating (keep='last'). This indicates a deeper issue.")
        result = result[~result.index.duplicated(keep='last')]
    return result


def _apply_existing_updates(current: pd.DataFrame, new_events: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Apply updates to existing employees in the current snapshot.

    Args:
        current: Current snapshot DataFrame
        new_events: DataFrame containing update events
        year: Current simulation year

    Returns:
        Updated DataFrame with applied changes
    """
    logger.debug("Starting _apply_existing_updates for year %d", year)

    # Make a copy to avoid modifying the original
    current = current.copy()

    # Update simulation_year for all existing employees
    current[SIMULATION_YEAR] = year
    logger.debug("Updated simulation_year=%d for %d existing employees", year, len(current))

    # Compensation updates - process FIRST (merit raises set base compensation)
    comp_upd = new_events[new_events["event_type"] == EVT_COMP]
    if not comp_upd.empty:
        last_comp = comp_upd.sort_values("event_time").groupby(EMP_ID).tail(1)
        # Match using EMP_ID column instead of index (handles both indexed and column-based snapshots)
        if EMP_ID in current.columns:
            valid_emp_ids = last_comp[last_comp[EMP_ID].isin(current[EMP_ID])][EMP_ID]
            if not valid_emp_ids.empty:
                # Update using boolean indexing to handle integer indices
                for emp_id in valid_emp_ids:
                    comp_value = last_comp[last_comp[EMP_ID] == emp_id]["value_num"].iloc[0]
                    current.loc[current[EMP_ID] == emp_id, EMP_GROSS_COMP] = comp_value
                logger.debug("Updated compensation for %d employees", len(valid_emp_ids))
        else:
            # Fallback to index-based matching for legacy snapshots
            valid_emp_ids = last_comp[last_comp[EMP_ID].isin(current.index)][EMP_ID]
            if not valid_emp_ids.empty:
                current.loc[valid_emp_ids, EMP_GROSS_COMP] = last_comp.set_index(EMP_ID).loc[valid_emp_ids, "value_num"]
                logger.debug("Updated compensation for %d employees", len(valid_emp_ids))

    # COLA updates - apply the COLA amount to the current compensation AFTER merit raises (DEFENSIVE)
    cola_upd = new_events[new_events["event_type"] == EVT_COLA]
    if not cola_upd.empty:
        # Group by employee and get the last COLA for each
        last_cola = cola_upd.sort_values("event_time").groupby(EMP_ID).tail(1)
        # Add the COLA amount to the current compensation (DEFENSIVE CALCULATION)
        updated_count = 0
        
        # Handle both column-based and index-based snapshots
        if EMP_ID in current.columns:
            # Use column-based matching for snapshots with integer indices
            for _, row in last_cola.iterrows():
                emp_id = row[EMP_ID]
                matching_rows = current[current[EMP_ID] == emp_id]
                if not matching_rows.empty:
                    # DEFENSIVE: Always read the most current compensation value
                    current_comp = float(matching_rows[EMP_GROSS_COMP].iloc[0])
                    cola_amount = float(row["value_num"])
                    new_comp = current_comp + cola_amount
                    current.loc[current[EMP_ID] == emp_id, EMP_GROSS_COMP] = new_comp
                    logger.debug(f"Applied COLA {cola_amount:.2f} to current comp {current_comp:.2f} → {new_comp:.2f} for {emp_id}")
                    updated_count += 1
        else:
            # Fallback to index-based matching for legacy snapshots
            for emp_id, row in last_cola.set_index(EMP_ID).iterrows():
                if emp_id in current.index:
                    # DEFENSIVE: Always read the most current compensation value
                    current_comp = float(current.at[emp_id, EMP_GROSS_COMP])
                    cola_amount = float(row["value_num"])
                    new_comp = current_comp + cola_amount
                    current.at[emp_id, EMP_GROSS_COMP] = new_comp
                    logger.debug(f"Applied COLA {cola_amount:.2f} to current comp {current_comp:.2f} → {new_comp:.2f} for {emp_id}")
                    updated_count += 1
                    
        if updated_count > 0:
            logger.debug("Applied COLA updates to %d employees", updated_count)

    # Raise updates - apply raise amount to current compensation (DEFENSIVE CALCULATION)
    raise_upd = new_events[new_events["event_type"] == EVT_RAISE]
    if not raise_upd.empty:
        updated_count = 0
        for _, row in raise_upd.iterrows():
            emp_id = row[EMP_ID]
            if emp_id in current.index:
                try:
                    # DEFENSIVE: Always read the most current compensation value
                    current_comp = float(current.at[emp_id, EMP_GROSS_COMP])

                    # Try to get raise amount from value_num first
                    if pd.notna(row["value_num"]):
                        # If value_num is available, use it directly
                        current.at[emp_id, EMP_GROSS_COMP] = current_comp + float(row["value_num"])
                        updated_count += 1
                    elif pd.notna(row["value_json"]):
                        # Fall back to value_json if value_num is not available
                        import json
                        raise_data = json.loads(row["value_json"])
                        if "new_comp" in raise_data:
                            # If new_comp is provided, use it directly (but this is from stale data)
                            # DEFENSIVE: Apply the raise percentage to current compensation instead
                            if "raise_pct" in raise_data:
                                raise_pct = float(raise_data["raise_pct"])
                                new_comp = current_comp * (1 + raise_pct)
                                current.at[emp_id, EMP_GROSS_COMP] = new_comp
                                logger.debug(f"Applied {raise_pct:.1%} raise to current comp {current_comp:.2f} → {new_comp:.2f} for {emp_id}")
                            else:
                                # Fallback to provided new_comp if no percentage available
                                current.at[emp_id, EMP_GROSS_COMP] = float(raise_data["new_comp"])
                            updated_count += 1
                        elif "amount" in raise_data:
                            # Otherwise, add the raise amount to current comp
                            current.at[emp_id, EMP_GROSS_COMP] = current_comp + float(raise_data["amount"])
                            updated_count += 1
                        elif "raise_pct" in raise_data:
                            # DEFENSIVE: Apply percentage to current compensation
                            raise_pct = float(raise_data["raise_pct"])
                            new_comp = current_comp * (1 + raise_pct)
                            current.at[emp_id, EMP_GROSS_COMP] = new_comp
                            logger.debug(f"Applied {raise_pct:.1%} raise to current comp {current_comp:.2f} → {new_comp:.2f} for {emp_id}")
                            updated_count += 1
                except (json.JSONDecodeError, (KeyError, ValueError, TypeError)) as e:
                    logger.warning("Error processing raise event for %s: %s", emp_id, str(e))
        if updated_count > 0:
            logger.debug("Processed raise events for %d employees", updated_count)

    # Promotion updates - handle employee level changes
    promo_upd = new_events[new_events["event_type"] == EVT_PROMOTION]
    if not promo_upd.empty:
        import json
        updated_count = 0
        for _, row in promo_upd.iterrows():
            emp_id = row[EMP_ID]
            if emp_id in current.index:
                # Extract to_level from the value_json
                if not pd.isna(row["value_json"]):
                    try:
                        promo_data = json.loads(row["value_json"])
                        to_level = promo_data.get("to_level")
                        if to_level is not None:
                            # Update the employee level
                            current.at[emp_id, EMP_LEVEL] = to_level
                            # Update level source if the column exists
                            if EMP_LEVEL_SOURCE in current.columns:
                                # Handle categorical column
                                if pd.api.types.is_categorical_dtype(current[EMP_LEVEL_SOURCE]):
                                    if 'promotion' not in current[EMP_LEVEL_SOURCE].cat.categories:
                                        current[EMP_LEVEL_SOURCE] = current[EMP_LEVEL_SOURCE].cat.add_categories(['promotion'])
                                current.at[emp_id, EMP_LEVEL_SOURCE] = 'promotion'
                            updated_count += 1
                            logger.debug("Promoted employee %s to level %s", emp_id, to_level)
                    except (json.JSONDecodeError, AttributeError) as e:
                        logger.warning("Error processing promotion event for %s: %s", emp_id, str(e))
        if updated_count > 0:
            logger.info("Processed %d promotion events", updated_count)

    # Termination updates
    term_upd = new_events[new_events["event_type"].isin([EVT_TERM, EVT_NEW_HIRE_TERM])]
    if not term_upd.empty:
        last_term = term_upd.sort_values("event_time").groupby(EMP_ID).tail(1)
        # Only update employees that exist in current
        valid_emp_ids = last_term[last_term[EMP_ID].isin(current.index)][EMP_ID]
        if not valid_emp_ids.empty:
            current.loc[valid_emp_ids, EMP_TERM_DATE] = last_term.set_index(EMP_ID).loc[valid_emp_ids, "event_time"]
            current["active"] = current[EMP_TERM_DATE].isna()
            logger.info("Processed termination events for %d employees", len(valid_emp_ids))

    # Update tenure for all employees
    as_of = pd.Timestamp(f"{year}-12-31")
    current = apply_tenure(current, EMP_HIRE_DATE, as_of, out_tenure_col=EMP_TENURE, out_band_col="tenure_band")

    # Ensure all required columns exist and have correct types
    current = ensure_columns_and_types(current)

    logger.debug("Completed _apply_existing_updates for year %d", year)
    return current

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def update(prev_snapshot: pd.DataFrame, new_events: pd.DataFrame, snapshot_year: int) -> pd.DataFrame:  # noqa: D401
    """
    Return a new snapshot by applying new_events to prev_snapshot.
    
    ENHANCED VERSION: Includes comprehensive data integrity checks and robust error handling
    to prevent employee record loss during snapshot updates.

    Args:
        prev_snapshot: Previous snapshot DataFrame
        new_events: DataFrame containing new events to apply
        snapshot_year: Current simulation year

    Returns:
        New snapshot DataFrame with updates applied
    """
    logger.debug("Starting snapshot update for year %d", snapshot_year)
    
    # INTEGRITY CHECK: Validate input snapshot
    initial_employee_count = len(prev_snapshot)
    initial_employee_ids = set(prev_snapshot.index) if not prev_snapshot.empty else set()
    
    logger.debug(f"[SNAPSHOT INTEGRITY] Starting with {initial_employee_count} employees")
    
    # Validate employee IDs in input
    if not prev_snapshot.empty and EMP_ID in prev_snapshot.columns:
        na_ids = prev_snapshot[EMP_ID].isna().sum()
        duplicate_ids = prev_snapshot[EMP_ID].duplicated().sum()
        if na_ids > 0:
            logger.warning(f"[SNAPSHOT INTEGRITY] Input snapshot has {na_ids} rows with NA employee IDs")
        if duplicate_ids > 0:
            logger.warning(f"[SNAPSHOT INTEGRITY] Input snapshot has {duplicate_ids} duplicate employee IDs")

    # Handle empty events case efficiently
    if new_events.empty:
        logger.debug("No new events to apply, returning copy of previous snapshot")
        snapshot = prev_snapshot.copy()
        snapshot[SIMULATION_YEAR] = snapshot_year
        
        # Ensure all required columns exist before type conversion
        snapshot = _ensure_required_columns_safe(snapshot, logger)
        
        # Apply type conversion safely
        result = _apply_dtypes_safe(snapshot, logger)
        
        logger.debug(f"[SNAPSHOT INTEGRITY] Returning {len(result)} employees (no events case)")
        return result

    # Ensure we don't modify the input
    cur = prev_snapshot.copy()

    # Sort events by time for consistent processing (ensures COLA is processed last)
    # With distinct timestamps: Promotion (00:00:30) → Merit (00:01) → COLA (00:02)
    new_events = new_events.sort_values(["event_time"], ascending=[True])

    # Process new hires and updates with enhanced error handling
    try:
        # STEP 1: Process new hires first
        logger.debug(f"[SNAPSHOT INTEGRITY] Before new hires: {len(cur)} employees")
        cur = _apply_new_hires(cur, new_events, snapshot_year)
        logger.debug(f"[SNAPSHOT INTEGRITY] After new hires: {len(cur)} employees")

        # STEP 2: Process updates to existing employees
        logger.debug(f"[SNAPSHOT INTEGRITY] Before existing updates: {len(cur)} employees")
        cur = _apply_existing_updates(cur, new_events, snapshot_year)
        logger.debug(f"[SNAPSHOT INTEGRITY] After existing updates: {len(cur)} employees")

        # STEP 3: ROBUST Index and ID management
        if cur.empty:
            logger.warning("[SNAPSHOT INTEGRITY] DataFrame became empty after processing events!")
            # Return minimal valid snapshot
            return _create_empty_snapshot_with_schema(snapshot_year)
        
        # Ensure EMP_ID column exists and is properly set
        if EMP_ID not in cur.columns:
            logger.warning(f"[SNAPSHOT INTEGRITY] {EMP_ID} column missing, reconstructing from index")
            cur[EMP_ID] = cur.index.astype(str)
        
        # INTEGRITY CHECK: Validate no employee IDs are lost during index operations
        pre_index_ids = set(cur[EMP_ID].dropna())

        # Set index safely - FIXED: Don't overwrite EMP_ID column with index values
        try:
            # Ensure the index is set to employee IDs without overwriting the EMP_ID column
            if cur.index.name != EMP_ID:
                cur = cur.set_index(EMP_ID, drop=False)
                cur.index.name = EMP_ID
            # Ensure EMP_ID column matches the index
            cur[EMP_ID] = cur.index.astype(str)
        except Exception as e:
            logger.error(f"[SNAPSHOT INTEGRITY] Error during index operations: {e}")
            # Fallback: ensure index is set to employee IDs
            if EMP_ID in cur.columns:
                cur = cur.set_index(EMP_ID, drop=False)
                cur.index.name = EMP_ID

        post_index_ids = set(cur[EMP_ID].dropna())
        lost_during_index = pre_index_ids - post_index_ids
        if lost_during_index:
            logger.error(f"[SNAPSHOT INTEGRITY] Lost {len(lost_during_index)} employees during index operations: {list(lost_during_index)[:5]}")

        # STEP 4: SAFE column and type management
        logger.debug(f"[SNAPSHOT INTEGRITY] Before column/type operations: {len(cur)} employees")
        
        # Ensure all required columns exist with appropriate defaults
        cur = _ensure_required_columns_safe(cur, logger)
        
        # INTEGRITY CHECK: Verify no employees lost during column operations
        if len(cur) != len(post_index_ids):
            logger.warning(f"[SNAPSHOT INTEGRITY] Employee count changed during column operations: {len(post_index_ids)} → {len(cur)}")

        # STEP 5: SAFE type conversion and column selection
        result = _apply_dtypes_safe(cur, logger)
        
        # FINAL INTEGRITY CHECK
        final_employee_count = len(result)
        final_employee_ids = set(result.index) if not result.empty else set()
        
        if final_employee_count != initial_employee_count:
            net_change = final_employee_count - initial_employee_count
            lost_ids = initial_employee_ids - final_employee_ids
            gained_ids = final_employee_ids - initial_employee_ids
            
            logger.warning(f"[SNAPSHOT INTEGRITY] Employee count changed: {initial_employee_count} → {final_employee_count} (net: {net_change:+d})")
            if lost_ids:
                logger.warning(f"[SNAPSHOT INTEGRITY] Lost employee IDs: {sorted(list(lost_ids))[:10]}{'...' if len(lost_ids) > 10 else ''}")
            if gained_ids:
                logger.info(f"[SNAPSHOT INTEGRITY] Gained employee IDs: {sorted(list(gained_ids))[:10]}{'...' if len(gained_ids) > 10 else ''}")

        logger.debug(f"[SNAPSHOT INTEGRITY] Successfully updated snapshot for year {snapshot_year}. Final shape: {result.shape}")
        return result

    except Exception as e:
        logger.error(f"[SNAPSHOT INTEGRITY] Error updating snapshot for year {snapshot_year}: {e}", exc_info=True)
        # Log current state for debugging
        logger.error(f"[SNAPSHOT INTEGRITY] Current DataFrame shape: {cur.shape if 'cur' in locals() else 'undefined'}")
        logger.error(f"[SNAPSHOT INTEGRITY] Current columns: {cur.columns.tolist() if 'cur' in locals() and hasattr(cur, 'columns') else 'undefined'}")
        raise


def _ensure_required_columns_safe(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Safely ensure all required columns exist in the DataFrame with appropriate defaults.
    
    Args:
        df: DataFrame to process
        logger: Logger for diagnostics
        
    Returns:
        DataFrame with all required columns
    """
    missing_columns = []
    for col in SNAPSHOT_COLS:
        if col not in df.columns:
            missing_columns.append(col)
            # Set appropriate default value based on column type
            if col in SNAPSHOT_DTYPES:
                default_val = pd.NA
                # Some columns have specific defaults
                if col == EMP_ACTIVE:
                    default_val = True  # Default to active
                elif col == EMP_EXITED:
                    default_val = False  # Default to not exited
                elif col in [EMP_CONTR, EMPLOYER_CORE, EMPLOYER_MATCH]:
                    default_val = 0.0  # Default monetary values to 0
                elif col == IS_ELIGIBLE:
                    default_val = False  # Default to not eligible
                
                df[col] = default_val
    
    if missing_columns:
        logger.debug(f"[COLUMN SAFETY] Added {len(missing_columns)} missing columns: {missing_columns}")
    
    return df


def _apply_dtypes_safe(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Safely apply data types and column selection with comprehensive error handling.
    
    Args:
        df: DataFrame to process
        logger: Logger for diagnostics
        
    Returns:
        DataFrame with correct types and columns
    """
    try:
        # First, select only required columns (but don't drop rows)
        available_cols = [col for col in SNAPSHOT_COLS if col in df.columns]
        missing_cols = [col for col in SNAPSHOT_COLS if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"[DTYPE SAFETY] Missing columns during selection: {missing_cols}")
        
        # Select available columns
        result = df[available_cols].copy()
        
        # Apply data types column by column to avoid dropping rows on single failures
        conversion_errors = []
        for col in available_cols:
            if col in SNAPSHOT_DTYPES:
                try:
                    result[col] = result[col].astype(SNAPSHOT_DTYPES[col])
                except Exception as e:
                    conversion_errors.append((col, str(e)))
                    logger.warning(f"[DTYPE SAFETY] Failed to convert column {col}: {e}")
                    # Keep the column as-is rather than dropping rows
        
        if conversion_errors:
            logger.warning(f"[DTYPE SAFETY] {len(conversion_errors)} columns had type conversion issues")
        
        # Ensure index name is set
        result.index.name = EMP_ID
        
        return result
        
    except Exception as e:
        logger.error(f"[DTYPE SAFETY] Critical error during dtype application: {e}")
        # Return DataFrame as-is rather than failing completely
        df.index.name = EMP_ID
        return df


def _create_empty_snapshot_with_schema(snapshot_year: int) -> pd.DataFrame:
    """
    Create an empty snapshot DataFrame with proper schema.
    
    Args:
        snapshot_year: Year for the snapshot
        
    Returns:
        Empty DataFrame with correct schema
    """
    # Create empty DataFrame with required columns
    empty_data = {col: pd.Series([], dtype=SNAPSHOT_DTYPES.get(col, 'object')) for col in SNAPSHOT_COLS}
    empty_data[SIMULATION_YEAR] = pd.Series([], dtype='int64')
    
    result = pd.DataFrame(empty_data)
    result.index.name = EMP_ID
    
    return result
