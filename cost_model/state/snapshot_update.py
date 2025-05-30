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
        # Only update employees that exist in current
        valid_emp_ids = last_comp[last_comp[EMP_ID].isin(current.index)][EMP_ID]
        if not valid_emp_ids.empty:
            current.loc[valid_emp_ids, EMP_GROSS_COMP] = last_comp.set_index(EMP_ID).loc[valid_emp_ids, "value_num"]
            logger.debug("Updated compensation for %d employees", len(valid_emp_ids))

    # COLA updates - apply the COLA amount to the current compensation AFTER merit raises
    cola_upd = new_events[new_events["event_type"] == EVT_COLA]
    if not cola_upd.empty:
        # Group by employee and get the last COLA for each
        last_cola = cola_upd.sort_values("event_time").groupby(EMP_ID).tail(1)
        # Add the COLA amount to the current compensation
        updated_count = 0
        for emp_id, row in last_cola.set_index(EMP_ID).iterrows():
            if emp_id in current.index:
                current.at[emp_id, EMP_GROSS_COMP] += row["value_num"]
                updated_count += 1
        if updated_count > 0:
            logger.debug("Applied COLA updates to %d employees", updated_count)

    # Raise updates - apply raise amount to current compensation
    raise_upd = new_events[new_events["event_type"] == EVT_RAISE]
    if not raise_upd.empty:
        updated_count = 0
        for _, row in raise_upd.iterrows():
            emp_id = row[EMP_ID]
            if emp_id in current.index:
                try:
                    # Try to get raise amount from value_num first
                    if pd.notna(row["value_num"]):
                        # If value_num is available, use it directly
                        current.at[emp_id, EMP_GROSS_COMP] += row["value_num"]
                        updated_count += 1
                    elif pd.notna(row["value_json"]):
                        # Fall back to value_json if value_num is not available
                        import json
                        raise_data = json.loads(row["value_json"])
                        if "new_comp" in raise_data:
                            # If new_comp is provided, use it directly
                            current.at[emp_id, EMP_GROSS_COMP] = float(raise_data["new_comp"])
                            updated_count += 1
                        elif "amount" in raise_data:
                            # Otherwise, add the raise amount to current comp
                            current.at[emp_id, EMP_GROSS_COMP] += float(raise_data["amount"])
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

    Args:
        prev_snapshot: Previous snapshot DataFrame
        new_events: DataFrame containing new events to apply
        snapshot_year: Current simulation year

    Returns:
        New snapshot DataFrame with updates applied
    """
    logger.debug("Starting snapshot update for year %d", snapshot_year)

    # Handle empty events case efficiently
    if new_events.empty:
        logger.debug("No new events to apply, returning copy of previous snapshot")
        snapshot = prev_snapshot.copy()
        snapshot[SIMULATION_YEAR] = snapshot_year
        return snapshot.astype(SNAPSHOT_DTYPES)

    # Ensure we don't modify the input
    cur = prev_snapshot.copy()

    # Sort events by time and type for consistent processing
    new_events = new_events.sort_values(["event_time", "event_type"], ascending=[True, True])

    # Process new hires and updates
    try:
        # Process new hires first
        cur = _apply_new_hires(cur, new_events, snapshot_year)

        # Then process updates to existing employees
        cur = _apply_existing_updates(cur, new_events, snapshot_year)

        # Ensure EMP_ID is properly set as string and is the index
        cur[EMP_ID] = cur.index.astype(str)
        cur.index = cur[EMP_ID]  # Ensure index is set to EMP_ID

        # Ensure all required columns exist with correct types
        for col in SNAPSHOT_COLS:
            if col not in cur.columns:
                if col in SNAPSHOT_DTYPES:
                    cur[col] = pd.NA

        # Convert to final types and select only required columns
        result = cur[SNAPSHOT_COLS].astype(SNAPSHOT_DTYPES)
        result.index.name = EMP_ID

        # Handle categorical columns
        from pandas import CategoricalDtype
        if 'job_level_source' in result.columns and isinstance(result['job_level_source'].dtype, CategoricalDtype):
            result['job_level_source'] = result['job_level_source'].astype("category")

        logger.debug("Successfully updated snapshot for year %d. Shape: %s", snapshot_year, str(result.shape))
        return result

    except Exception as e:
        logger.error("Error updating snapshot for year %d: %s", snapshot_year, str(e), exc_info=True)
        raise
