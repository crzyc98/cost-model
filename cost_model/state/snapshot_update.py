"""Incrementally update a workforce snapshot given new events.

## QuickStart

To update an existing workforce snapshot with new events programmatically:

```python
import pandas as pd
import logging
from cost_model.state.snapshot_update import update
from cost_model.state.event_log import EVT_HIRE, EVT_TERM, EVT_COMP

# Configure logging to see detailed information about the update process
logging.basicConfig(level=logging.DEBUG)

# Load an existing snapshot
prev_snapshot = pd.read_parquet('data/snapshot_2024.parquet')

# Create or load new events that occurred since the previous snapshot
new_events = pd.DataFrame([
    # New hire event
    {
        'event_id': 'evt_001',
        'event_time': pd.Timestamp('2025-01-15'),
        'employee_id': 'EMP101',
        'event_type': EVT_HIRE,
        'value_num': 60000.0,  # Starting compensation
        'value_json': None,
        'meta': 'New hire - Software Engineer'
    },
    # Termination event for existing employee
    {
        'event_id': 'evt_002',
        'event_time': pd.Timestamp('2025-03-31'),
        'employee_id': 'EMP050',  # Existing employee ID
        'event_type': EVT_TERM,
        'value_num': None,
        'value_json': None,
        'meta': 'Voluntary resignation'
    },
    # Compensation change for existing employee
    {
        'event_id': 'evt_003',
        'event_time': pd.Timestamp('2025-04-01'),
        'employee_id': 'EMP025',  # Existing employee ID
        'event_type': EVT_COMP,
        'value_num': 85000.0,  # New compensation
        'value_json': None,
        'meta': 'Annual merit increase'
    }
])

# Update the snapshot for the new year
snapshot_year = 2025
updated_snapshot = update(prev_snapshot, new_events, snapshot_year)

# Examine the results
print(f"Updated snapshot has {len(updated_snapshot)} employees")
print(f"Active employees: {updated_snapshot['active'].sum()}")

# Check for the new hire
if 'EMP101' in updated_snapshot.index:
    print(f"New hire EMP101 added with compensation: {updated_snapshot.loc['EMP101', 'employee_gross_compensation']}")

# Check for the terminated employee
if 'EMP050' in updated_snapshot.index:
    is_active = updated_snapshot.loc['EMP050', 'active']
    term_date = updated_snapshot.loc['EMP050', 'employee_termination_date']
    print(f"EMP050 active status: {is_active}, termination date: {term_date}")

# Save the updated snapshot
updated_snapshot.to_parquet(f'data/snapshot_{snapshot_year}.parquet')
```

This demonstrates how to update a snapshot with various event types and verify the results.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

from .schema import (
    EVT_HIRE,
    EVT_COMP,
    EVT_TERM,
    EMP_ID,
    EMP_HIRE_DATE,
    EMP_BIRTH_DATE,
    EMP_GROSS_COMP,
    EMP_TERM_DATE,
    EMP_TENURE,
    SNAPSHOT_COLS,
    SNAPSHOT_DTYPES,
)
from .snapshot_utils import (
    get_first_event,
    get_last_event,
    extract_hire_details,
    ensure_columns_and_types,
)
from .tenure import apply_tenure

logger = logging.getLogger(__name__)

__all__: List[str] = ["update"]

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _apply_new_hires(current: pd.DataFrame, new_events: pd.DataFrame, year: int) -> pd.DataFrame:
    hires = get_first_event(new_events, EVT_HIRE)
    new_ids = hires[~hires[EMP_ID].isin(current.index)][EMP_ID].unique()
    if len(new_ids) == 0:
        return current

    logger.debug("%d new hires to append", len(new_ids))
    batch = new_events[new_events[EMP_ID].isin(new_ids)]
    first_hire = get_first_event(batch, EVT_HIRE)
    details = extract_hire_details(first_hire)

    last_comp = get_last_event(batch, EVT_COMP).set_index(EMP_ID)["value_num"].rename(EMP_GROSS_COMP)
    last_term = get_last_event(batch, EVT_TERM).set_index(EMP_ID)["event_time"].rename(EMP_TERM_DATE)

    new_df = pd.DataFrame(index=pd.Index(new_ids, name=EMP_ID))
    new_df = new_df.merge(first_hire.set_index(EMP_ID)["event_time"].rename(EMP_HIRE_DATE), left_index=True, right_index=True, how="left")
    new_df = new_df.merge(details, left_index=True, right_index=True, how="left")
    new_df = new_df.merge(last_comp, left_index=True, right_index=True, how="left")
    new_df = new_df.merge(last_term, left_index=True, right_index=True, how="left")
    new_df["active"] = new_df[EMP_TERM_DATE].isna()

    # Debug: Verify birth date types before ensure_columns_and_types
    if not pd.api.types.is_datetime64_any_dtype(new_df[EMP_BIRTH_DATE]):
        logger.warning(f"Birth dates not in datetime format: {new_df[EMP_BIRTH_DATE].dtype}")
        new_df[EMP_BIRTH_DATE] = pd.to_datetime(new_df[EMP_BIRTH_DATE], errors='coerce')
    
    new_df = ensure_columns_and_types(new_df)

    as_of = pd.Timestamp(f"{year}-12-31")
    new_df = apply_tenure(new_df, EMP_HIRE_DATE, as_of, out_tenure_col=EMP_TENURE, out_band_col="tenure_band")

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
    # --- END VERBOSE DEBUGGING ---

    result = pd.concat([current, new_df], verify_integrity=True, copy=False) # verify_integrity can now be True
    
    # Post-concat check (should be redundant if pre-checks are thorough, but good safety net)
    if not result.index.is_unique:
        # This case should ideally not be reached if pre-concat checks are correct.
        post_concat_dups = result.index[result.index.duplicated()].unique().tolist()
        logger.error(f"CRITICAL: Duplicate EMP_IDs persisted after concat: {post_concat_dups}. Deduplicating (keep='last'). This indicates a deeper issue.")
        result = result[~result.index.duplicated(keep='last')]
    return result


def _apply_existing_updates(current: pd.DataFrame, new_events: pd.DataFrame, year: int) -> pd.DataFrame:
    # Compensation updates
    comp_upd = new_events[new_events["event_type"] == EVT_COMP]
    if not comp_upd.empty:
        last_comp = comp_upd.sort_values("event_time").groupby(EMP_ID).tail(1)
        current.loc[last_comp[EMP_ID], EMP_GROSS_COMP] = last_comp.set_index(EMP_ID)["value_num"]

    # Termination updates
    term_upd = new_events[new_events["event_type"] == EVT_TERM]
    if not term_upd.empty:
        last_term = term_upd.sort_values("event_time").groupby(EMP_ID).tail(1)
        current.loc[last_term[EMP_ID], EMP_TERM_DATE] = last_term.set_index(EMP_ID)["event_time"]
        current["active"] = current[EMP_TERM_DATE].isna()

    as_of = pd.Timestamp(f"{year}-12-31")
    current = apply_tenure(current, EMP_HIRE_DATE, as_of, out_tenure_col=EMP_TENURE, out_band_col="tenure_band")
    return current

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def update(prev_snapshot: pd.DataFrame, new_events: pd.DataFrame, snapshot_year: int) -> pd.DataFrame:  # noqa: D401
    """Return a new snapshot by applying *new_events* to *prev_snapshot*."""
    if new_events.empty:
        return prev_snapshot.astype(SNAPSHOT_DTYPES)

    cur = prev_snapshot.copy()
    new_events = new_events.sort_values(["event_time", "event_type"], ascending=[True, True])

    cur = _apply_new_hires(cur, new_events, snapshot_year)
    cur = _apply_existing_updates(cur, new_events, snapshot_year)

    cur[EMP_ID] = cur.index.astype(str)
    # SNAPSHOT_COLS already contains EMP_TENURE, so avoid duplication
    cur = cur[SNAPSHOT_COLS].astype(SNAPSHOT_DTYPES)
    cur.index.name = EMP_ID
    return cur
