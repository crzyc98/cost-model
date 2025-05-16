# cost_model/state/snapshot_build.py
"""
Build a full employee snapshot from an event log.
QuickStart: see docs/cost_model/state/snapshot_build.md
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
    EMP_LEVEL,
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
from .tenure import assign_tenure_band

logger = logging.getLogger(__name__)

__all__: List[str] = ["build_full"]


def _empty_snapshot() -> pd.DataFrame:
    """Return an empty, correctly typed snapshot DataFrame."""
    df = pd.DataFrame(columns=SNAPSHOT_COLS)
    df = df.astype(SNAPSHOT_DTYPES)
    df.index.name = EMP_ID
    return df


def build_full(events: pd.DataFrame, snapshot_year: int) -> pd.DataFrame:  # noqa: D401
    """Build an end-of-year workforce snapshot from *events*.

    Parameters
    ----------
    events : DataFrame
        Complete event log (any order) conforming to EVENT_COLS.
    snapshot_year : int
        Calendar year whose 31-Dec is the *as-of* date for tenure calcs.
    """
    if events.empty:
        logger.warning("build_full called with empty events DataFrame.")
        return _empty_snapshot()

    logger.info("Building full snapshot from %d events…", len(events))

    events = events.sort_values(["event_time", "event_type"], ascending=[True, True])

    # ------------------------------------------------------------------
    # First hire, last comp, last term per employee
    hires = get_first_event(events, EVT_HIRE)
    if hires.empty:
        logger.warning("No hire events found – returning empty snapshot.")
        return _empty_snapshot()

    hire_details = extract_hire_details(hires)
    hire_dates = hires.set_index(EMP_ID)["event_time"].rename(EMP_HIRE_DATE)

    comps = get_last_event(events, EVT_COMP)
    last_comp = comps.set_index(EMP_ID)["value_num"].rename(EMP_GROSS_COMP)

    terms = get_last_event(events, EVT_TERM)
    last_term = terms.set_index(EMP_ID)["event_time"].rename(EMP_TERM_DATE)

    # ------------------------------------------------------------------
    # Assemble snapshot
    snap = pd.DataFrame(hire_dates)
    snap = snap.merge(hire_details, left_index=True, right_index=True, how="left")
    snap = snap.merge(last_comp, left_index=True, right_index=True, how="left")
    snap = snap.merge(last_term, left_index=True, right_index=True, how="left")

    snap["active"] = snap[EMP_TERM_DATE].isna()

    snap = ensure_columns_and_types(snap)

    # Calculate tenure for active and terminated employees
    as_of = pd.Timestamp(f"{snapshot_year}-12-31")
    
    # Ensure datetime types
    snap[EMP_HIRE_DATE] = pd.to_datetime(snap[EMP_HIRE_DATE], errors='coerce')
    snap[EMP_TERM_DATE] = pd.to_datetime(snap[EMP_TERM_DATE], errors='coerce')
    
    # For active employees, calculate to year-end
    active_mask = snap[EMP_ACTIVE].fillna(False)
    snap.loc[active_mask, EMP_TENURE] = (
        (as_of - snap.loc[active_mask, EMP_HIRE_DATE]).dt.days / 365.25
    ).round(3)
    
    # For terminated employees, calculate to termination date
    term_mask = (~active_mask) & snap[EMP_TERM_DATE].notna()
    snap.loc[term_mask, EMP_TENURE] = (
        (snap.loc[term_mask, EMP_TERM_DATE] - 
         snap.loc[term_mask, EMP_HIRE_DATE]).dt.days / 365.25
    ).round(3)
    
    # Assign tenure bands based on calculated tenure
    snap[EMP_TENURE_BAND] = snap[EMP_TENURE].apply(assign_tenure_band).astype(pd.StringDtype())
    
    # Log any employees with missing or invalid tenure
    missing_tenure = snap[snap[EMP_TENURE].isna()].index.tolist()
    if missing_tenure:
        logger.warning(
            "Could not calculate tenure for %d employees (missing/invalid dates): %s",
            len(missing_tenure),
            missing_tenure[:10]  # Only show first 10 to avoid log spam
        )

    snap[EMP_ID] = snap.index.astype(str)

    logger.info("Full snapshot built (shape=%s)", snap.shape)
    # Defensive: ensure unique index before returning
    if not snap.index.is_unique:
        dups = snap.index[snap.index.duplicated()].unique()
        logger.error(f"Duplicate EMP_IDs in snapshot: {dups.tolist()} – dropping duplicates (keeping last)")
        snap = snap[~snap.index.duplicated(keep='last')]
    return snap
