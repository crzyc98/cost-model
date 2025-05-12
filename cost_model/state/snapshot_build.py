"""Build a full employee snapshot from an event log.

## QuickStart

To build a complete workforce snapshot from an event log programmatically:

```python
import pandas as pd
import logging
from pathlib import Path
from cost_model.state.snapshot_build import build_full
from cost_model.state.event_log import EVT_HIRE, EVT_COMP, EVT_TERM

# Configure logging to see detailed information
logging.basicConfig(level=logging.INFO)

# Load or create an event log
events_df = pd.read_parquet('data/events_2025.parquet')

# Alternatively, create a synthetic event log for testing
synth_events = pd.DataFrame([
    # Hire events
    {
        'event_id': 'evt_h1',
        'event_time': pd.Timestamp('2025-01-15'),
        'employee_id': 'EMP001',
        'event_type': EVT_HIRE,
        'value_num': 75000.0,  # Starting compensation
        'value_json': '{"role": "Engineer", "birth_date": "1990-05-12"}',
        'meta': 'Initial hire'
    },
    {
        'event_id': 'evt_h2',
        'event_time': pd.Timestamp('2025-02-01'),
        'employee_id': 'EMP002',
        'event_type': EVT_HIRE,
        'value_num': 85000.0,
        'value_json': '{"role": "Manager", "birth_date": "1985-08-23"}',
        'meta': 'Initial hire'
    },
    # Compensation change event
    {
        'event_id': 'evt_c1',
        'event_time': pd.Timestamp('2025-06-01'),
        'employee_id': 'EMP001',
        'event_type': EVT_COMP,
        'value_num': 80000.0,  # New compensation
        'value_json': None,
        'meta': 'Mid-year adjustment'
    },
    # Termination event
    {
        'event_id': 'evt_t1',
        'event_time': pd.Timestamp('2025-09-30'),
        'employee_id': 'EMP002',
        'event_type': EVT_TERM,
        'value_num': None,
        'value_json': None,
        'meta': 'Voluntary termination'
    }
])

# Build a snapshot for the specified year
snapshot_year = 2025
snapshot_df = build_full(events_df, snapshot_year)

# Examine the snapshot
print(f"Snapshot contains {len(snapshot_df)} employees")
print(f"Active employees: {snapshot_df['active'].sum()}")
print(f"Terminated employees: {(~snapshot_df['active']).sum()}")

# Check tenure calculations
print("\nTenure distribution:")
if 'tenure_band' in snapshot_df.columns:
    print(snapshot_df['tenure_band'].value_counts())

# Save the snapshot
output_dir = Path('output/snapshots')
output_dir.mkdir(parents=True, exist_ok=True)
snapshot_df.to_parquet(output_dir / f'snapshot_{snapshot_year}.parquet')
```

This demonstrates how to build a complete workforce snapshot from an event log, including handling of hire, compensation, and termination events.
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
    EMP_ROLE,
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

    # Tenure & band
    as_of = pd.Timestamp(f"{snapshot_year}-12-31")
    snap = apply_tenure(
        snap,
        hire_col=EMP_HIRE_DATE,
        as_of=as_of,
        out_tenure_col=EMP_TENURE,
        out_band_col="tenure_band",
    )

    snap[EMP_ID] = snap.index.astype(str)

    logger.info("Full snapshot built (shape=%s)", snap.shape)
    # Defensive: ensure unique index before returning
    if not snap.index.is_unique:
        dups = snap.index[snap.index.duplicated()].unique()
        logger.error(f"Duplicate EMP_IDs in snapshot: {dups.tolist()} – dropping duplicates (keeping last)")
        snap = snap[~snap.index.duplicated(keep='last')]
    return snap
