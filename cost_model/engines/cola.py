# cost_model/engines/cola.py
"""
Engine for simulating cost-of-living adjustments (COLA) during workforce simulations.
QuickStart: see docs/cost_model/engines/cola.md
"""

import json
import logging
from typing import List

import pandas as pd

from cost_model.state.event_log import EVENT_COLS, EVT_COLA, create_event
from cost_model.state.schema import EMP_ID, EMP_TERM_DATE
from cost_model.state.schema import EMP_GROSS_COMP  # to read existing salary

logger = logging.getLogger(__name__)

def cola(
    snapshot: pd.DataFrame,
    hazard_slice: pd.DataFrame,
    as_of: pd.Timestamp,
    days_into_year: int = 0,
    jitter_days: int = 0,
    rng=None
) -> List[pd.DataFrame]:
    """
    Emit a cost-of-living adjustment (EVT_COLA) for each active employee, vectorized.
    - Uses hazard_slice['cola_pct'].iloc[0] (KeyError if missing).
    - Timing: as_of + days_into_year (+ optional jitter_days, if set).
    - If jitter_days > 0, applies uniform random jitter (+/- jitter_days//2) to event_time per employee.
    - Returns a single DataFrame of EVT_COLA events (EVENT_COLS schema).
    """
    year = int(hazard_slice["simulation_year"].iloc[0])
    base_time = pd.to_datetime(as_of) + pd.Timedelta(days=days_into_year)
    cola_pct = float(hazard_slice["cola_pct"].iloc[0])  # KeyError if missing

    # Filter active employees
    mask = snapshot[EMP_TERM_DATE].isna() | (snapshot[EMP_TERM_DATE] > base_time)
    active = snapshot.loc[mask, [EMP_ID, EMP_GROSS_COMP]].copy()
    if cola_pct == 0.0 or active.empty:
        return [pd.DataFrame(columns=EVENT_COLS)]

    # Compute comp changes
    active["old_comp"] = active[EMP_GROSS_COMP].astype(float)
    active["new_comp"] = (active["old_comp"] * (1 + cola_pct)).round(2)
    active["delta"] = active["new_comp"] - active["old_comp"]

    # Optional jitter in timing
    if jitter_days > 0:
        import numpy as np
        if rng is None:
            rng = np.random.default_rng()
        offsets = rng.integers(-jitter_days//2, jitter_days//2 + 1, size=len(active))
        active["event_time"] = [base_time + pd.Timedelta(days=int(d)) for d in offsets]
    else:
        active["event_time"] = base_time

    def mk_event(row):
        return create_event(
            event_time=row["event_time"],
            employee_id=row[EMP_ID],
            event_type=EVT_COLA,
            value_num=row["delta"],
            meta=f"COLA {cola_pct:.1%}"
        )

    events = active.apply(mk_event, axis=1).tolist()
    df = pd.DataFrame(events, columns=EVENT_COLS)
    return [df.sort_values("event_time", ignore_index=True)]