# cost_model/plan_rules/contribution_increase.py
"""
Emission of contribution increase events when an employee raises their deferral rate
beyond a configured threshold.
QuickStart: see docs/cost_model/plan_rules/contribution_increase.md
"""
import uuid
import json
from typing import List
from datetime import datetime

import pandas as pd

from cost_model.config.plan_rules import ContributionIncreaseConfig
from cost_model.utils.columns import EMP_ID, EMP_DEFERRAL_RATE

# canonical event schema
EVENT_COLS = [
    "event_id", "event_time", EMP_ID,
    "event_type", "value_num", "value_json", "meta"
]
EVENT_DTYPES = {
    "event_id": "string",
    "event_time": "datetime64[ns]",
    EMP_ID: "string",
    "event_type": "string",
    "value_num": "float64",
    "value_json": "string",
    "meta": "string",
}

# default prior events that set an initial deferral rate
DEFAULT_DEFERRAL_EVENT_TYPES = {"EVT_ENROLL", "EVT_AUTO_INCREASE"}


from cost_model.utils.columns import EMP_TENURE

def run(
    snapshot: pd.DataFrame,
    events: pd.DataFrame,
    as_of: pd.Timestamp,
    cfg: ContributionIncreaseConfig
) -> List[pd.DataFrame]:
    """
    Identify employees whose current deferral rate has increased by at least
    `cfg.min_increase_pct` compared to their most recent prior deferral election
    event (e.g., EVT_ENROLL or EVT_AUTO_INCREASE) on or before `as_of`.

    Emits one event per employee whose increase meets or exceeds the threshold.

    Args:
        snapshot: DataFrame indexed by EMP_ID containing EMP_DEFERRAL_RATE.
        events: Event log DataFrame with canonical schema (columns: event_id, event_time,
                EMP_ID, event_type, value_num, value_json, meta).
        as_of: Timestamp cutoff for considering prior elections.
        cfg: ContributionIncreaseConfig with fields:
            - min_increase_pct: float threshold for rate increase
            - event_type: string to assign to generated events
            - prior_event_types: optional Set[str] overriding default event types

    Returns:
        A list containing a single DataFrame of increase events (or an empty list).
    """
    # 0) Check if config is present and enabled
    if not cfg or not getattr(cfg, 'enabled', True): # Default to enabled if 'enabled' attr is missing
        # import logging # Add this import if logging is desired here
        # logger = logging.getLogger(__name__)
        # logger.debug("[ContributionIncrease] rule skipped, config missing or disabled.")
        return []

    # Ensure snapshot is indexed by EMP_ID
    current_snapshot = snapshot.copy()
    if EMP_ID in current_snapshot.columns and current_snapshot.index.name != EMP_ID:
        current_snapshot = current_snapshot.set_index(EMP_ID, drop=False)
    elif current_snapshot.index.name != EMP_ID:
        # This implies EMP_ID is not a column and not the index name, which is an issue.
        if EMP_ID not in current_snapshot.columns:
             raise ValueError(f"{EMP_ID} column not found in snapshot and not set as index for contribution_increase.")

    # Check for necessary columns
    if EMP_DEFERRAL_RATE not in current_snapshot.columns:
        # Or log warning and return empty
        return []
    if not {EMP_ID, "event_type", "event_time", "value_num"}.issubset(events.columns):
        # Or log warning and return empty
        return []

    rows = []
    # Determine which event types count as prior deferral settings
    prior_types = getattr(cfg, 'prior_event_types', DEFAULT_DEFERRAL_EVENT_TYPES)
    
    # Prepare event dates once
    event_dates_col = events['event_time'].dt.date
    as_of_date_val = as_of.date()

    # Iterate over each employee's current rate
    for emp, row_data in current_snapshot.iterrows(): # emp is EMP_ID from index
        new_rate = row_data.get(EMP_DEFERRAL_RATE)
        if new_rate is None or pd.isna(new_rate):
            continue
            
        # Filter prior deferral events for this employee up to as_of
        mask = (
            events['event_type'].isin(prior_types) &
            (events[EMP_ID].astype(str) == str(emp)) &
            (event_dates_col <= as_of_date_val)
        )
        prior = events.loc[mask]

        # Determine the previous rate
        if prior.empty:
            old_rate = 0.0
        else:
            last = prior.sort_values('event_time').iloc[-1]
            old_rate = last.get('value_num') or 0.0

        # Compute delta and compare to threshold
        delta = new_rate - old_rate
        if delta >= cfg.min_increase_pct:
            payload = {'old_rate': old_rate, 'new_rate': new_rate, 'delta': delta}
            rows.append({
                'event_id': str(uuid.uuid4()),
                'event_time': as_of,
                EMP_ID: emp,
                'event_type': cfg.event_type,
                'value_num': None, # Or new_rate, depending on convention for this event_type
                'value_json': json.dumps(payload),
                'meta': None,
            })

    # Return empty list if no rows, else a single DataFrame
    if not rows:
        return []
    df = pd.DataFrame(rows, columns=EVENT_COLS).astype(EVENT_DTYPES)
    return [df]
