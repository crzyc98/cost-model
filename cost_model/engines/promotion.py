# cost_model/engines/promotion.py
"""
Engine for simulating employee promotions and associated merit raises during workforce simulations.
QuickStart: see docs/cost_model/engines/promotion.md
"""

import pandas as pd
import json
from typing import List
from cost_model.state.event_log import EVENT_COLS, EVT_PROMOTION, EVT_RAISE, create_event
from cost_model.utils.columns import EMP_ID, EMP_ROLE, EMP_GROSS_COMP

def promote(
    snapshot: pd.DataFrame,
    rules: dict,
    promo_time: pd.Timestamp,
    raise_time: pd.Timestamp = None
) -> List[pd.DataFrame]:
    """
    Vectorized promotion and merit-raise event emission.
    - Uses rules dict for role hierarchy (rules['next_title']) and merit pct (rules['merit_pct']).
    - Allows separate timestamps for promotions and raises (raise_time defaults to promo_time).
    - Returns [promotions_df, raises_df] (both EVENT_COLS-compliant).
    """
    if "eligible_for_promotion" not in snapshot.columns:
        # If the column is missing, no one is eligible
        return [pd.DataFrame(columns=EVENT_COLS), pd.DataFrame(columns=EVENT_COLS)]
    promotees = snapshot[snapshot["eligible_for_promotion"] == True].copy()
    if promotees.empty:
        return [pd.DataFrame(columns=EVENT_COLS), pd.DataFrame(columns=EVENT_COLS)]

    # Promotion event
    next_title_map = rules.get("next_title", {})
    merit_pct = rules.get("merit_pct", 0.10)
    raise_time = raise_time or promo_time

    # Build new titles
    new_titles = promotees[EMP_ROLE].map(next_title_map).fillna("Senior " + promotees[EMP_ROLE].astype(str))

    promotions = pd.DataFrame({
        "event_time": promo_time,
        EMP_ID: promotees[EMP_ID],
        "event_type": EVT_PROMOTION,
        "value_num": 0.0,  # No numeric value for promotion events
        "value_json": "{}",  # Empty JSON for compatibility
        "meta": "Promotion based on eligibility"
    })
    # Merit raises
    raises = pd.DataFrame({
        "event_time": raise_time,
        EMP_ID: promotees[EMP_ID],
        "event_type": EVT_RAISE,
        "value_num": promotees[EMP_GROSS_COMP] * merit_pct,
        "value_json": "{}",  # Empty JSON for compatibility
        "meta": "Merit raise concurrent with promotion"
    })
    # Assign event_id if needed, slice to schema
    for df in (promotions, raises):
        if "event_id" not in df:
            df["event_id"] = df.index.map(lambda i: f"evt_{i:06d}")
    promotions = promotions[EVENT_COLS]
    raises = raises[EVENT_COLS]
    return [promotions, raises]
