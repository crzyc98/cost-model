# cost_model/engines/promotion.py
"""
Engine for simulating employee promotions and associated merit raises during workforce simulations.
QuickStart: see docs/cost_model/engines/promotion.md
"""

import pandas as pd
import json
from typing import List
from cost_model.state.event_log import EVENT_COLS, EVT_PROMOTION, EVT_RAISE, create_event
from cost_model.state.schema import EMP_ID, EMP_ROLE, EMP_GROSS_COMP

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

    # Create promotion events
    promotions = []
    raises = []
    
    for _, row in promotees.iterrows():
        # Create promotion event
        promo_event = create_event(
            event_time=promo_time,
            employee_id=row[EMP_ID],
            event_type=EVT_PROMOTION,
            value_num=0.0,  # No numeric value for promotion events
            value_json=json.dumps({"new_role": next_title_map.get(row[EMP_ROLE], f"Senior {row[EMP_ROLE]}")}),
            meta="Promotion based on eligibility"
        )
        promotions.append(promo_event)
        
        # Create raise event
        raise_amount = row[EMP_GROSS_COMP] * merit_pct
        raise_event = create_event(
            event_time=raise_time,
            employee_id=row[EMP_ID],
            event_type=EVT_RAISE,
            value_num=raise_amount,
            value_json=json.dumps({"merit_pct": merit_pct, "previous_comp": row[EMP_GROSS_COMP]}),
            meta="Merit raise concurrent with promotion"
        )
        raises.append(raise_event)
    
    # Convert to DataFrames with proper schema
    promotions_df = pd.DataFrame(promotions, columns=EVENT_COLS) if promotions else pd.DataFrame(columns=EVENT_COLS)
    raises_df = pd.DataFrame(raises, columns=EVENT_COLS) if raises else pd.DataFrame(columns=EVENT_COLS)
    return [promotions, raises]
