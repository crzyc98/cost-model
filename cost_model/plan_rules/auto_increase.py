import pandas as pd
from cost_model.config.plan_rules import AutoIncreaseConfig
from cost_model.utils.columns import EMP_ID, EMP_DEFERRAL_RATE
from typing import List
import uuid


def run(
    snapshot: pd.DataFrame,
    events: pd.DataFrame,
    as_of: pd.Timestamp,
    cfg: AutoIncreaseConfig,
) -> List[pd.DataFrame]:
    """
    For employees who’ve been enrolled (or are auto-enrolled) and haven’t opted out,
    generate a single EVT_AUTO_INCREASE event at `as_of` that bumps their deferral rate
    by cfg.increase_rate, capped at cfg.cap_rate.
    """
    EVT_ENROLL = "EVT_ENROLL"
    EVT_AUTO_ENROLL = "EVT_AUTO_ENROLL"
    EVT_OPT_OUT = "EVT_OPT_OUT"
    EVT_AUTO_INCREASE = "EVT_AUTO_INCREASE"

    EVENT_COLS = [
        "event_id", "event_time", EMP_ID, "event_type",
        "value_num", "value_json", "meta"
    ]

    current_snapshot = snapshot.copy()
    if EMP_ID in current_snapshot.columns and current_snapshot.index.name != EMP_ID:
        current_snapshot = current_snapshot.set_index(EMP_ID)
    elif current_snapshot.index.name != EMP_ID:
        if EMP_ID not in current_snapshot.columns:
             raise ValueError(f"{EMP_ID} column not found in snapshot and not set as index for auto_increase.")

    if not {EMP_ID, "event_type", "event_time"}.issubset(events.columns):
        return []
        
    event_dates = events["event_time"].dt.date
    as_of_date = as_of.date()
    
    enrolled_ids = set(
        events.loc[
            (events["event_type"].isin([EVT_ENROLL, EVT_AUTO_ENROLL]))
            & (event_dates <= as_of_date),
            EMP_ID,
        ].astype(str)
    )
    opted_out_ids = set(
        events.loc[
            (events["event_type"] == EVT_OPT_OUT) & (event_dates <= as_of_date),
            EMP_ID,
        ].astype(str)
    )
    already_increased_ids = set(
        events.loc[
            (events["event_type"] == EVT_AUTO_INCREASE)
            & (event_dates == as_of_date),
            EMP_ID,
        ].astype(str)
    )
    candidates = enrolled_ids - opted_out_ids - already_increased_ids

    out_events = []
    for emp in candidates:
        if emp not in current_snapshot.index:
            continue
        
        if EMP_DEFERRAL_RATE not in current_snapshot.columns:
            continue 
            
        old_rate = float(current_snapshot.loc[emp, EMP_DEFERRAL_RATE])
        new_rate = min(old_rate + cfg.increase_pct, cfg.cap)
        if new_rate <= old_rate:
            continue
        
        event_data = {
            "event_id": str(uuid.uuid4()),
            "event_time": as_of,
            EMP_ID: emp,
            "event_type": EVT_AUTO_INCREASE,
            "value_num": new_rate,
            "value_json": pd.io.json.dumps({"old_rate": old_rate, "new_rate": new_rate}),
            "meta": None,
        }
        out_events.append(event_data)

    if not out_events:
        return []
        
    df_out = pd.DataFrame(out_events, columns=EVENT_COLS)
    return [df_out]
