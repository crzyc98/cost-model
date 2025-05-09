from datetime import date
import pandas as pd
from cost_model.config.plan_rules import AutoIncreaseConfig
from typing import List
import uuid

def run(
    snapshot: pd.DataFrame,
    events: pd.DataFrame,
    as_of: pd.Timestamp,
    cfg: AutoIncreaseConfig
) -> List[pd.DataFrame]:
    """
    For employees who’ve been enrolled (or are auto-enrolled) and haven’t opted out,
    generate a single EVT_AUTO_INCREASE event at `as_of` that bumps their deferral rate
    by cfg.increase_rate, capped at cfg.cap_rate.
    """
    EVT_ENROLL = 'EVT_ENROLL'
    EVT_AUTO_ENROLL = 'EVT_AUTO_ENROLL'
    EVT_OPT_OUT = 'EVT_OPT_OUT'
    EVT_AUTO_INCREASE = 'EVT_AUTO_INCREASE'

    # 1. Identify candidates: enrolled or auto-enrolled as of as_of
    if snapshot.index.name != 'employee_id':
        snapshot = snapshot.set_index('employee_id')

    enrolled_ids = set(
        events.loc[
            (events['event_type'].isin([EVT_ENROLL, EVT_AUTO_ENROLL])) & (events['event_time'] <= as_of),
            'employee_id'
        ].astype(str)
    )
    # Exclude anyone with OPT_OUT as of as_of
    opted_out_ids = set(
        events.loc[
            (events['event_type'] == EVT_OPT_OUT) & (events['event_time'] <= as_of),
            'employee_id'
        ].astype(str)
    )
    # Exclude anyone with an increase event at as_of
    already_increased_ids = set(
        events.loc[
            (events['event_type'] == EVT_AUTO_INCREASE) & (events['event_time'] == as_of),
            'employee_id'
        ].astype(str)
    )
    candidates = enrolled_ids - opted_out_ids - already_increased_ids

    out = []
    for emp in candidates:
        if emp not in snapshot.index:
            continue
        old_rate = float(snapshot.loc[emp, 'employee_deferral_rate'])
        new_rate = min(old_rate + cfg.increase_pct, cfg.cap)
        if new_rate <= old_rate:
            continue
        rows = []
        rows.append({
            "event_id": str(uuid.uuid4()),
            "event_time": as_of,
            "employee_id": emp,
            "event_type": EVT_AUTO_INCREASE,
            'value_json': [{'old_rate': old_rate, 'new_rate': new_rate}],
            'meta': [None]
        })
        out.append(pd.DataFrame(rows))
    return out
