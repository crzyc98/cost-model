import uuid
import json
from datetime import timedelta
from typing import List
import pandas as pd
from cost_model.config.plan_rules import EligibilityEventsConfig
from cost_model.utils.columns import EMP_ID, EMP_HIRE_DATE

# canonical event schema
EVENT_COLS = ["event_id","event_time",EMP_ID,"event_type","value_num","value_json","meta"]
EVENT_DTYPES = {
    "event_id":"string","event_time":"datetime64[ns]",EMP_ID:"string",
    "event_type":"string","value_num":"float64","value_json":"string","meta":"string"
}

from typing import Optional

def run(
    snapshot: pd.DataFrame,
    events: pd.DataFrame,
    as_of: pd.Timestamp,
    prev_as_of: Optional[pd.Timestamp],
    cfg: EligibilityEventsConfig
) -> list:
    """
    Emit milestone eligibility events for employees who cross a configured milestone between prev_as_of and as_of.
    """
    # 1. Build milestones in months
    mm = set(cfg.milestone_months or []) | {y * 12 for y in (cfg.milestone_years or [])}
    if not mm:
        return []
    # 2. Default prev_as_of to before everyone if None
    if prev_as_of is None:
        prev = snapshot[EMP_HIRE_DATE].min() if EMP_HIRE_DATE in snapshot.columns else pd.Timestamp("1900-01-01")
    else:
        prev = prev_as_of
    # 3. Ensure index
    if snapshot.index.name != EMP_ID and EMP_ID in snapshot.columns:
        snapshot = snapshot.set_index(EMP_ID, drop=False)
    elif snapshot.index.name != EMP_ID:
        # This case implies EMP_ID is not a column and not the index name, which is an issue.
        # Or EMP_ID is the index, but snapshot.index.name is None (e.g. RangeIndex if no name set)
        # For safety, let's assume if EMP_ID is not a column, it must be the index.
        # If it's truly missing, subsequent operations will fail, which is desired behavior.
        if EMP_ID not in snapshot.columns and EMP_ID != snapshot.index.name:
             raise ValueError(f"{EMP_ID} column not found in snapshot and not set as index.")

    # Check if EMP_HIRE_DATE exists
    if EMP_HIRE_DATE not in snapshot.columns:
        # Potentially return empty or log a warning, as milestones can't be calculated.
        # For now, let's assume it must exist or it's an error in the input snapshot.
        raise ValueError(f"{EMP_HIRE_DATE} column not found in snapshot.")

    # 4. Compute service months at both dates
    def svc_months(dt):
        return (
            (dt.year - snapshot[EMP_HIRE_DATE].dt.year) * 12 +
            (dt.month - snapshot[EMP_HIRE_DATE].dt.month)
        )
    svc_prev = svc_months(prev)
    svc_now = svc_months(as_of)
    rows = []
    for emp in snapshot.index: # Assumes index is EMP_ID
        pv = svc_prev.loc[emp]
        nv = svc_now.loc[emp]
        # print(f"[MILESTONE DEBUG] emp={emp}, svc_prev={pv}, svc_now={nv}")
        for m in sorted(mm):
            # print(f"[MILESTONE DEBUG]   checking milestone {m}...", end=" ")
            if pv < m <= nv:
                # print("FIRED")
                et = cfg.event_type_map.get(m)
                if not et:
                    continue
                rows.append({
                    "event_id": str(uuid.uuid4()),
                    "event_time": as_of,
                    EMP_ID: emp,
                    "event_type": et,
                    "value_num": None,
                    "value_json": json.dumps({"milestone_months": m}),
                    "meta": None,
                })
            # else:
                # print("not fired")
    if not rows:
        return []
    df = pd.DataFrame(rows, columns=EVENT_COLS).astype(EVENT_DTYPES)
    return [df]