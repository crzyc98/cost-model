# cost_model/engines/comp.py

from typing import List
import pandas as pd
from cost_model.state.event_log import EVENT_COLS, EVT_COMP
from cost_model.utils.columns import EMP_ID, EMP_TERM_DATE, EMP_ROLE, EMP_GROSS_COMP
import json


def bump(
    snapshot: pd.DataFrame, hazard_slice: pd.DataFrame, as_of: pd.Timestamp
) -> List[pd.DataFrame]:
    """
    For each *active* employee in `snapshot`, applies the compensation‐raise percentage
    from `hazard_slice` based on their role and tenure band, and returns a list containing
    one DataFrame of 'comp' events with columns EVENT_COLS.
    """
    # 1. Filter to active employees only
    active = snapshot[
        (snapshot[EMP_TERM_DATE].isna()) | (snapshot[EMP_TERM_DATE] > as_of)
    ].copy()
    if EMP_ID not in active.columns:
        # If EMP_ID (index name) is not a column, reset index. 
        # This happens if snapshot was set_index(EMP_ID) without drop=False and then not reset.
        # Or if EMP_ID is indeed the index name as expected by snapshot schema.
        if active.index.name == EMP_ID:
            active = active.reset_index() # Makes EMP_ID a column
        else:
            # This case would be unexpected if snapshot conforms to schema where EMP_ID is index or column
            raise ValueError(f"{EMP_ID} not found as index or column in active snapshot slice.")

    # 2. Merge in comp_raise_pct
    # Assuming hazard_slice uses EMP_ROLE for 'role'. 'tenure_band' is specific.
    df = active.merge(
        hazard_slice[[EMP_ROLE, "tenure_band", "comp_raise_pct"]],
        on=[EMP_ROLE, "tenure_band"],
        how="left",
    )
    # 3. For everyone with comp_raise_pct > 0, build one event row
    events = []
    for _, row in df.iterrows():
        pct = row["comp_raise_pct"]
        if pct is not None and pct > 0:
            old_comp = row.get(EMP_GROSS_COMP, 0)
            new_comp = old_comp * (1 + pct)
            meta = {"old_comp": old_comp, "new_comp": new_comp, "pct": pct}
            events.append(
                {
                    "event_time": as_of,
                    EMP_ID: row[EMP_ID],
                    "event_type": EVT_COMP,
                    "value_num": pct,
                    "meta": json.dumps(meta),
                }
            )
    return [pd.DataFrame(events, columns=EVENT_COLS)]
