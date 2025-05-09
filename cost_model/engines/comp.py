# cost_model/engines/comp.py

from typing import List
import pandas as pd
from cost_model.state.event_log import EVENT_COLS, EVT_COMP
import json

def bump(
    snapshot: pd.DataFrame,
    hazard_slice: pd.DataFrame,
    as_of: pd.Timestamp
) -> List[pd.DataFrame]:
    """
    For each *active* employee in `snapshot`, applies the compensation‐raise percentage
    from `hazard_slice` based on their role and tenure band, and returns a list containing
    one DataFrame of 'comp' events with columns EVENT_COLS.
    """
    # 1. Filter to active employees only
    active = snapshot[(snapshot['term_date'].isna()) | (snapshot['term_date'] > as_of)].copy()
    if 'employee_id' not in active.columns:
        active = active.reset_index()  # ensure employee_id is a column
    # 2. Merge in comp_raise_pct
    df = active.merge(
        hazard_slice[['role', 'tenure_band', 'comp_raise_pct']],
        on=['role', 'tenure_band'], how='left'
    )
    # 3. For everyone with comp_raise_pct > 0, build one event row
    events = []
    for _, row in df.iterrows():
        pct = row['comp_raise_pct']
        if pct is not None and pct > 0:
            old_comp = row.get('current_compensation', 0)
            new_comp = old_comp * (1 + pct)
            meta = {
                'old_comp': old_comp,
                'new_comp': new_comp,
                'pct': pct
            }
            events.append({
                'event_time': as_of,
                'employee_id': row['employee_id'],
                'event_type': EVT_COMP,
                'value_num': pct,
                'meta': json.dumps(meta)
            })
    return [pd.DataFrame(events, columns=EVENT_COLS)]
