# cost_model/engines/term.py

from typing import List
import pandas as pd
import numpy as np
from cost_model.state.event_log import EVENT_COLS, EVT_TERM
import math
import json

def _random_dates_in_year(year, n, rng):
    # Return n random dates uniformly in [Jan 1, Dec 31] of the given year
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")
    days = (end - start).days + 1
    offsets = rng.integers(0, days, size=n)
    return [start + pd.Timedelta(days=int(o)) for o in offsets]

def run(
    snapshot: pd.DataFrame,
    hazard_slice: pd.DataFrame,
    rng: np.random.Generator,
    deterministic: bool
) -> List[pd.DataFrame]:
    """
    Simulate terminations across the active population in `snapshot`.
    """
    # Use the year from the snapshot or default to current year
    year = pd.Timestamp.now().year
    if 'event_time' in snapshot.columns and not snapshot['event_time'].isna().all():
        year = pd.Timestamp(snapshot['event_time'].min()).year
    as_of = pd.Timestamp(f"{year}-01-01")
    active = snapshot[(snapshot['term_date'].isna()) | (snapshot['term_date'] > as_of)].copy()
    if 'employee_id' not in active.columns:
        active = active.reset_index()
    n = len(active)
    if n == 0:
        return [pd.DataFrame(columns=EVENT_COLS)]
    # Merge in term_rate
    df = active.merge(
        hazard_slice[['role', 'tenure_band', 'term_rate']],
        on=['role', 'tenure_band'], how='left'
    )
    # Decide who terminates
    if deterministic:
        rate = df['term_rate'].mean() if not df['term_rate'].isna().all() else 0
        k = int(math.ceil(n * rate))
        if k == 0:
            return [pd.DataFrame(columns=EVENT_COLS)]
        losers = rng.choice(df['employee_id'], size=k, replace=False)
    else:
        probs = df['term_rate'].fillna(0).values
        draw = rng.random(n)
        losers = df.loc[draw < probs, 'employee_id'].tolist()
    # Assign random dates
    dates = _random_dates_in_year(year, len(losers), rng)
    # Build events
    events = [{
        'event_time': dt,
        'employee_id': emp,
        'event_type': EVT_TERM,
        'value': np.nan,
        'meta': ''
    } for emp, dt in zip(losers, dates)]
    return [pd.DataFrame(events, columns=EVENT_COLS)]
