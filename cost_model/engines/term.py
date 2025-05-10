# cost_model/engines/term.py

import pandas as pd
import numpy as np
from typing import List
from cost_model.state.event_log import EVENT_COLS, EVT_TERM
from cost_model.utils.columns import EMP_ID, EMP_TERM_DATE, EMP_ROLE
import logging

logger = logging.getLogger(__name__)

def _random_dates_in_year(year: int, n: int, rng: np.random.Generator):
    start = pd.Timestamp(f"{year}-01-01")
    end   = pd.Timestamp(f"{year}-12-31")
    days  = (end - start).days + 1
    offsets = rng.integers(0, days, size=n)
    return [start + pd.Timedelta(days=int(o)) for o in offsets]

def run(
    snapshot: pd.DataFrame,
    hazard_slice: pd.DataFrame, 
    rng: np.random.Generator,
    deterministic: bool,
) -> List[pd.DataFrame]:
    """
    Simulate terminations for one simulation year.
    Assumes hazard_slice already filtered to simulation_year == year.
    """
    # Extract the sim year (must be present as 'simulation_year')
    year = int(hazard_slice['simulation_year'].iloc[0])

    # Determine “as_of” start of year
    as_of = pd.Timestamp(f"{year}-01-01")

    # Select only active employees as of Jan 1
    active = snapshot[
        snapshot[EMP_TERM_DATE].isna() | (snapshot[EMP_TERM_DATE] > as_of)
    ].copy()
    n = len(active)
    if n == 0:
        return [pd.DataFrame(columns=EVENT_COLS)]

    # Merge in just this year’s term_rate
    df = active.merge(
        hazard_slice[[EMP_ROLE, 'tenure_band', 'term_rate']],
        on=[EMP_ROLE, 'tenure_band'],
        how='left',
    )

    # Decide terminations
    if deterministic:
        rate = df['term_rate'].fillna(0).mean()
        k    = int(np.ceil(n * rate))
        if k == 0:
            return [pd.DataFrame(columns=EVENT_COLS)]
        losers = rng.choice(df[EMP_ID], size=k, replace=False)
    else:
        probs = df['term_rate'].fillna(0).values
        draw  = rng.random(n)
        losers = df.loc[draw < probs, EMP_ID].tolist()
        if not losers:
            return [pd.DataFrame(columns=EVENT_COLS)]

    # Assign random dates and build events
    dates = _random_dates_in_year(year, len(losers), rng)
    events = [{
        'event_id':   f"evt_term_{year}_{idx:04d}",
        EMP_ID:       emp,
        'event_type': EVT_TERM,
        'event_date': dt,
        'year':       year,
        'value_num':  np.nan,
        'value_json': None,
        'notes':      f"Termination for {emp} in {year}"
    } for idx, (emp, dt) in enumerate(zip(losers, dates))]

    return [pd.DataFrame(events, columns=EVENT_COLS)]

def run_new_hires(
    snapshot: pd.DataFrame,
    hazard_slice: pd.DataFrame,
    rng: np.random.Generator,
    year: int,
    deterministic: bool
) -> List[pd.DataFrame]:
    """
    Terminate only those employees whose hire date is in `year`,
    using the `new_hire_termination_rate` from hazard_slice.
    """
    as_of = pd.Timestamp(f"{year}-01-01")
    # Identify new hires in this year
    from cost_model.utils.columns import EMP_HIRE_DATE
    df_nh = snapshot[
        (snapshot[EMP_HIRE_DATE] >= as_of) &
        ((snapshot[EMP_TERM_DATE].isna()) | (snapshot[EMP_TERM_DATE] > as_of))
    ].copy()
    if df_nh.empty:
        return [pd.DataFrame(columns=EVENT_COLS)]
    # Use the pre-filtered hazard_slice for this year
    rate = hazard_slice['new_hire_termination_rate'].iloc[0] if 'new_hire_termination_rate' in hazard_slice.columns else 0.0
    df_nh['new_hire_termination_rate'] = rate
    # Now proceed with deterministic/probabilistic logic as before
    n = len(df_nh)
    if n == 0 or rate == 0.0:
        return [pd.DataFrame(columns=EVENT_COLS)]
    if deterministic:
        k = int(np.ceil(n * rate))
        if k == 0:
            return [pd.DataFrame(columns=EVENT_COLS)]
        losers = np.random.choice(df_nh[EMP_ID], size=k, replace=False)
        losers = rng.choice(df_rate[EMP_ID], size=k, replace=False) if k > 0 else []
    else:
        draw = rng.random(n)
        losers = df_rate.loc[draw < rates, EMP_ID].tolist()
    if not losers:
        return [pd.DataFrame(columns=EVENT_COLS)]
    dates = _random_dates_in_year(year, len(losers), rng)
    events = [{
        'event_id':   f"evt_nh_term_{year}_{i:04d}",
        EMP_ID:       emp,
        'event_type': EVT_TERM,
        'event_date': dt,
        'year':       year,
        'value_num':  np.nan,
        'value_json': None,
        'notes':      f"New-hire termination for {emp} in {year}"
    } for i, (emp, dt) in enumerate(zip(losers, dates))]
    return [pd.DataFrame(events, columns=EVENT_COLS)]