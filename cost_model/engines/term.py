# cost_model/engines/term.py

import pandas as pd
import numpy as np
import math
import json
from typing import List
from cost_model.state.event_log import EVENT_COLS, EVT_TERM, EVT_COMP, create_event
from cost_model.utils.columns import EMP_ID, EMP_ROLE, EMP_GROSS_COMP, EMP_HIRE_DATE, EMP_TERM_DATE
import logging

logger = logging.getLogger(__name__)

def _random_dates_in_year(year: int, n: int, rng: np.random.Generator):
    start = pd.Timestamp(f"{year}-01-01")
    end   = pd.Timestamp(f"{year}-12-31")
    days  = (end - start).days + 1
    offsets = rng.integers(0, days, size=n)
    return [start + pd.Timedelta(days=int(o)) for o in offsets]

from cost_model.utils.columns import EMP_TENURE

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

    # ensure 'role' from hazard_slice is mapped to EMP_ROLE for merge
    hz = hazard_slice[['role', 'tenure_band', 'term_rate']].rename(columns={'role': EMP_ROLE})
    df = active.merge(
        hz,
        on=[EMP_ROLE, 'tenure_band'],
        how='left',
    )

    # Decide terminations
    if deterministic:
        rate = df['term_rate'].fillna(0).mean()
        k    = int(math.ceil(n * rate))
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
    term_events = []
    comp_events = []
    for emp, dt in zip(losers, dates):
        # Termination event
        hire_date = snapshot.loc[snapshot[EMP_ID] == emp, EMP_HIRE_DATE].iloc[0] if EMP_HIRE_DATE in snapshot.columns and not snapshot.loc[snapshot[EMP_ID] == emp, EMP_HIRE_DATE].isna().iloc[0] else None
        tenure_days = int((dt - hire_date).days) if hire_date is not None else None
        # Prorate comp for days worked in year
        if EMP_GROSS_COMP in snapshot.columns:
            comp = snapshot.loc[snapshot[EMP_ID] == emp, EMP_GROSS_COMP].iloc[0]
        else:
            comp = None
        start_of_year = pd.Timestamp(f"{year}-01-01")
        days_worked = (dt - start_of_year).days + 1 if dt >= start_of_year else None
        prorated = comp * (days_worked / 365.25) if comp is not None and days_worked is not None else None
        # Termination event
        term_events.append(create_event(
            event_time=dt,
            employee_id=emp,
            event_type=EVT_TERM,
            value_num=None,
            value_json=json.dumps({
                "reason": "termination",
                "tenure_days": tenure_days
            }),
            meta=f"Termination for {emp} in {year}"
        ))
        # Prorated comp event
        if prorated is not None:
            comp_events.append(create_event(
                event_time=dt,
                employee_id=emp,
                event_type=EVT_COMP,
                value_num=prorated,
                value_json=None,
                meta=f"Prorated final-year comp for {emp} ({days_worked} days)"
            ))
    df_term = pd.DataFrame(term_events, columns=EVENT_COLS).sort_values("event_time", ignore_index=True)
    df_comp = pd.DataFrame(comp_events, columns=EVENT_COLS).sort_values("event_time", ignore_index=True)
    return [df_term, df_comp]

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
        # Use df_nh consistently instead of df_rate
        losers = rng.choice(df_nh[EMP_ID], size=k, replace=False) if k > 0 else []
    else:
        draw = rng.random(n)
        rates = df_nh['new_hire_termination_rate']
        losers = df_nh.loc[draw < rates, EMP_ID].tolist()
    if not losers:
        return [pd.DataFrame(columns=EVENT_COLS)]
    dates = _random_dates_in_year(year, len(losers), rng)
    term_events = []
    comp_events = []
    for emp, dt in zip(losers, dates):
        # Termination event
        hire_date = snapshot.loc[snapshot[EMP_ID] == emp, EMP_HIRE_DATE].iloc[0] if EMP_HIRE_DATE in snapshot.columns and not snapshot.loc[snapshot[EMP_ID] == emp, EMP_HIRE_DATE].isna().iloc[0] else None
        tenure_days = int((dt - hire_date).days) if hire_date is not None else None
        # Prorate comp for days worked in year
        if EMP_GROSS_COMP in snapshot.columns:
            comp = snapshot.loc[snapshot[EMP_ID] == emp, EMP_GROSS_COMP].iloc[0]
        else:
            comp = None
        start_of_year = pd.Timestamp(f"{year}-01-01")
        days_worked = (dt - start_of_year).days + 1 if dt >= start_of_year else None
        prorated = comp * (days_worked / 365.25) if comp is not None and days_worked is not None else None
        # Termination event
        term_events.append(create_event(
            event_time=dt,
            employee_id=emp,
            event_type=EVT_TERM,
            value_num=None,
            value_json=json.dumps({
                "reason": "new_hire_termination",
                "tenure_days": tenure_days
            }),
            meta=f"New-hire termination for {emp} in {year}"
        ))
        # Prorated comp event
        if prorated is not None:
            comp_events.append(create_event(
                event_time=dt,
                employee_id=emp,
                event_type=EVT_COMP,
                value_num=prorated,
                value_json=None,
                meta=f"Prorated final-year comp for {emp} ({days_worked} days)"
            ))
    df_term = pd.DataFrame(term_events, columns=EVENT_COLS).sort_values("event_time", ignore_index=True)
    df_comp = pd.DataFrame(comp_events, columns=EVENT_COLS).sort_values("event_time", ignore_index=True)
    return [df_term, df_comp]