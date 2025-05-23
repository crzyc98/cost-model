"""
Deterministic New-Hire Termination Logic

This module provides deterministic and stochastic termination logic for new hires, migrated from term.py as part of the hiring/termination flow migration.
"""
import pandas as pd
import numpy as np
import json
from typing import List
from cost_model.state.event_log import EVENT_COLS, EVT_TERM, EVT_COMP, create_event
from cost_model.utils.columns import EMP_ID, EMP_GROSS_COMP, EMP_HIRE_DATE, EMP_TERM_DATE
import logging

logger = logging.getLogger(__name__)

def random_dates_between(start_dates, end_date, rng: np.random.Generator):
    result = []
    for start in start_dates:
        start = pd.Timestamp(start)
        days = max(0, (end_date - start).days)
        if days == 0:
            result.append(start)
        else:
            offset = rng.integers(0, days + 1)
            result.append(start + pd.Timedelta(days=int(offset)))
    return result

def run_new_hires(
    snapshot: pd.DataFrame,
    hazard_slice: pd.DataFrame,
    rng: np.random.Generator,
    year: int,
    deterministic: bool
) -> List[pd.DataFrame]:
    """
    Terminate only those employees whose hire date is in `year`, using the `new_hire_termination_rate` from hazard_slice.
    """
    as_of = pd.Timestamp(f"{year}-01-01")
    end_of_year = pd.Timestamp(f"{year}-12-31")
    df_nh = snapshot[
        (snapshot[EMP_HIRE_DATE] >= as_of) &
        ((snapshot[EMP_TERM_DATE].isna()) | (snapshot[EMP_TERM_DATE] > as_of))
    ].copy()
    if df_nh.empty:
        return [pd.DataFrame(columns=EVENT_COLS), pd.DataFrame(columns=EVENT_COLS)]
    # Get termination rate
    nh_term_rate = hazard_slice['new_hire_termination_rate'].iloc[0] if 'new_hire_termination_rate' in hazard_slice else 0.0
    n = len(df_nh)
    k = int(round(n * nh_term_rate))
    if deterministic:
        exit_ids = df_nh[EMP_ID].iloc[:k].values
    else:
        exit_ids = rng.choice(df_nh[EMP_ID], size=k, replace=False)
    loser_hire_dates = [df_nh.loc[df_nh[EMP_ID] == emp, EMP_HIRE_DATE].iloc[0] for emp in exit_ids]
    dates = random_dates_between(loser_hire_dates, end_of_year, rng)
    term_events = []
    comp_events = []
    for emp, hire_date, term_date in zip(exit_ids, loser_hire_dates, dates):
        tenure_days = int((term_date - hire_date).days)
        comp = df_nh.loc[df_nh[EMP_ID] == emp, EMP_GROSS_COMP].iloc[0] if EMP_GROSS_COMP in df_nh.columns else None
        days_worked = (term_date - hire_date).days + 1
        prorated = comp * (days_worked / 365.25) if comp is not None else None
        term_events.append(create_event(
            event_time=term_date,
            employee_id=emp,
            event_type=EVT_TERM,
            value_num=None,
            value_json=json.dumps({
                "reason": "new_hire_termination",
                "tenure_days": tenure_days
            }),
            meta=f"New-hire termination for {emp} in {year}"
        ))
        if prorated is not None:
            comp_events.append(create_event(
                event_time=term_date,
                employee_id=emp,
                event_type=EVT_COMP,
                value_num=prorated,
                value_json=None,
                meta=f"Prorated comp for {emp} ({days_worked} days from hire to term)"
            ))
    df_term = pd.DataFrame(term_events, columns=EVENT_COLS).sort_values("event_time", ignore_index=True)
    df_comp = pd.DataFrame(comp_events, columns=EVENT_COLS).sort_values("event_time", ignore_index=True)
    return [df_term, df_comp]
