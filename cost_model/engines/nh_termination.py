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
from cost_model.state.schema import NEW_HIRE_TERM_RATE
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
    nh_term_rate = hazard_slice[NEW_HIRE_TERM_RATE].iloc[0] if NEW_HIRE_TERM_RATE in hazard_slice.columns else 0.0
    n = len(df_nh)
    k = min(int(round(n * nh_term_rate)), n)  # Ensure k is not larger than n
    
    # Create a mapping of employee IDs to their row indices for safer lookups
    if k <= 0:
        return [pd.DataFrame(columns=EVENT_COLS), pd.DataFrame(columns=EVENT_COLS)]
    
    # Select k employees to terminate
    if deterministic:
        # For deterministic, select the first k rows
        selected_indices = df_nh.index[:k]
    else:
        # For stochastic, randomly select k indices
        selected_indices = rng.choice(df_nh.index, size=k, replace=False)
    
    # Get the employee IDs and hire dates for the selected employees
    selected_employees = df_nh.loc[selected_indices]
    exit_ids = selected_employees[EMP_ID].values
    loser_hire_dates = selected_employees[EMP_HIRE_DATE].values
    dates = random_dates_between(loser_hire_dates, end_of_year, rng)
    term_events = []
    comp_events = []
    # Create a DataFrame for selected employees with all the data we need
    selected_df = selected_employees.copy()
    selected_df['term_date'] = dates
    
    for i, row in selected_df.iterrows():
        emp = row[EMP_ID]
        hire_date = row[EMP_HIRE_DATE]
        term_date = row['term_date']
        tenure_days = int((term_date - hire_date).days)
        comp = row[EMP_GROSS_COMP] if EMP_GROSS_COMP in selected_df.columns else None
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
