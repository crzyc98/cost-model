# cost_model/engines/term.py
"""
Engine for simulating employee terminations during workforce simulations.
QuickStart: see docs/cost_model/engines/term.md
"""

import pandas as pd
import numpy as np
import math
import json
from typing import List
from cost_model.state.event_log import EVENT_COLS, EVT_TERM, EVT_COMP, create_event
from cost_model.utils.columns import EMP_ID, EMP_LEVEL, EMP_GROSS_COMP, EMP_HIRE_DATE, EMP_TERM_DATE
import logging

logger = logging.getLogger(__name__)

def random_dates_in_year(year: int, n: int, rng: np.random.Generator):
    """Generate random dates within a given year."""
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")
    days = (end - start).days + 1
    offsets = rng.integers(0, days, size=n)
    return [start + pd.Timedelta(days=int(o)) for o in offsets]

def random_dates_between(start_dates, end_date, rng: np.random.Generator):
    """Generate random dates between each start date and end date."""
    result = []
    for start in start_dates:
        # Ensure start is a pandas Timestamp
        start = pd.Timestamp(start)
        # Calculate max days between start and end
        days = max(0, (end_date - start).days)
        if days == 0:
            # If hire date is the same as end of year, use that date
            result.append(start)
        else:
            # Generate random offset (0 to days inclusive)
            offset = rng.integers(0, days + 1)
            result.append(start + pd.Timedelta(days=int(offset)))
    return result

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
    # Determine "as_of" start of year
    as_of = pd.Timestamp(f"{year}-01-01")
    logger = logging.getLogger(__name__)
    
    # Ensure EMP_HIRE_DATE is datetime
    snapshot[EMP_HIRE_DATE] = pd.to_datetime(snapshot[EMP_HIRE_DATE], errors='coerce')
    
    # Select only experienced actives as of Jan 1 (hire-date BEFORE Jan 1)
    active = snapshot[
        ((snapshot[EMP_TERM_DATE].isna()) | (snapshot[EMP_TERM_DATE] > as_of))
        & (snapshot[EMP_HIRE_DATE] < as_of)
    ].copy()
    
    # Ensure employee_level is Int64 to match the schema
    active[EMP_LEVEL] = pd.to_numeric(active[EMP_LEVEL], errors='coerce').astype('Int64')
    n = len(active)
    logger.info(f"[TERM] Year {year}: {n} active employees eligible for termination.")
    if n == 0:
        logger.info(f"[TERM] Year {year}: No active employees for termination.")
        return [pd.DataFrame(columns=EVENT_COLS)]
        
    # Debug: Log the columns in hazard_slice
    logger.debug(f"[TERM] Year {year}: hazard_slice columns: {hazard_slice.columns.tolist()}")
    logger.debug(f"[TERM] Year {year}: hazard_slice head: {hazard_slice.head().to_dict()}")
    
    # The hazard table uses 'level' instead of 'employee_level'
    # Ensure consistent data types before merge
    hz = hazard_slice[['level', 'tenure_band', 'term_rate']].copy()
    hz = hz.rename(columns={'level': EMP_LEVEL})
    # Convert to Int64 to match the schema's expected type
    hz[EMP_LEVEL] = pd.to_numeric(hz[EMP_LEVEL], errors='coerce').astype('Int64')
    # Also ensure tenure_band is string type as expected
    hz['tenure_band'] = hz['tenure_band'].astype(str)
    
    # Log hazard table and active employees before merge for debugging
    logger.debug(f"[TERM] Year {year}: Active employee levels: {active[EMP_LEVEL].unique().tolist()}")
    logger.debug(f"[TERM] Year {year}: Active employee tenure bands: {active['tenure_band'].unique().tolist()}")
    logger.debug(f"[TERM] Year {year}: Hazard table levels: {hz[EMP_LEVEL].unique().tolist()}")
    logger.debug(f"[TERM] Year {year}: Hazard table tenure bands: {hz['tenure_band'].unique().tolist()}")
    
    # Merge hazard rates into active employees
    df = active.merge(
        hz,
        on=[EMP_LEVEL, 'tenure_band'],
        how='left',
    )
    
    # CRITICAL: Verify the merge didn't inflate the employee count
    if len(df) != n:
        logger.error(f"[TERM] Year {year}: CRITICAL ERROR - Merge inflated employee count from {n} to {len(df)}!")
        logger.error(f"[TERM] Year {year}: This will cause incorrect termination counts.")
        # Force df back to the original active employees to prevent inflation
        # Keep only the first occurrence of each employee to avoid duplicates
        if EMP_ID in df.columns:
            df = df.drop_duplicates(subset=[EMP_ID])
        else:
            df = df.iloc[:n] # Fallback if EMP_ID not available
        logger.info(f"[TERM] Year {year}: Corrected employee count back to {len(df)}.")
    
    # Verify we still have the same employees
    assert len(df) == n, f"Employee count mismatch after hazard table merge: {len(df)} != {n}"
    
    missing_hazard = df['term_rate'].isna().sum()
    logger.info(f"[TERM] Year {year}: {missing_hazard} employees missing hazard/term_rate after merge.")
    logger.info(f"[TERM] Year {year}: term_rate stats: min={df['term_rate'].min()}, max={df['term_rate'].max()}, mean={df['term_rate'].mean()}, median={df['term_rate'].median()}.")
    
    # Decide terminations
    if deterministic:
        rate = df['term_rate'].fillna(0).mean()
        k = int(math.ceil(n * rate))
        logger.info(f"[TERM] Year {year}: Deterministic mode, using mean rate {rate:.4f}, selecting {k} for termination.")
        if k == 0:
            logger.info(f"[TERM] Year {year}: No employees selected for termination (k=0).")
            return [pd.DataFrame(columns=EVENT_COLS)]
        losers = rng.choice(df[EMP_ID], size=k, replace=False)
    else:
        probs = df['term_rate'].fillna(0).values
        logger.info(f"[TERM] Year {year}: Stochastic mode, min prob={probs.min()}, max prob={probs.max()}.")
        draw = rng.random(n)
        losers = df.loc[draw < probs, EMP_ID].tolist()
        logger.info(f"[TERM] Year {year}: {len(losers)} employees selected for termination.")
        
    if len(losers) == 0:
        logger.info(f"[TERM] Year {year}: No employees selected for termination (stochastic draw).")
        return [pd.DataFrame(columns=EVENT_COLS)]
        
    # Assign random dates and build events
    dates = random_dates_in_year(year, len(losers), rng)
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
    end_of_year = pd.Timestamp(f"{year}-12-31")
    
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
        
    if len(losers) == 0:
        return [pd.DataFrame(columns=EVENT_COLS)]
    
    # Fix: Get hire dates for all losers
    loser_hire_dates = []
    for emp in losers:
        hire_date = snapshot.loc[snapshot[EMP_ID] == emp, EMP_HIRE_DATE].iloc[0]
        loser_hire_dates.append(hire_date)
    
    # Fix: Generate termination dates that are AFTER hire dates
    dates = random_dates_between(loser_hire_dates, end_of_year, rng)
    
    term_events = []
    comp_events = []
    
    for emp, hire_date, term_date in zip(losers, loser_hire_dates, dates):
        # Calculate tenure days - now we're sure term_date is after hire_date
        tenure_days = int((term_date - hire_date).days)
        
        # Prorate comp for days worked from hire date to term date
        if EMP_GROSS_COMP in snapshot.columns:
            comp = snapshot.loc[snapshot[EMP_ID] == emp, EMP_GROSS_COMP].iloc[0]
        else:
            comp = None
            
        days_worked = (term_date - hire_date).days + 1
        prorated = comp * (days_worked / 365.25) if comp is not None else None
        
        # Termination event
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
        
        # Prorated comp event
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
