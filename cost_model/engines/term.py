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
from cost_model.state.schema import (
    EMP_ID,
    EMP_LEVEL,
    EMP_GROSS_COMP,
    EMP_HIRE_DATE,
    EMP_TERM_DATE,
    EMP_TENURE_BAND,
    TERM_RATE,
    SIMULATION_YEAR,
    NEW_HIRE_TERM_RATE
)
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
    # Extract the sim year using standardized name
    year = int(hazard_slice[SIMULATION_YEAR].iloc[0])
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

    # 1. Pull out only the columns you need and dedupe the hazard table
    hz = (
        hazard_slice[[EMP_LEVEL, EMP_TENURE_BAND, TERM_RATE]]
        .drop_duplicates(subset=[EMP_LEVEL, EMP_TENURE_BAND])
    )

    # 2. Merge with many-to-one validation (one hazard row to many employees)
    merged_df = active.merge(
        hz,
        on=[EMP_LEVEL, EMP_TENURE_BAND],
        how="left",
        validate="many_to_one"
    )
    
    # Validate merge results
    n = len(merged_df)
    if n == 0:
        logger.warning(
            f"[TERM] Year {year}: No employees matched the hazard table. "
            f"Check if {EMP_LEVEL} and {EMP_TENURE_BAND} values in snapshot match hazard table."
        )
        return [pd.DataFrame(columns=EVENT_COLS)]
    
    # Check for missing or zero rates
    missing_rates = merged_df[TERM_RATE].isna()
    zero_rates = (merged_df[TERM_RATE] == 0)
    
    if missing_rates.any():
        logger.warning(
            f"[TERM] Year {year}: {missing_rates.sum()} employees missing {TERM_RATE} after merge. "
            f"Filling with 0. Check hazard table coverage for all {EMP_LEVEL} and {EMP_TENURE_BAND} combinations."
        )
        merged_df[TERM_RATE] = merged_df[TERM_RATE].fillna(0)
    
    logger.debug(f"[TERM] Year {year}: {n} employees processed, {zero_rates.sum()} with zero {TERM_RATE} after merge.")
    logger.info(
        f"[TERM] Year {year}: {TERM_RATE} stats - "
        f"min={merged_df[TERM_RATE].min():.4f}, "
        f"max={merged_df[TERM_RATE].max():.4f}, "
        f"mean={merged_df[TERM_RATE].mean():.4f}, "
        f"median={merged_df[TERM_RATE].median():.4f}."
    )
    
    # Decide terminations
    if deterministic:
        rate = merged_df[TERM_RATE].mean()  # Already filled NAs with 0
        k = int(math.ceil(n * rate))
        logger.info(f"[TERM] Year {year}: Deterministic mode, using mean rate {rate:.4f}, selecting {k} for termination.")
        if k == 0:
            logger.info(f"[TERM] Year {year}: No employees selected for termination (k=0).")
            return [pd.DataFrame(columns=EVENT_COLS)]
        losers = rng.choice(merged_df[EMP_ID], size=k, replace=False)
    else:
        probs = merged_df[TERM_RATE].values  # Already filled NAs with 0
        logger.info(
            f"[TERM] Year {year}: Stochastic mode - "
            f"min prob={probs.min():.4f}, max prob={probs.max():.4f}, "
            f"employees with non-zero prob={(probs > 0).sum()}/{n} ({(probs > 0).mean()*100:.1f}%)"
        )
        draw = rng.random(n)  # Use n which is len(merged_df)
        losers = merged_df.loc[draw < probs, EMP_ID].tolist()
        logger.info(f"[TERM] Year {year}: {len(losers)} employees selected for termination.")
        
    if len(losers) == 0:
        logger.info(f"[TERM] Year {year}: No employees selected for termination (stochastic draw).")
        return [pd.DataFrame(columns=EVENT_COLS)]
        
    # Ensure we don't have duplicate employee IDs in the termination list
    if len(losers) > 0:
        # Convert to set to remove duplicates, then back to list
        unique_losers = list(set(losers))
        if len(unique_losers) < len(losers):
            logger.warning(f"Removed {len(losers) - len(unique_losers)} duplicate employee IDs from termination list")
        losers = unique_losers
        
    # Assign random dates and build events
    dates = random_dates_in_year(year, len(losers), rng)
    term_events = []
    comp_events = []
    
    for emp, dt in zip(losers, dates):
        # Skip if we've already processed this employee
        if any(e['employee_id'] == emp for e in term_events):
            logger.warning(f"Skipping duplicate termination for employee {emp}")
            continue
            
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
    rate = hazard_slice[NEW_HIRE_TERM_RATE].iloc[0] if NEW_HIRE_TERM_RATE in hazard_slice.columns else 0.0
    df_nh[NEW_HIRE_TERM_RATE] = rate
    
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
        rates = df_nh[NEW_HIRE_TERM_RATE]
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
