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
from cost_model.utils.tenure_utils import standardize_tenure_band

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

    # Log employee_level and employee_tenure_band distributions for diagnostics
    logger.warning(f"[TERM DIAGNOSTIC YR={year}] Before hazard merge: {len(active)} employees")
    
    # Log distribution of employee levels
    level_counts = active[EMP_LEVEL].value_counts().to_dict()
    logger.warning(f"[TERM DIAGNOSTIC YR={year}] Employee level distribution: {level_counts}")
    
    # Log distribution of tenure bands before standardization
    tenure_counts = active[EMP_TENURE_BAND].value_counts().to_dict()
    logger.warning(f"[TERM DIAGNOSTIC YR={year}] Tenure band distribution before standardization: {tenure_counts}")
    
    # Log sample of employee data for debugging
    sample_employees = active[[EMP_ID, EMP_LEVEL, EMP_TENURE_BAND]].sample(min(5, len(active)), random_state=42)
    logger.warning(f"[TERM DIAGNOSTIC YR={year}] Sample of employee data before standardization:\n{sample_employees.to_string()}")
    
    # Standardize tenure bands to ensure consistent format for the merge
    active[EMP_TENURE_BAND] = active[EMP_TENURE_BAND].map(standardize_tenure_band)
    
    # CRITICAL: Also standardize the hazard slice to ensure matching
    hazard_slice[EMP_TENURE_BAND] = hazard_slice[EMP_TENURE_BAND].map(standardize_tenure_band)
    
    # Log distribution after standardization
    tenure_counts_std = active[EMP_TENURE_BAND].value_counts().to_dict()
    logger.warning(f"[TERM DIAGNOSTIC YR={year}] Tenure band distribution after standardization: {tenure_counts_std}")
    
    # Log sample of hazard table data for debugging
    if len(hazard_slice) > 0:
        hazard_sample = hazard_slice[[EMP_LEVEL, EMP_TENURE_BAND, TERM_RATE]].drop_duplicates()
        logger.warning(f"[TERM DIAGNOSTIC YR={year}] Hazard table entries (sample):\n{hazard_sample.head(10).to_string()}")
        
    # STEP 2: Get a list of all unique employee_level and tenure_band combinations in the active population
    # This will help us identify which combinations we need to check against the hazard table
    active_combos = set(zip(active[EMP_LEVEL], active[EMP_TENURE_BAND]))
    logger.debug(f"[TERM DIAGNOSTIC YR={year}] Unique level-tenure combinations in active employees: {active_combos}")
    
    # 1. Pull out only the columns you need, dedupe the hazard table, and ensure consistent tenure bands
    hz = (
        hazard_slice[[EMP_LEVEL, EMP_TENURE_BAND, TERM_RATE]]
        .drop_duplicates(subset=[EMP_LEVEL, EMP_TENURE_BAND])
    )
    
    # Ensure hazard table tenure bands are standardized right before the merge
    hz[EMP_TENURE_BAND] = hz[EMP_TENURE_BAND].map(standardize_tenure_band)
    
    # DIAGNOSTIC: Log hazard table coverage
    hz_combos = set(zip(hz[EMP_LEVEL], hz[EMP_TENURE_BAND]))
    logger.warning(f"[TERM DIAGNOSTIC YR={year}] Hazard table contains {len(hz_combos)} level-tenure combinations")
    logger.warning(f"[TERM DIAGNOSTIC YR={year}] Sample of hazard table combinations: {list(hz_combos)[:5]}")
    
    # Log the unique values in each to verify they're matching format
    logger.warning(f"[TERM DIAGNOSTIC YR={year}] Unique employee tenure bands before merge: {active[EMP_TENURE_BAND].unique().tolist()}")
    logger.warning(f"[TERM DIAGNOSTIC YR={year}] Unique hazard tenure bands before merge: {hz[EMP_TENURE_BAND].unique().tolist()}")
    
    # 2. Merge with many-to-one validation (one hazard row to many employees)
    merged_df = active.merge(
        hz,
        on=[EMP_LEVEL, EMP_TENURE_BAND],
        how="left",
        validate="many_to_one"
    )
    
    # Validate merge results
    # Preventative fix: Drop rows with NA EMP_ID from merged_df before termination selection
    if EMP_ID in merged_df.columns and merged_df[EMP_ID].isna().any():
        na_ids_count = merged_df[EMP_ID].isna().sum()
        logger.warning(
            f"[TERM] Year {year}: Found {na_ids_count} employees with NA {EMP_ID} in merged_df "
            f"before termination selection. These will be excluded."
        )
        merged_df = merged_df.dropna(subset=[EMP_ID])
        if merged_df.empty:
            logger.warning(
                f"[TERM] Year {year}: No employees remaining in merged_df after removing NA {EMP_ID}s."
            )
            return [pd.DataFrame(columns=EVENT_COLS), pd.DataFrame(columns=EVENT_COLS)]

    n = len(merged_df) # Number of candidates
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
        # DIAGNOSTIC: Examine which employee level and tenure band combinations are missing
        missing_df = merged_df[missing_rates][[EMP_ID, EMP_LEVEL, EMP_TENURE_BAND]]
        missing_combos = missing_df.groupby([EMP_LEVEL, EMP_TENURE_BAND]).size().reset_index()
        missing_combos.columns = [EMP_LEVEL, EMP_TENURE_BAND, 'count']
        
        logger.warning(
            f"[TERM DIAGNOSTIC YR={year}] Missing level-tenure combinations:\n"
            f"{missing_combos.to_string()}"
        )
        
        # Show a sample of employee IDs with missing rates
        sample_size = min(5, missing_df.shape[0])
        if sample_size > 0:
            sample_missing = missing_df.sample(sample_size, random_state=42) if sample_size > 1 else missing_df.iloc[:1]
            logger.warning(
                f"[TERM DIAGNOSTIC YR={year}] Sample of {sample_size} employees missing term rates:\n"
                f"{sample_missing.to_string()}"
            )
        
        # FALLBACK STRATEGY: For level 2 with '0-1' tenure (specifically identified in logs)
        # Use a reasonable fallback rate based on other rates in the hazard table
        level2_01yr_mask = (merged_df[EMP_LEVEL] == 2) & (merged_df[EMP_TENURE_BAND] == '0-1') & missing_rates
        level2_01yr_count = level2_01yr_mask.sum()
        
        if level2_01yr_count > 0:
            # Find the average termination rate for level 2 employees of all tenure bands
            level2_rates = hz[hz[EMP_LEVEL] == 2][TERM_RATE]
            
            # If we have level 2 rates, use their mean; otherwise, use level 1 with 0-1 tenure as a proxy
            if not level2_rates.empty:
                fallback_rate = level2_rates.mean()
                logger.info(f"[TERM] Year {year}: Using average rate {fallback_rate:.4f} from level 2 employees as fallback")
            else:
                # Try to get the rate for level 1, 0-1 as a proxy
                level1_01yr = hz[(hz[EMP_LEVEL] == 1) & (hz[EMP_TENURE_BAND] == '0-1')]
                if not level1_01yr.empty:
                    fallback_rate = level1_01yr[TERM_RATE].iloc[0]
                    logger.info(f"[TERM] Year {year}: Using level 1, 0-1 rate {fallback_rate:.4f} as proxy for level 2, 0-1")
                else:
                    # Last resort - use global average
                    fallback_rate = hz[TERM_RATE].mean()
                    logger.info(f"[TERM] Year {year}: Using global average rate {fallback_rate:.4f} as last resort fallback")
            
            # Apply the fallback rate
            merged_df.loc[level2_01yr_mask, TERM_RATE] = fallback_rate
            logger.info(f"[TERM] Year {year}: Applied fallback rate {fallback_rate:.4f} to {level2_01yr_count} level 2 employees with 0-1 tenure")
            
            # Update missing rates count after applying fallback
            missing_rates = merged_df[TERM_RATE].isna()
        
        # For any remaining NAs, fill with 0 (default behavior)
        remaining_na_count = missing_rates.sum()
        if remaining_na_count > 0:
            logger.warning(
                f"[TERM] Year {year}: {remaining_na_count} employees still missing {TERM_RATE} after fallback logic. "
                f"Filling with 0. Check hazard table coverage for all {EMP_LEVEL} and {EMP_TENURE_BAND} combinations."
            )
            merged_df[TERM_RATE] = merged_df[TERM_RATE].fillna(0)
        else:
            logger.info(f"[TERM] Year {year}: All missing term rates successfully handled with fallback logic.")
    
    logger.debug(f"[TERM] Year {year}: {n} employees processed, {zero_rates.sum()} with zero {TERM_RATE} after merge.")
    
    # Add distribution of term rates by level and tenure band for transparency
    rate_by_level_tenure = merged_df.groupby([EMP_LEVEL, EMP_TENURE_BAND])[TERM_RATE].mean().reset_index()
    logger.debug(f"[TERM DIAGNOSTIC YR={year}] Term rate by level and tenure:\n{rate_by_level_tenure.to_string()}")
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
        # Only compare non-NA IDs and use EMP_ID constant
        valid_events = (e for e in term_events if pd.notna(e[EMP_ID]))
        if any(e[EMP_ID] == emp for e in valid_events):
            logger.warning(f"Skipping duplicate termination for employee {emp}")
            continue
            
        # Termination event
        hire_date = snapshot.loc[snapshot[EMP_ID] == emp, EMP_HIRE_DATE].iloc[0] if EMP_HIRE_DATE in snapshot.columns and not snapshot.loc[snapshot[EMP_ID] == emp, EMP_HIRE_DATE].isna().iloc[0] else None
        tenure_days = int((dt - hire_date).days) if hire_date is not None else None
        
        # Safeguard: Skip processing if emp is NA (should ideally be caught by merged_df cleaning)
        if pd.isna(emp):
            logger.warning(
                f"[TERM] Year {year}: Skipping termination processing for an entry with NA employee ID ('{emp}') "
                f"within the main processing loop (safeguard)."
            )
            continue

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
        
    # Log new hire tenure distribution for diagnostics
    if EMP_TENURE_BAND in df_nh.columns:
        # Log distribution before standardization
        tenure_counts = df_nh[EMP_TENURE_BAND].value_counts().to_dict()
        logger.debug(f"[TERM NEW HIRE DIAGNOSTIC YR={year}] New hire tenure distribution before standardization: {tenure_counts}")
        
        # Standardize tenure bands to ensure consistent format for new hires
        df_nh[EMP_TENURE_BAND] = df_nh[EMP_TENURE_BAND].map(standardize_tenure_band)
        
        # Log distribution after standardization
        tenure_counts_std = df_nh[EMP_TENURE_BAND].value_counts().to_dict()
        logger.debug(f"[TERM NEW HIRE DIAGNOSTIC YR={year}] New hire tenure distribution after standardization: {tenure_counts_std}")
        
        # CRITICAL: Also standardize the hazard slice to ensure matching
        hazard_slice[EMP_TENURE_BAND] = hazard_slice[EMP_TENURE_BAND].map(standardize_tenure_band)
        
        # Log unique tenure bands from both sides to verify they match
        logger.warning(f"[TERM NEW HIRE DIAGNOSTIC YR={year}] Unique new hire tenure bands: {df_nh[EMP_TENURE_BAND].unique().tolist()}")
        logger.warning(f"[TERM NEW HIRE DIAGNOSTIC YR={year}] Unique hazard tenure bands: {hazard_slice[EMP_TENURE_BAND].unique().tolist() if EMP_TENURE_BAND in hazard_slice.columns else 'N/A'}")
        
    # Use the pre-filtered hazard_slice for this year
    # Ensure hazard_slice has standardized tenure bands before using rates
    if EMP_TENURE_BAND in hazard_slice.columns:
        # One more standardization to be absolutely certain
        hazard_slice[EMP_TENURE_BAND] = hazard_slice[EMP_TENURE_BAND].map(standardize_tenure_band)
    
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
    
    # Get hire dates for all losers (with validation)
    loser_hire_dates = []
    for emp in losers:
        hire_date_series = snapshot.loc[snapshot[EMP_ID] == emp, EMP_HIRE_DATE]
        if hire_date_series.empty:
            logger.warning(f"[TERM NEW HIRE] Year {year}: Could not find hire date for employee {emp}. Skipping termination.")
            continue
        hire_date = hire_date_series.iloc[0]
        loser_hire_dates.append(hire_date)
    
    # Update losers list to match the filtered hire dates
    if len(loser_hire_dates) < len(losers):
        logger.warning(f"[TERM NEW HIRE] Year {year}: Reduced new hire terminations from {len(losers)} to {len(loser_hire_dates)} due to missing hire dates")
        losers = [emp for i, emp in enumerate(losers) if i < len(loser_hire_dates)]
    
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
