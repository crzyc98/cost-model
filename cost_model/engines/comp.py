# cost_model/engines/comp.py
"""
Engine for simulating compensation changes during workforce simulations.
QuickStart: see docs/cost_model/engines/comp.md
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import List

from cost_model.state.event_log import EVENT_COLS, EVT_COMP
from cost_model.state.schema import (
    EMP_ID, EMP_TERM_DATE, EMP_LEVEL, EMP_GROSS_COMP, EMP_HIRE_DATE,
    EMP_TENURE, EMP_TENURE_BAND, SIMULATION_YEAR
)
from cost_model.dynamics.sampling.salary import DefaultSalarySampler
from cost_model.engines.cola import cola

logger = logging.getLogger(__name__)

# Cache to track conflicts we've already warned about to avoid repetitive warnings
_warned_conflicts = set()


def clear_warning_cache():
    """Clear the warning cache. Call this at the start of a new simulation run."""
    global _warned_conflicts
    _warned_conflicts.clear()
    logger.debug("[COMP] Cleared compensation conflict warning cache")


def _ensure_level_and_band(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Make sure EMP_LEVEL and EMP_TENURE_BAND exist for every row.
    If missing, derive them on the fly.
    """
    import math
    # 1. level: fall back to `employee_level` or cast numeric levels to str
    if EMP_LEVEL not in df.columns:
        if 'employee_level' in df.columns:
            df[EMP_LEVEL] = df['employee_level']
        else:
            df[EMP_LEVEL] = 0   # put everyone in bucket 0 so they at least get *a* match

    # 2. tenure_band: cheap vectorised calc off hire_date
    if EMP_TENURE_BAND not in df.columns or df[EMP_TENURE_BAND].isna().any():
        jan1 = pd.Timestamp(f"{year}-01-01")
        hire_dates = pd.to_datetime(df[EMP_HIRE_DATE], errors='coerce')
        tenure_years = ((jan1 - hire_dates).dt.days / 365.25).fillna(0)

        def band(t):
            if t < 1:  return '0-1'
            elif t < 3: return '1-3'
            elif t < 5: return '3-5'
            elif t < 10: return '5-10'
            else: return '10+'
        df[EMP_TENURE_BAND] = tenure_years.map(band)

    return df

def bump(
    snapshot: pd.DataFrame,
    hazard_slice: pd.DataFrame,
    as_of: pd.Timestamp,
    rng: np.random.Generator
) -> List[pd.DataFrame]:
    """
    Generate granular compensation events based on hazard table data.

    Implements the conceptual mapping:
    - EVT_COLA → cola_pct (inflation adjustment for everyone)
    - EVT_COMP → merit_raise_pct (merit increase for active employees)
    - EVT_PROMOTION → promotion_raise_pct (handled separately in promotion logic)

    Returns list of event DataFrames: [cola_events, merit_events]
    """
    # 1) Derive year and filter active
    year = int(hazard_slice["simulation_year"].iloc[0])
    as_of = pd.Timestamp(as_of)

    # Ensure the two merge keys are always present
    snapshot = _ensure_level_and_band(snapshot.copy(), year)

    active = snapshot[
        snapshot[EMP_TERM_DATE].isna() | (snapshot[EMP_TERM_DATE] > as_of)
    ].copy()

    # 2) Ensure EMP_ID is a column
    if EMP_ID not in active.columns:
        if active.index.name == EMP_ID:
            active = active.reset_index()
        else:
            raise ValueError(f"{EMP_ID} not found in active snapshot")

    # 3) Generate COLA events first (applies to everyone)
    cola_events = generate_cola_events(
        snapshot=snapshot,
        hazard_slice=hazard_slice,
        as_of=as_of,
        rng=rng
    )

    # 4) Generate merit raise events (EVT_COMP) using merit_raise_pct
    merit_events = _generate_merit_events(
        active=active,
        hazard_slice=hazard_slice,
        as_of=as_of,
        year=year
    )

    # Return both event types
    return cola_events + merit_events


def _generate_merit_events(
    active: pd.DataFrame,
    hazard_slice: pd.DataFrame,
    as_of: pd.Timestamp,
    year: int
) -> List[pd.DataFrame]:
    """
    Generate EVT_COMP events using merit_raise_pct from hazard table.

    Args:
        active: Active employees DataFrame
        hazard_slice: Hazard table slice for the year
        as_of: Event timestamp
        year: Simulation year

    Returns:
        List containing merit event DataFrame
    """
    # Check if we have the new granular schema or need to fall back to legacy
    if 'merit_raise_pct' in hazard_slice.columns:
        merit_col = 'merit_raise_pct'
        logger.info("[COMP] Using new granular schema: merit_raise_pct for EVT_COMP events")
    elif 'comp_raise_pct' in hazard_slice.columns:
        merit_col = 'comp_raise_pct'
        logger.info("[COMP] Using legacy schema: comp_raise_pct for EVT_COMP events")
    else:
        logger.warning("[COMP] No merit raise column found in hazard slice. Returning empty events.")
        return [pd.DataFrame(columns=EVENT_COLS)]

    # Determine level column dynamically
    level_col = 'level' if 'level' in hazard_slice.columns else 'employee_level'
    if level_col not in hazard_slice.columns:
        possible_level_cols = [col for col in hazard_slice.columns if 'level' in col.lower()]
        if possible_level_cols:
            level_col = possible_level_cols[0]
        else:
            raise KeyError(f"No level column found in hazard_slice. Available columns: {hazard_slice.columns.tolist()}")

    logger.debug(f"[COMP] Using column '{level_col}' as EMP_LEVEL merge key; hazard rows={len(hazard_slice)}")

    # Prepare hazard slice for merge
    hz = hazard_slice[[level_col, EMP_TENURE_BAND, merit_col]].rename(columns={level_col: EMP_LEVEL})

    # Deduplicate hazard slice to prevent duplicate employee records after merge
    hz_dedup = hz.drop_duplicates(subset=[EMP_LEVEL, EMP_TENURE_BAND])
    if len(hz_dedup) < len(hz):
        logger.debug(f"[COMP] Removed {len(hz) - len(hz_dedup)} duplicate (level, tenure_band) combinations in hazard slice for year {year}")

    # Merge active employees with hazard data
    df = active.merge(
        hz_dedup,
        on=[EMP_LEVEL, EMP_TENURE_BAND],
        how='left'
    ).fillna({merit_col: 0})

    # Only process employees with positive merit raises
    excluded = active[~active[EMP_ID].isin(df[EMP_ID])]
    if not excluded.empty:
        logger.warning(f"[COMP.MERIT] {len(excluded)} active employees excluded from merit bump due to missing hazard table match. EMP_IDs: {excluded[EMP_ID].tolist()}")

    df = df[df[merit_col] > 0].copy()
    if df.empty:
        logger.info("[COMP.MERIT] No employees eligible for merit raises")
        return [pd.DataFrame(columns=EVENT_COLS)]

    # Calculate new compensation based on merit raise percentage
    df["old_comp"] = df[EMP_GROSS_COMP].astype(float).fillna(0.0)
    df["new_comp"] = (df["old_comp"] * (1 + df[merit_col])).round(2)

    # Create EVT_COMP events
    df["event_id"] = df.index.map(lambda i: f"evt_comp_{year}_{i:04d}")
    df["event_time"] = as_of
    df["event_type"] = EVT_COMP
    df["value_num"] = df["new_comp"]  # New total compensation

    # Store metadata in JSON
    import json as json_module
    df["value_json"] = df.apply(lambda row: json_module.dumps({
        "reason": "merit_raise",
        "pct": row[merit_col],
        "old_comp": row["old_comp"],
        "new_comp": row["new_comp"]
    }), axis=1)

    df["meta"] = df.apply(lambda row: f"Merit raise for {row[EMP_ID]}: {row['old_comp']} -> {row['new_comp']} (+{row[merit_col]*100:.2f}%)", axis=1)
    df[SIMULATION_YEAR] = year

    # Select only EVENT_COLS
    events = df[EVENT_COLS]

    logger.info(f"[COMP.MERIT] Generated {len(events)} merit raise events for year {year}")
    return [events]


def extract_promotion_raise_config_from_hazard(hazard_slice: pd.DataFrame) -> dict:
    """
    Extract promotion raise configuration from hazard table's promotion_raise_pct column.

    This function implements the conceptual mapping: EVT_PROMOTION → promotion_raise_pct

    Args:
        hazard_slice: Hazard table slice containing promotion_raise_pct

    Returns:
        Dictionary mapping "{from_level}_to_{to_level}" to raise percentage
        Compatible with existing promotion system expectations
    """
    if 'promotion_raise_pct' not in hazard_slice.columns:
        logger.warning("[COMP.PROMOTION] No promotion_raise_pct column in hazard slice. Using default config.")
        return {}

    # Determine level column dynamically
    level_col = 'level' if 'level' in hazard_slice.columns else 'employee_level'
    if level_col not in hazard_slice.columns:
        possible_level_cols = [col for col in hazard_slice.columns if 'level' in col.lower()]
        if possible_level_cols:
            level_col = possible_level_cols[0]
        else:
            logger.warning("[COMP.PROMOTION] No level column found in hazard slice. Using default config.")
            return {}

    # Extract unique promotion raise percentages by level
    promo_config = {}
    unique_levels = sorted(hazard_slice[level_col].unique())

    for level in unique_levels:
        level_data = hazard_slice[hazard_slice[level_col] == level]
        promotion_raise_pct = level_data['promotion_raise_pct'].iloc[0]  # Take first occurrence

        # Create mapping for promotion from this level to next level
        next_level = level + 1
        key = f"{level}_to_{next_level}"
        promo_config[key] = float(promotion_raise_pct)

        logger.debug(f"[COMP.PROMOTION] Level {level} → {next_level}: {promotion_raise_pct:.1%}")

    logger.info(f"[COMP.PROMOTION] Extracted promotion raise config from hazard table: {len(promo_config)} mappings")
    return promo_config


def generate_cola_events(
    snapshot: pd.DataFrame,
    hazard_slice: pd.DataFrame,
    as_of: pd.Timestamp,
    days_into_year: int = 0,
    jitter_days: int = 0,
    rng: np.random.Generator = None
) -> List[pd.DataFrame]:
    """
    Generate COLA (Cost of Living Adjustment) events for active employees.

    This function integrates the dedicated cola() function from cost_model.engines.cola
    into the main compensation engine workflow.

    Args:
        snapshot: Current workforce snapshot
        hazard_slice: Hazard table slice containing cola_pct
        as_of: Base timestamp for COLA events
        days_into_year: Days to add to as_of for COLA timing (default: 0)
        jitter_days: Optional random jitter in days for event timing (default: 0)
        rng: Random number generator for jitter (optional)

    Returns:
        List containing a single DataFrame of EVT_COLA events
    """
    logger.info(f"[COMP.COLA] Generating COLA events for year {int(hazard_slice['simulation_year'].iloc[0])}")

    try:
        # Get the COLA rate from the hazard slice
        # Use the maximum COLA rate if there are multiple rows (to ensure we don't miss COLA)
        if 'cola_pct' not in hazard_slice.columns:
            logger.warning("[COMP.COLA] No cola_pct column in hazard_slice. Returning empty events.")
            return [pd.DataFrame(columns=EVENT_COLS)]

        cola_pct = float(hazard_slice["cola_pct"].max())  # Use max to ensure we get non-zero COLA if it exists
        year = int(hazard_slice['simulation_year'].iloc[0])

        # Create a simple hazard slice for the cola() function
        simple_hazard_slice = pd.DataFrame({
            'simulation_year': [year],
            'cola_pct': [cola_pct]
        })

        logger.info(f"[COMP.COLA] Using COLA rate {cola_pct:.1%} for year {year}")

        # Call the dedicated cola function with the simplified hazard slice
        cola_events = cola(
            snapshot=snapshot,
            hazard_slice=simple_hazard_slice,
            as_of=as_of,
            days_into_year=days_into_year,
            jitter_days=jitter_days,
            rng=rng
        )

        # Log summary
        if cola_events and len(cola_events) > 0 and not cola_events[0].empty:
            events_df = cola_events[0]
            logger.info(f"[COMP.COLA] Generated {len(events_df)} COLA events with rate {cola_pct:.1%}")
        else:
            logger.info("[COMP.COLA] No COLA events generated (rate is 0% or no active employees)")

        return cola_events

    except KeyError as e:
        if "cola_pct" in str(e):
            logger.warning("[COMP.COLA] No cola_pct found in hazard_slice. Returning empty events.")
            return [pd.DataFrame(columns=EVENT_COLS)]
        else:
            raise
    except Exception as e:
        logger.error(f"[COMP.COLA] Error generating COLA events: {e}")
        return [pd.DataFrame(columns=EVENT_COLS)]