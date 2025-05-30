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
    Apply the comp_raise_pct from hazard_slice for each active employee,
    and emit one DataFrame of compensation bump events adhering to EVENT_COLS.
    """
    # 1) Derive year and filter active
    year = int(hazard_slice["simulation_year"].iloc[0])
    as_of = pd.Timestamp(as_of)

    # NEW – makes sure the two merge keys are always present
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

    # 3) Merge in the raise pct
    # Dynamically choose the level column based on what's present in hazard_slice
    level_col = 'level' if 'level' in hazard_slice.columns else 'employee_level'
    if level_col not in hazard_slice.columns:
        # Fallback to any column that might represent employee level
        possible_level_cols = [col for col in hazard_slice.columns if 'level' in col.lower()]
        if possible_level_cols:
            level_col = possible_level_cols[0]
        else:
            raise KeyError(f"No level column found in hazard_slice. Available columns: {hazard_slice.columns.tolist()}")

    logger.debug(f"[COMP] Using column '{level_col}' as EMP_LEVEL merge key; hazard rows={len(hazard_slice)}")

    hz = hazard_slice[[level_col, EMP_TENURE_BAND, 'comp_raise_pct']].rename(columns={level_col: EMP_LEVEL})

    # Deduplicate hazard slice to prevent duplicate employee records after merge
    hz_dedup = hz.drop_duplicates(subset=[EMP_LEVEL, EMP_TENURE_BAND])
    if len(hz_dedup) < len(hz):
        # Provide more detailed diagnostic information about the duplicates
        duplicates = hz[hz.duplicated(subset=[EMP_LEVEL, EMP_TENURE_BAND], keep=False)]
        duplicate_combos = duplicates[[EMP_LEVEL, EMP_TENURE_BAND, 'comp_raise_pct']].drop_duplicates()

        logger.debug(f"[COMP] Found {len(hz) - len(hz_dedup)} duplicate (level, tenure_band) combinations in hazard slice for year {year}")
        logger.debug(f"[COMP] Duplicate combinations: {duplicate_combos.to_dict('records')}")

        # Check if the duplicates have different comp_raise_pct values
        conflicting_rates = duplicates.groupby([EMP_LEVEL, EMP_TENURE_BAND])['comp_raise_pct'].nunique()
        conflicts = conflicting_rates[conflicting_rates > 1]

        if not conflicts.empty:
            # Create a unique signature for this set of conflicts to avoid repeated warnings
            conflict_signature = tuple(sorted((level, tenure_band) for (level, tenure_band), _ in conflicts.items()))

            if conflict_signature not in _warned_conflicts:
                # Add to cache to prevent repeated warnings
                _warned_conflicts.add(conflict_signature)

                # Create a more concise summary of conflicts
                conflict_summary = []
                for (level, tenure_band), _ in conflicts.items():
                    rates = duplicates[(duplicates[EMP_LEVEL] == level) & (duplicates[EMP_TENURE_BAND] == tenure_band)]['comp_raise_pct'].unique()
                    rates_str = f"[{', '.join(f'{r:.3f}' for r in sorted(rates))}]"
                    conflict_summary.append(f"Level {level}/Tenure {tenure_band}: {rates_str}")

                logger.warning(f"[COMP] Hazard table has conflicting comp_raise_pct values for {len(conflicts)} combinations (using first occurrence):")
                logger.warning(f"[COMP] Conflicts: {'; '.join(conflict_summary)}")
                logger.warning(f"[COMP] Fix: Ensure hazard table has unique (year, level, tenure_band) combinations with consistent rates")
            else:
                # We've already warned about these conflicts, just log at debug level
                logger.debug(f"[COMP] Skipping repeated warning for {len(conflicts)} known conflicting (level, tenure_band) combinations")
        else:
            # Duplicates have same rates, so this is just redundant data
            logger.debug(f"[COMP] Duplicate (level, tenure_band) combinations have consistent comp_raise_pct values - removing redundant rows")

    df = active.merge(
        hz_dedup,
        on=[EMP_LEVEL, EMP_TENURE_BAND],
        how='left'
    ).fillna({'comp_raise_pct': 0})

    # 4) Only rows with a positive raise
    excluded = active[~active[EMP_ID].isin(df[EMP_ID])]
    if not excluded.empty:
        logger.warning(f"[COMP.BUMP] {len(excluded)} active employees excluded from comp bump due to missing hazard table match. EMP_IDs: {excluded[EMP_ID].tolist()}")
    df = df[df["comp_raise_pct"] > 0].copy()
    if df.empty:
        # Even if bump returns an empty list, keep a placeholder DataFrame
        # so that snapshot.update() later still sees the COLA rows.
        comp_events = [pd.DataFrame(columns=EVENT_COLS)]
    else:
        # Process the compensation events normally
        # --- 3. compute old + new comp ---
        df["old_comp"] = df[EMP_GROSS_COMP].astype(float).fillna(0.0)
        df["new_comp"] = (df["old_comp"] * (1 + df["comp_raise_pct"])).round(2)

        # tenure in years as of Jan1
        jan1 = pd.Timestamp(f"{year}-01-01")
        hire_dates = pd.to_datetime(df[EMP_HIRE_DATE], errors="coerce")
        tenure = ((jan1 - hire_dates).dt.days / 365.25).astype(int)
        df[EMP_TENURE] = tenure  # REQUIRED for sampler's mask
        mask_second = tenure == 1
        if mask_second.any():
            sampler = DefaultSalarySampler(rng=rng)
            # Use normal distribution for second-year bumps
            mean = df.loc[mask_second, "comp_raise_pct"].mean()
            df.loc[mask_second, "new_comp"] = sampler.sample_second_year(
                df.loc[mask_second],
                comp_col="old_comp",
                dist={"type": "normal", "mean": mean, "std": 0.01},
                rng=rng
            )

        df["event_id"]    = df.index.map(lambda i: f"evt_comp_{year}_{i:04d}")
        df["event_time"]  = as_of
        df["event_type"]  = EVT_COMP
        # **this** is what snapshot.update will write into EMP_GROSS_COMP
        df["value_num"]   = df["new_comp"]
        # keep pct / audit in JSON
        import json as json_module  # Import with different name to avoid scope issues
        df["value_json"] = df.apply(lambda row: json_module.dumps({
            "reason": "annual_raise",
            "pct": row["comp_raise_pct"],
            "old_comp": row["old_comp"],
            "new_comp": row["new_comp"]
        }), axis=1)
        df["meta"] = df.apply(lambda row: f"Annual raise for {row[EMP_ID]}: {row['old_comp']} -> {row['new_comp']} (+{row['comp_raise_pct']*100:.2f}%)", axis=1)
        # Add simulation year for EVENT_COLS compliance
        df[SIMULATION_YEAR] = year
        # Remove notes column if present
        if "notes" in df.columns:
            df = df.drop(columns=["notes"])

        # 6) Slice to exactly the EVENT_COLS schema
        events = df[EVENT_COLS]

        # Assert/Log uniqueness of EMP_IDs
        if events[EMP_ID].duplicated().any():
            logger.error(f"[COMP.BUMP] Duplicate EMP_IDs found in comp bump events: {events[EMP_ID][events[EMP_ID].duplicated()].tolist()}")
            raise ValueError("Duplicate EMP_IDs in comp bump events!")

        # Debug log: summary of bumps
        logger.debug(f"[COMP.BUMP] Applied {len(events)} comp bumps for year {year}. Pct range: {events['value_json'].apply(lambda x: json.loads(x)['pct']).min():.2%} to {events['value_json'].apply(lambda x: json.loads(x)['pct']).max():.2%}")

        comp_events = [events]

    # Generate COLA events FIRST and apply them to get updated compensation
    cola_events_list = generate_cola_events(
        snapshot=snapshot,
        hazard_slice=hazard_slice,
        as_of=as_of,
        rng=rng
    )

    # Apply COLA events to snapshot to get updated compensation for annual raises
    if cola_events_list and len(cola_events_list) > 0 and not cola_events_list[0].empty:
        logger.debug("[COMP.BUMP] Applying COLA events to snapshot before calculating annual raises")
        # Create a temporary snapshot with COLA applied
        from cost_model.state.snapshot_update import _apply_existing_updates
        temp_snapshot = snapshot.copy()
        cola_df = cola_events_list[0]
        temp_snapshot = _apply_existing_updates(temp_snapshot, cola_df, year)

        # Recalculate annual raises using COLA-adjusted compensation
        if not df.empty:
            # Update the compensation values in df to use COLA-adjusted values
            for idx in df.index:
                emp_id = df.at[idx, EMP_ID]
                if emp_id in temp_snapshot.index:
                    updated_comp = temp_snapshot.at[emp_id, EMP_GROSS_COMP]
                    # Recalculate the raise using updated compensation
                    old_comp = df.at[idx, "old_comp"]
                    raise_pct = df.at[idx, "comp_raise_pct"]
                    new_comp = updated_comp * (1 + raise_pct)

                    # Update the event values
                    df.at[idx, "old_comp"] = updated_comp
                    df.at[idx, "new_comp"] = new_comp
                    df.at[idx, "value_num"] = new_comp

                    # Update the value_json
                    import json as json_module
                    value_json = json_module.loads(df.at[idx, "value_json"])
                    value_json["old_comp"] = updated_comp
                    value_json["new_comp"] = new_comp
                    df.at[idx, "value_json"] = json_module.dumps(value_json)

                    logger.debug(f"[COMP.BUMP] Updated raise for {emp_id}: {old_comp:.2f} → {updated_comp:.2f} → {new_comp:.2f}")

            comp_events = [df]

        logger.info(f"[COMP.BUMP] Recalculated {len(df) if not df.empty else 0} annual raises using COLA-adjusted compensation")

    # Combine COLA events with updated annual raise events
    all_events = cola_events_list + comp_events
    return all_events


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