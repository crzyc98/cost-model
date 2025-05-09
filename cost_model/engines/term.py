# cost_model/engines/term.py

from typing import List
import pandas as pd
import numpy as np
from cost_model.state.event_log import EVENT_COLS, EVT_TERM
from cost_model.utils.columns import EMP_ID, EMP_TERM_DATE, EMP_ROLE, EMP_HIRE_DATE
import math
import logging

logger = logging.getLogger(__name__)

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
    deterministic: bool,
) -> List[pd.DataFrame]:
    """
    Simulate terminations across the active population in `snapshot`.
    """
    # Use the year from the snapshot or default to current year
    # This 'year' is for determining 'as_of', the simulation year for which hazard_slice is already filtered should be used.
    # We expect hazard_slice to be pre-filtered for a single year.
    sim_year_from_hazard = hazard_slice['year'].iloc[0] if not hazard_slice.empty and 'year' in hazard_slice.columns else 'UNKNOWN_YEAR_TERM'
    # logger.info(f"[TERM.RUN YR={sim_year_from_hazard}] Input snapshot shape: {snapshot.shape}") # Too verbose for every call

    year_for_dates = pd.Timestamp.now().year # Default for random date generation if no other info
    if not snapshot.empty and "event_time" in snapshot.columns and not snapshot["event_time"].isna().all():
        # This logic might pick up a past event time; we care about the current simulation period's year.
        # Better to rely on sim_year_from_hazard for decision making if possible.
        year_for_dates = pd.Timestamp(snapshot["event_time"].min()).year 
    # For term events, the year should be the one from hazard_slice, which is current sim year.
    year_for_dates = sim_year_from_hazard if isinstance(sim_year_from_hazard, (int, float)) else year_for_dates

    as_of = pd.Timestamp(f"{year_for_dates}-01-01")
    active = snapshot[
        (snapshot[EMP_TERM_DATE].isna()) | (snapshot[EMP_TERM_DATE] > as_of)
    ].copy()
    if EMP_ID not in active.columns:
        active = active.reset_index()
    n = len(active)

    if str(sim_year_from_hazard) == '2026': # Log only for 2026 to reduce noise
        logger.info(f"[TERM.RUN YR={sim_year_from_hazard}] Snapshot EOY active count (passed to term.run): {snapshot[snapshot[EMP_TERM_DATE].isna()].shape[0]}")
        logger.info(f"[TERM.RUN YR={sim_year_from_hazard}] Number of active employees for termination consideration (n): {n}")
        logger.info(f"[TERM.RUN YR={sim_year_from_hazard}] as_of date for filtering: {as_of}")

    if n == 0:
        if str(sim_year_from_hazard) == '2026':
            logger.info(f"[TERM.RUN YR={sim_year_from_hazard}] No active employees (n=0), returning empty term events.")
        return [pd.DataFrame(columns=EVENT_COLS)]
    # Merge in term_rate
    df = active.merge(
        hazard_slice[[EMP_ROLE, "tenure_band", "term_rate"]],
        on=[EMP_ROLE, "tenure_band"],
        how="left",
    )

    if str(sim_year_from_hazard) == '2026':
        logger.info(f"[TERM.RUN YR={sim_year_from_hazard}] DataFrame 'df' shape after merge: {df.shape}")
        logger.info(f"[TERM.RUN YR={sim_year_from_hazard}] Sample of term_rate in df (first 5):\n{df[['term_rate']].head().to_string()}")
        logger.info(f"[TERM.RUN YR={sim_year_from_hazard}] NaN term_rates after merge: {df['term_rate'].isna().sum()}")

    # Decide who terminates
    if deterministic:
        rate = df["term_rate"].mean() if not df["term_rate"].isna().all() else 0
        k = int(math.ceil(n * rate))
        if str(sim_year_from_hazard) == '2026':
            logger.info(f"[TERM.RUN YR={sim_year_from_hazard}] Deterministic mode. Mean rate: {rate}, Num to term (k): {k}")
        if k == 0:
            return [pd.DataFrame(columns=EVENT_COLS)]
        losers = rng.choice(df[EMP_ID], size=k, replace=False)
    else:
        probs = df["term_rate"].fillna(0).values
        draw = rng.random(n)
        losers = df.loc[draw < probs, EMP_ID].tolist()
        if str(sim_year_from_hazard) == '2026':
            logger.info(f"[TERM.RUN YR={sim_year_from_hazard}] Probabilistic mode. Probs array sample (first 5): {probs[:5]}")
            logger.info(f"[TERM.RUN YR={sim_year_from_hazard}] Number of losers: {len(losers)}")
            # logger.info(f"[TERM.RUN YR={sim_year_from_hazard}] Draws sample (first 5): {draw[:5]}") # Potentially verbose

    if not losers:
        if str(sim_year_from_hazard) == '2026':
             logger.info(f"[TERM.RUN YR={sim_year_from_hazard}] No losers identified, returning empty term events.")
        return [pd.DataFrame(columns=EVENT_COLS)]

    # Assign random dates
    dates = _random_dates_in_year(year_for_dates, len(losers), rng)
    # Build events
    events_data = []
    for idx, (emp, dt) in enumerate(zip(losers, dates)):
        event_id_str = f"evt_term_{sim_year_from_hazard}_{idx:04d}"
        events_data.append({
            "event_id": event_id_str,
            EMP_ID: emp,
            "event_type": EVT_TERM,
            "event_date": dt,
            "year": sim_year_from_hazard, # Ensure year is correctly assigned
            "value_num": np.nan,
            "value_json": None, # Align with EVENT_COLS structure
            "notes": f"Termination for {emp} in {sim_year_from_hazard}" # Align with EVENT_COLS structure
        })
    term_events = pd.DataFrame(events_data, columns=EVENT_COLS) # Create DataFrame with explicit columns

    logger.debug(f"Generated {len(term_events)} termination events for year {sim_year_from_hazard}.")
    return [term_events]
