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
    EMP_ID, EMP_TERM_DATE, EMP_ROLE, EMP_GROSS_COMP, EMP_HIRE_DATE,
    EMP_TENURE, EMP_TENURE_BAND
)
from cost_model.dynamics.sampling.salary import DefaultSalarySampler
from cost_model.state.event_log import EVENT_COLS, EVT_COMP
import logging

logger = logging.getLogger(__name__)

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
    from cost_model.state.schema import EMP_TENURE
    # 1) Derive year and filter active
    year = int(hazard_slice["simulation_year"].iloc[0])
    as_of = pd.Timestamp(as_of)
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
    # ensure 'role' from hazard_slice is mapped to EMP_ROLE for merge
    hz = hazard_slice[['role', EMP_TENURE_BAND, 'comp_raise_pct']].rename(columns={'role': EMP_ROLE})
    df = active.merge(
        hz,
        on=[EMP_ROLE, EMP_TENURE_BAND],
        how='left'
    ).fillna({'comp_raise_pct': 0})

    # 4) Only rows with a positive raise
    excluded = active[~active[EMP_ID].isin(df[EMP_ID])]
    if not excluded.empty:
        logger.warning(f"[COMP.BUMP] {len(excluded)} active employees excluded from comp bump due to missing hazard table match. EMP_IDs: {excluded[EMP_ID].tolist()}")
    df = df[df["comp_raise_pct"] > 0].copy()
    if df.empty:
        return [pd.DataFrame(columns=EVENT_COLS)]

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
    df["value_json"] = df.apply(lambda row: json.dumps({
        "reason": "cola_bump",
        "pct": row["comp_raise_pct"],
        "old_comp": row["old_comp"],
        "new_comp": row["new_comp"]
    }), axis=1)
    df["meta"] = df.apply(lambda row: f"COLA bump for {row[EMP_ID]}: {row['old_comp']} -> {row['new_comp']} (+{row['comp_raise_pct']*100:.2f}%)", axis=1)
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
    return [events]