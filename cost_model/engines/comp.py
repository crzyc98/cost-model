# cost_model/engines/comp.py

import pandas as pd
import numpy as np
import json
import logging
from typing import List

from cost_model.state.event_log import EVENT_COLS, EVT_COMP
from cost_model.utils.columns import (
    EMP_ID, EMP_TERM_DATE, EMP_ROLE, EMP_GROSS_COMP, EMP_HIRE_DATE
)
from cost_model.dynamics.sampling.salary import DefaultSalarySampler

logger = logging.getLogger(__name__)
from typing import List
from cost_model.state.event_log import EVENT_COLS, EVT_COMP
from cost_model.utils.columns import EMP_ID, EMP_TERM_DATE, EMP_ROLE, EMP_GROSS_COMP
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
    from cost_model.utils.columns import EMP_TENURE
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
    hz = hazard_slice[['role', 'tenure_band', 'comp_raise_pct']].rename(columns={'role': EMP_ROLE})
    df = active.merge(
        hz,
        on=[EMP_ROLE, 'tenure_band'],
        how='left'
    ).fillna({'comp_raise_pct': 0})

    # 4) Only rows with a positive raise
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
    df["tenure"] = tenure  # REQUIRED for sampler's mask
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
    return [events]