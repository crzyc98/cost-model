# cost_model/engines/comp.py

import pandas as pd
import numpy as np
import json
from typing import List
from cost_model.state.event_log import EVENT_COLS, EVT_COMP
from cost_model.utils.columns import EMP_ID, EMP_TERM_DATE, EMP_ROLE, EMP_GROSS_COMP
import logging

logger = logging.getLogger(__name__)

def bump(
    snapshot: pd.DataFrame,
    hazard_slice: pd.DataFrame,
    as_of: pd.Timestamp
) -> List[pd.DataFrame]:
    """
    Apply the comp_raise_pct from hazard_slice for each active employee,
    and emit one DataFrame of compensation bump events adhering to EVENT_COLS.
    """
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
    df = active.merge(
        hazard_slice[[EMP_ROLE, "tenure_band", "comp_raise_pct"]],
        on=[EMP_ROLE, "tenure_band"],
        how="left"
    ).fillna({"comp_raise_pct": 0})

    # 4) Only rows with a positive raise
    df = df[df["comp_raise_pct"] > 0].copy()
    if df.empty:
        return [pd.DataFrame(columns=EVENT_COLS)]

    # 5) Build event columns
    df.reset_index(drop=True, inplace=True)
    df["event_id"]    = df.index.map(lambda i: f"evt_comp_{year}_{i:04d}")
    df["event_type"]  = EVT_COMP
    df["event_date"]  = as_of
    df["year"]        = year
    df["value_num"]   = df["comp_raise_pct"]
    df["old_comp"]    = df.get(EMP_GROSS_COMP, np.nan)
    df["new_comp"]    = df["old_comp"] * (1 + df["comp_raise_pct"])
    df["pct"]         = df["comp_raise_pct"]
    df["value_json"]  = df.apply(
        lambda row: json.dumps({
            "old_comp": row["old_comp"],
            "new_comp": row["new_comp"],
            "pct":      row["pct"]
        }),
        axis=1
    )
    df["notes"]       = None  # or fill in a descriptive string

    # 6) Select and return
    events = df[EVENT_COLS]
    return [events]