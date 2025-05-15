# cost_model/engines/compensation.py
"""
Functions related to compensation changes during simulation dynamics.
QuickStart: see docs/cost_model/engines/compensation.md
"""

import pandas as pd
import numpy as np
import json
from typing import Optional, Dict, Any
from cost_model.state.event_log import EVENT_COLS, EVT_RAISE, create_event
from cost_model.utils.columns import EMP_ID, EMP_GROSS_COMP, EMP_LEVEL, EMP_LEVEL_SOURCE

def update_salary(
    df: pd.DataFrame,
    params: Dict[str, Any],
    rng: Optional[np.random.Generator] = None
) -> pd.DataFrame:
    """
    Apply COLA, promotion raise, and merit distribution to EMP_GROSS_COMP in that order.
    Logs EVT_RAISE events for each step and returns a new DataFrame of raise events.

    params should include:
      - COLA_rate: float
      - promo_raise_pct: Dict[str, float]
      - merit_dist: Dict[str, Dict[str, float]]
      - event_time: pd.Timestamp (optional)
    """
    rng = rng or np.random.default_rng()
    event_time = getattr(params, 'event_time', pd.Timestamp.now())
    events = []
    # Ensure EMP_ID column exists for event logging
    if EMP_ID not in df.columns:
        df[EMP_ID] = df.index
    # Preserve original salaries for bump calculations
    base_comp = df[EMP_GROSS_COMP].copy()

    # 1) COLA bump
    cola_rate = getattr(params, 'COLA_rate', 0.0)
    if cola_rate:
        for idx, old in base_comp.items():
            bump = old * cola_rate
            df.at[idx, EMP_GROSS_COMP] += bump
            evt = create_event(
                event_time=event_time,
                employee_id=df.at[idx, EMP_ID],
                event_type=EVT_RAISE,
                value_num=bump,
                meta='COLA adjustment'
            )
            events.append(evt)

    # 2) Promotion bump (mask-based, avoids pandas NA ambiguity)
    promo_map = getattr(params, 'promo_raise_pct', {})
    if promo_map and EMP_LEVEL_SOURCE in df.columns:
        promo_sources = ["markov-promo", "rule-promo"]
        mask = df[EMP_LEVEL_SOURCE].isin(promo_sources)
        for idx in df.loc[mask].index:
            lvl_from = int(df.at[idx, EMP_LEVEL])
            key = f"{lvl_from}_to_{lvl_from + 1}"
            pct = promo_map.get(key, 0.0)
            if pct:
                old = base_comp.loc[idx]
                bump = old * pct
                df.at[idx, EMP_GROSS_COMP] += bump
                evt = create_event(
                    event_time=event_time,
                    employee_id=df.at[idx, EMP_ID],
                    event_type=EVT_RAISE,
                    value_num=bump,
                    meta='Promotion bump'
                )
                events.append(evt)

    # 3) Merit distribution (level-specific)
    merit_map = getattr(params, 'merit_dist', {})
    if merit_map:
        for idx, row in df.iterrows():
            lvl = row.get(EMP_LEVEL)
            if lvl not in merit_map:
                continue
            dist = merit_map.get(lvl, {}) or {}
            mu = dist.get('mu', 0.0)
            sigma = dist.get('sigma', 0.0)
            pct = rng.normal(mu, sigma)
            old = df.at[idx, EMP_GROSS_COMP]
            bump = base_comp.loc[idx] * pct
            new_val = old + bump
            df.at[idx, EMP_GROSS_COMP] = new_val
            evt = create_event(
                event_time=event_time,
                employee_id=df.at[idx, EMP_ID],
                event_type=EVT_RAISE,
                value_num=bump,
                meta='Merit distribution'
            )
            events.append(evt)

    # Return DataFrame of raise events
    return pd.DataFrame(events, columns=EVENT_COLS)
