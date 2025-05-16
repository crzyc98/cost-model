# cost_model/state/schema.py
"""Centralized schema constants for cost_model state and events.

This module defines event type constants, employee column constants,
 snapshot column ordering, and pandas dtypes. All other modules
 should import from here instead of redefining their own versions.

QuickStart: see docs/cost_model/state/schema.md
"""
from __future__ import annotations

import pandas as pd
from typing import List

# -----------------------------------------------------------------------------
# Event type constants (re-export from event_log so there is a single source)
# -----------------------------------------------------------------------------
try:
    from .event_log import EVT_HIRE, EVT_TERM, EVT_COMP, EVT_COLA, EVT_PROMOTION, EVT_RAISE, EVT_CONTRIB  # type: ignore
except ImportError:  # pragma: no cover  – stand-alone import safety
    EVT_HIRE = "EVT_HIRE"
    EVT_TERM = "EVT_TERM"
    EVT_COMP = "EVT_COMP"
    EVT_COLA = "EVT_COLA"
    EVT_PROMOTION = "EVT_PROMOTION"
    EVT_RAISE = "EVT_RAISE"
    EVT_CONTRIB = "EVT_CONTRIB"

EVENT_COLS: List[str] = [
    "event_id",
    "event_time",
    "employee_id",
    "event_type",
    "value_num",
    "value_json",
    "meta",
]

# -----------------------------------------------------------------------------
# Employee column constants (import from utils.columns where possible)
# -----------------------------------------------------------------------------
try:
    from ..utils.columns import (
        EMP_ID,
        EMP_HIRE_DATE,
        EMP_BIRTH_DATE,
        EMP_GROSS_COMP,
        EMP_TERM_DATE,
        EMP_DEFERRAL_RATE,
        EMP_TENURE,
        EMP_LEVEL,
        EMP_ACTIVE,
        EMP_TENURE_BAND,
        EMP_LEVEL_SOURCE,
        EMP_EXITED,
        SIMULATION_YEAR,
        TERM_RATE,
        COMP_RAISE_PCT,
        NEW_HIRE_TERM_RATE,
        COLA_PCT,
        CFG,
    )
except ImportError:  # pragma: no cover  – stand-alone import safety
    EMP_ID = "employee_id"
    EMP_HIRE_DATE = "hire_date"
    EMP_BIRTH_DATE = "birth_date"
    EMP_GROSS_COMP = "current_comp"
    EMP_TERM_DATE = "term_date"
    EMP_DEFERRAL_RATE = "employee_deferral_rate"
    EMP_TENURE = "tenure_years"
    EMP_ACTIVE = "active"
    EMP_TENURE_BAND = "tenure_band"
    EMP_LEVEL = "employee_level"
    EMP_LEVEL_SOURCE = "job_level_source"
    EMP_EXITED = "exited"
    SIMULATION_YEAR = "simulation_year"
    TERM_RATE = "term_rate"
    COMP_RAISE_PCT = "comp_raise_pct"
    NEW_HIRE_TERM_RATE = "new_hire_term_rate"
    COLA_PCT = "cola_pct"
    CFG = "cfg"

# -----------------------------------------------------------------------------
# Snapshot schema definition
# -----------------------------------------------------------------------------
SNAPSHOT_COLS = [
    EMP_ID,
    EMP_HIRE_DATE,
    EMP_BIRTH_DATE,
    EMP_GROSS_COMP,
    EMP_TERM_DATE,
    EMP_ACTIVE,
    EMP_DEFERRAL_RATE,
    EMP_TENURE_BAND,
    EMP_TENURE,
    EMP_LEVEL,
    EMP_LEVEL_SOURCE,
    EMP_EXITED,
    SIMULATION_YEAR,
    TERM_RATE,
    COMP_RAISE_PCT,
    NEW_HIRE_TERM_RATE,
    COLA_PCT,
    CFG
]

SNAPSHOT_DTYPES = {
    EMP_ID: pd.StringDtype(),
    EMP_HIRE_DATE: "datetime64[ns]",
    EMP_BIRTH_DATE: "datetime64[ns]",
    EMP_GROSS_COMP: pd.Float64Dtype(),
    EMP_TERM_DATE: "datetime64[ns]",
    EMP_ACTIVE: pd.BooleanDtype(),
    EMP_DEFERRAL_RATE: pd.Float64Dtype(),
    EMP_TENURE_BAND: pd.StringDtype(),
    EMP_TENURE: "float64",
    EMP_LEVEL: pd.Int64Dtype(),
    EMP_LEVEL_SOURCE: pd.CategoricalDtype(categories=['hire', 'promotion', 'demotion', 'manual'], ordered=True),
    EMP_EXITED: pd.BooleanDtype(),
    TERM_RATE: pd.Float64Dtype(),
    COMP_RAISE_PCT: pd.Float64Dtype(),
    NEW_HIRE_TERM_RATE: pd.Float64Dtype(),
    COLA_PCT: pd.Float64Dtype(),
    CFG: pd.StringDtype()
}

__all__ = [
    # events
    "EVT_HIRE",
    "EVT_TERM",
    "EVT_COMP",
    "EVT_COLA",
    "EVT_PROMOTION",
    "EVT_RAISE",
    "EVT_CONTRIB",
    "EVENT_COLS",
    # employee cols
    "EMP_ID",
    "EMP_HIRE_DATE",
    "EMP_BIRTH_DATE",
    "EMP_GROSS_COMP",
    "EMP_TERM_DATE",
    "EMP_DEFERRAL_RATE",
    "EMP_TENURE",
    "EMP_ACTIVE",
    "EMP_TENURE_BAND",
    "EMP_LEVEL",
    "EMP_LEVEL_SOURCE",
    "EMP_EXITED",
    "SIMULATION_YEAR",
    "TERM_RATE",
    "COMP_RAISE_PCT",
    "NEW_HIRE_TERM_RATE",
    "COLA_PCT",
    "CFG",
    # snapshot
    "SNAPSHOT_COLS",
    "SNAPSHOT_DTYPES",
]
