# cost_model/state/schema.py
# flake8: noqa
"""Centralized schema constants for cost_model state and events.

This module defines:
  - Event type constants (from event_log)
  - Employee column constants (imported from utils.columns)
  - Snapshot column ordering and pandas dtypes

All other modules should import from here for consistency.

QuickStart: see docs/cost_model/state/schema.md
"""
from __future__ import annotations

import pandas as pd
from typing import List

# -----------------------------------------------------------------------------
# Event type constants (re-export from event_log so there is a single source)
# -----------------------------------------------------------------------------
try:
    from .event_log import (
        EVT_HIRE,
        EVT_TERM,
        EVT_COMP,
        EVT_COLA,
        EVT_PROMOTION,
        EVT_RAISE,
        EVT_CONTRIB,
    )  # type: ignore
except ImportError:  # pragma: no cover
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
        EMP_LEVEL_SOURCE,
        EMP_ACTIVE,
        EMP_EXITED,
        SIMULATION_YEAR,
        TERM_RATE,
    )
except ImportError:  # pragma: no cover
    EMP_ID = "employee_id"
    EMP_HIRE_DATE = "employee_hire_date"
    EMP_BIRTH_DATE = "employee_birth_date"
    EMP_GROSS_COMP = "employee_gross_compensation"
    EMP_TERM_DATE = "employee_termination_date"
    EMP_DEFERRAL_RATE = "employee_deferral_rate"
    EMP_TENURE = "employee_tenure"
    EMP_LEVEL = "employee_level"
    EMP_LEVEL_SOURCE = "job_level_source"
    EMP_ACTIVE = "active"
    EMP_EXITED = "exited"
    SIMULATION_YEAR = "simulation_year"
    TERM_RATE = "term_rate"

# Configuration columns for snapshot defaults
EMP_TENURE_BAND = "employee_tenure_band"
COMP_RAISE_PCT = "comp_raise_pct"
NEW_HIRE_TERM_RATE = "new_hire_term_rate"
COLA_PCT = "cola_pct"
CFG = "cfg"
COMP_RAISE_PCT = "comp_raise_pct"
NEW_HIRE_TERM_RATE = "new_hire_term_rate"
COLA_PCT = "cola_pct"
CFG = "cfg"

# -----------------------------------------------------------------------------
# Snapshot schema definition
# -----------------------------------------------------------------------------
# Only the columns actually produced by the snapshot builder are required here.
SNAPSHOT_COLS: List[str] = [
    EMP_ID,
    EMP_HIRE_DATE,
    EMP_BIRTH_DATE,
    EMP_GROSS_COMP,
    EMP_TERM_DATE,
    EMP_ACTIVE,
    EMP_DEFERRAL_RATE,
    EMP_TENURE,
    EMP_TENURE_BAND,
    EMP_LEVEL,
    EMP_LEVEL_SOURCE,
    EMP_EXITED,
    SIMULATION_YEAR,
]

SNAPSHOT_DTYPES: dict[str, object] = {
    EMP_ID: pd.StringDtype(),
    EMP_HIRE_DATE: "datetime64[ns]",
    EMP_BIRTH_DATE: "datetime64[ns]",
    EMP_GROSS_COMP: pd.Float64Dtype(),
    EMP_TERM_DATE: "datetime64[ns]",
    EMP_ACTIVE: pd.BooleanDtype(),
    EMP_DEFERRAL_RATE: pd.Float64Dtype(),
    EMP_TENURE: "float64",
    EMP_TENURE_BAND: pd.StringDtype(),
    EMP_LEVEL: pd.Int64Dtype(),
    EMP_LEVEL_SOURCE: pd.CategoricalDtype(
        categories=["hire", "promotion", "demotion", "manual"],
        ordered=True,
    ),
    EMP_EXITED: pd.BooleanDtype(),
    SIMULATION_YEAR: pd.Int64Dtype(),
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
    "EMP_TENURE_BAND",
    "EMP_LEVEL",
    "EMP_LEVEL_SOURCE",
    "EMP_ACTIVE",
    "EMP_EXITED",
    "SIMULATION_YEAR",
    # config defaults
    "TERM_RATE",
    "COMP_RAISE_PCT",
    "NEW_HIRE_TERM_RATE",
    "COLA_PCT",
    "CFG",
    # snapshot
    "SNAPSHOT_COLS",
    "SNAPSHOT_DTYPES",
]
