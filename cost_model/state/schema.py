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
        EVT_NEW_HIRE_TERM,
    )  # type: ignore
except ImportError:  # pragma: no cover
    EVT_HIRE = "EVT_HIRE"
    EVT_TERM = "EVT_TERM"
    EVT_COMP = "EVT_COMP"
    EVT_COLA = "EVT_COLA"
    EVT_PROMOTION = "EVT_PROMOTION"
    EVT_RAISE = "EVT_RAISE"
    EVT_CONTRIB = "EVT_CONTRIB"
    EVT_NEW_HIRE_TERM = "EVT_NEW_HIRE_TERM"

# -----------------------------------------------------------------------------
# Employee column constants (fully defined here; single source of truth)
# -----------------------------------------------------------------------------
EMP_SSN = "employee_ssn"
EMP_ROLE = "employee_role"
EMP_BIRTH_DATE = "employee_birth_date"
EMP_HIRE_DATE = "employee_hire_date"
EMP_TERM_DATE = "employee_termination_date"
EMP_GROSS_COMP = "employee_gross_compensation"
EMP_TENURE = "employee_tenure"  # Standard column for years of service
EMP_PLAN_YEAR_COMP = "employee_plan_year_compensation"
EMP_CAPPED_COMP = "employee_capped_compensation"
EMP_DEFERRAL_RATE = "employee_deferral_rate"
EMP_CONTR = "employee_contribution"

# Employer contribution columns
EMPLOYER_CORE = "employer_core_contribution"
EMPLOYER_CORE_CONTRIB = "employer_core_contribution"  # Alias for consistency
EMPLOYER_MATCH = "employer_match_contribution"
EMPLOYER_MATCH_CONTRIB = "employer_match_contribution"  # Alias for consistency

# Employee and simulation columns
EMP_ID = "employee_id"
EMP_LEVEL = "employee_level"
EMP_ACTIVE = "active"
EMP_TENURE_BAND = "employee_tenure_band"
EMP_LEVEL_SOURCE = "job_level_source"
EMP_EXITED = "exited"

# Simulation parameters
SIMULATION_YEAR = "simulation_year"
TERM_RATE = "term_rate"
COMP_RAISE_PCT = "comp_raise_pct"
NEW_HIRE_TERM_RATE = "new_hire_term_rate"
COLA_PCT = "cola_pct"
CFG = "cfg"

# Event log columns
EVENT_ID = "event_id"
EVENT_TYPE = "event_type"
EVENT_TIME = "event_time"
VALUE_JSON = "value_json"
META = "meta"

# Flags
IS_ELIGIBLE = "is_eligible"
IS_PARTICIPATING = "is_participating"
ELIGIBILITY_ENTRY_DATE = "eligibility_entry_date"
STATUS_COL = "status"
ACTIVE_STATUS = "Active"
INACTIVE_STATUS = "Inactive"
HOURS_WORKED = "hours_worked"

# Auto Enrollment (AE) columns
AE_OPTED_OUT = "ae_opted_out"
PROACTIVE_ENROLLED = "proactive_enrolled"
AUTO_ENROLLED = "auto_enrolled"

# Auto Increase (AI) columns
AI_ELIGIBLE = "ai_eligible"
AI_APPLIED = "ai_applied"
AI_SKIPPED = "ai_skipped"

# AE/AI event flags
BECAME_ELIGIBLE_DURING_YEAR = "became_eligible_during_year"
WINDOW_CLOSED_DURING_YEAR = "window_closed_during_year"

# Hazard table columns
HazardTable = "hazard_table"
AVG_DEFERRAL_PART = "avg_deferral_rate_participants"
AVG_DEFERRAL_TOTAL = "avg_deferral_rate_total"

# Summary/statistics columns
SUM_HEADCOUNT = "headcount"
SUM_ELIGIBLE = "eligible"
SUM_PARTICIPATING = "participating"
RATE_PARTICIP_ELIG = "participation_rate_eligible"
RATE_PARTICIP_TOTAL = "participation_rate_total"
PCT_EMP_COST_PLAN = "employer_cost_pct_plan_comp"
PCT_EMP_COST_CAP = "employer_cost_pct_capped_comp"

DATE_COLS = [EMP_HIRE_DATE, EMP_TERM_DATE, EMP_BIRTH_DATE]

# Central raw→standard mapping
RAW_TO_STD_COLS = {
    "ssn": EMP_SSN,
    "role": EMP_ROLE,
    "birth_date": EMP_BIRTH_DATE,
    "employee_birth_date": EMP_BIRTH_DATE,
    "hire_date": EMP_HIRE_DATE,
    "employee_hire_date": EMP_HIRE_DATE,
    "termination_date": EMP_TERM_DATE,
    "employee_termination_date": EMP_TERM_DATE,
    "gross_compensation": EMP_GROSS_COMP,
    "plan_year_compensation": EMP_PLAN_YEAR_COMP,
    "capped_compensation": EMP_CAPPED_COMP,
    "employee_deferral_pct": EMP_DEFERRAL_RATE,
    "pre_tax_deferral_percentage": EMP_DEFERRAL_RATE,
    "employee_contribution_amt": EMP_CONTR,
    "pre_tax_contributions": EMP_CONTR,
    "employer_core_contribution_amt": EMPLOYER_CORE,
    "employer_core_contribution": EMPLOYER_CORE,
    "employer_match_contribution_amt": EMPLOYER_MATCH,
    "employer_match_contribution": EMPLOYER_MATCH,
    "eligibility_entry_date": ELIGIBILITY_ENTRY_DATE,
}

def to_nullable_bool(series: pd.Series) -> pd.Series:
    """
    Convert a boolean-like Series into pandas’ nullable BooleanDtype.
    """
    return series.astype("boolean")

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
        EVT_NEW_HIRE_TERM,
    )  # type: ignore
except ImportError:  # pragma: no cover
    EVT_HIRE = "EVT_HIRE"
    EVT_TERM = "EVT_TERM"
    EVT_COMP = "EVT_COMP"
    EVT_COLA = "EVT_COLA"
    EVT_PROMOTION = "EVT_PROMOTION"
    EVT_RAISE = "EVT_RAISE"
    EVT_CONTRIB = "EVT_CONTRIB"
    EVT_NEW_HIRE_TERM = "EVT_NEW_HIRE_TERM"

except ImportError:  # pragma: no cover
    # Define all event types as fallback
    EVT_HIRE = "EVT_HIRE"
    EVT_TERM = "EVT_TERM"
    EVT_COMP = "EVT_COMP"
    EVT_COLA = "EVT_COLA"
    EVT_PROMOTION = "EVT_PROMOTION"
    EVT_RAISE = "EVT_RAISE"
    EVT_CONTRIB = "EVT_CONTRIB"
    EVT_NEW_HIRE_TERM = "EVT_NEW_HIRE_TERM"

    # Define all column constants as fallback
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

EVENT_COLS: List[str] = [
    "event_id",
    "event_time",
    "employee_id",
    "event_type",
    "value_num",
    "value_json",
    "meta",
    EMP_LEVEL_SOURCE,  # Added to propagate job_level_source for new hires
]

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
