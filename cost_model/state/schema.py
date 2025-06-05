# cost_model/state/schema.py
# flake8: noqa
"""Centralized schema constants for cost_model state and events.

This module defines:
  - Event type constants (from event_log)
  - Employee column constants (imported from utils.columns)
  - Snapshot column ordering and pandas dtypes
  - Summary/Reporting column constants for consistent reporting

All other modules should import from here for consistency.

QuickStart: see docs/cost_model/state/schema.md
"""
from __future__ import annotations

import pandas as pd
from typing import List

# Import age & tenure-related constants
from .tenure import TENURE_BAND_CATEGORICAL_DTYPE
from .age import AGE_BAND_CATEGORICAL_DTYPE

# Core identifier and simulation constants
EMP_ID = "employee_id"
SIMULATION_YEAR = "simulation_year"

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
EMP_BIRTH_DATE = "employee_birth_date"
EMP_HIRE_DATE = "employee_hire_date"
EMP_TERM_DATE = "employee_termination_date"
EMP_GROSS_COMP = "employee_gross_compensation"
EMP_TENURE = "employee_tenure"  # Standard column for years of service
EMP_AGE = "employee_age"  # Standard column for employee age
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
EMP_AGE_BAND = "employee_age_band"
EMP_LEVEL_SOURCE = "job_level_source"
EMP_EXITED = "exited"

# Yearly snapshot specific columns for enhanced tracking
EMP_STATUS_EOY = "employee_status_eoy"  # Status at end of year: 'Active', 'Terminated', etc.

# Simulation parameters
SIMULATION_YEAR = "simulation_year"
TERM_RATE = "term_rate"
COMP_RAISE_PCT = "comp_raise_pct"
NEW_HIRE_TERM_RATE = "new_hire_term_rate"
COLA_PCT = "cola_pct"
CFG = "cfg"

# Additional hazard table columns for dynamic generation
PROMOTION_RATE = "promotion_rate"
MERIT_RAISE_PCT = "merit_raise_pct"
PROMOTION_RAISE_PCT = "promotion_raise_pct"
NEW_HIRE_TERMINATION_RATE = "new_hire_termination_rate"
TENURE_BAND = "tenure_band"  # Used in hazard table (vs EMP_TENURE_BAND in snapshots)

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
ENROLLMENT_DATE = "enrollment_date"
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
EMP_AGE_BAND = "employee_age_band"
COMP_RAISE_PCT = "comp_raise_pct"
NEW_HIRE_TERM_RATE = "new_hire_term_rate"
COLA_PCT = "cola_pct"
CFG = "cfg"

# -----------------------------------------------------------------------------
# Summary/Reporting column constants
# -----------------------------------------------------------------------------
# These canonical column names should be used in all summary statistics, reporting,
# and plotting functions to ensure consistency across the codebase.

SUMMARY_YEAR = "year"
SUMMARY_ACTIVE_HEADCOUNT = "active_headcount"
SUMMARY_TERMINATIONS = "terminations"
SUMMARY_TOTAL_CONTRIBUTIONS = "total_contributions"
SUMMARY_TOTAL_ER_CONTRIBUTIONS = "total_er_contributions"
SUMMARY_TOTAL_EE_CONTRIBUTIONS = "total_ee_contributions"
SUMMARY_TOTAL_BENEFITS = "total_benefits"
SUMMARY_AVG_COMP = "avg_compensation"
SUMMARY_AVG_TENURE = "avg_tenure"
SUMMARY_AVG_AGE = "avg_age"
SUMMARY_NEW_HIRES = "new_hires"
SUMMARY_NEW_HIRE_TERMINATIONS = "new_hire_terminations"

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
    EMP_AGE,
    EMP_AGE_BAND,
    EMP_LEVEL,
    EMP_LEVEL_SOURCE,
    EMP_EXITED,
    EMP_STATUS_EOY,  # Employee status at end of year for yearly snapshots
    SIMULATION_YEAR,  # Ensure simulation_year is included in all snapshots
    # Contribution columns required for summary metrics
    EMP_CONTR,  # Employee contribution amount
    EMPLOYER_CORE,  # Employer core contribution amount
    EMPLOYER_MATCH,  # Employer match contribution amount
    # Eligibility and participation columns
    IS_ELIGIBLE,  # Employee eligibility status
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
    EMP_TENURE_BAND: TENURE_BAND_CATEGORICAL_DTYPE,
    EMP_AGE: "float64",
    EMP_AGE_BAND: AGE_BAND_CATEGORICAL_DTYPE,
    EMP_LEVEL: pd.Int64Dtype(),
    EMP_LEVEL_SOURCE: pd.CategoricalDtype(
        categories=["hire", "promotion", "demotion", "manual"],
        ordered=True,
    ),
    EMP_EXITED: pd.BooleanDtype(),
    EMP_STATUS_EOY: pd.CategoricalDtype(
        categories=["Active", "Terminated", "Inactive"],
        ordered=False,
    ),
    SIMULATION_YEAR: 'int64',  # Use standard int64 for simulation year
    # Contribution column types
    EMP_CONTR: pd.Float64Dtype(),  # Employee contribution amount
    EMPLOYER_CORE: pd.Float64Dtype(),  # Employer core contribution amount
    EMPLOYER_MATCH: pd.Float64Dtype(),  # Employer match contribution amount
    # Eligibility and participation column types
    IS_ELIGIBLE: pd.BooleanDtype(),  # Employee eligibility status
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
    "EMP_AGE",
    "EMP_AGE_BAND",
    "EMP_LEVEL",
    "EMP_LEVEL_SOURCE",
    "EMP_ACTIVE",
    "EMP_EXITED",
    "EMP_STATUS_EOY",
    "SIMULATION_YEAR",
    # contribution columns
    "EMP_CONTR",
    "EMPLOYER_CORE",
    "EMPLOYER_MATCH",
    "EMPLOYER_CORE_CONTRIB",
    "EMPLOYER_MATCH_CONTRIB",
    # eligibility columns
    "IS_ELIGIBLE",
    "IS_PARTICIPATING",
    "ELIGIBILITY_ENTRY_DATE",
    "STATUS_COL",
    "HOURS_WORKED",
    "AE_OPTED_OUT",
    "PROACTIVE_ENROLLED",
    "AUTO_ENROLLED",
    "ENROLLMENT_DATE",
    # config defaults
    "TERM_RATE",
    "COMP_RAISE_PCT",
    "NEW_HIRE_TERM_RATE",
    "COLA_PCT",
    "CFG",
    # hazard table columns
    "PROMOTION_RATE",
    "MERIT_RAISE_PCT",
    "PROMOTION_RAISE_PCT",
    "NEW_HIRE_TERMINATION_RATE",
    "TENURE_BAND",
    # snapshot
    "SNAPSHOT_COLS",
    "SNAPSHOT_DTYPES",
    # tenure
    "TENURE_BAND_CATEGORICAL_DTYPE",
    # age
    "AGE_BAND_CATEGORICAL_DTYPE",
    # summary/reporting columns
    "SUMMARY_YEAR",
    "SUMMARY_ACTIVE_HEADCOUNT",
    "SUMMARY_TERMINATIONS",
    "SUMMARY_TOTAL_CONTRIBUTIONS",
    "SUMMARY_TOTAL_ER_CONTRIBUTIONS",
    "SUMMARY_TOTAL_EE_CONTRIBUTIONS",
    "SUMMARY_TOTAL_BENEFITS",
    "SUMMARY_AVG_COMP",
    "SUMMARY_AVG_TENURE",
    "SUMMARY_NEW_HIRES",
    "SUMMARY_NEW_HIRE_TERMINATIONS",
]
