"""
Constants and schema definitions for workforce snapshots.
"""

from typing import Any, Dict, List

import pandas as pd

# Import event log constants
try:
    from ..event_log import EVENT_COLS, EVT_COMP, EVT_HIRE, EVT_TERM
except ImportError:
    # Fallbacks if running standalone
    EVT_HIRE, EVT_TERM, EVT_COMP = "EVT_HIRE", "EVT_TERM", "EVT_COMP"
    EVENT_COLS = [
        "event_id",
        "event_time",
        "employee_id",
        "event_type",
        "value_num",
        "value_json",
        "meta",
    ]

# Import column name constants
try:
    from ...utils.columns import (
        EMP_ACTIVE,
        EMP_BIRTH_DATE,
        EMP_DEFERRAL_RATE,
        EMP_EXITED,
        EMP_GROSS_COMP,
        EMP_HIRE_DATE,
        EMP_ID,
        EMP_LEVEL,
        EMP_LEVEL_SOURCE,
        EMP_TENURE,
        EMP_TENURE_BAND,
        EMP_TERM_DATE,
    )
except ImportError:
    # Fallbacks if running standalone
    EMP_ID = "employee_id"
    EMP_HIRE_DATE = "employee_hire_date"
    EMP_BIRTH_DATE = "employee_birth_date"
    EMP_GROSS_COMP = "employee_gross_compensation"
    EMP_TERM_DATE = "employee_termination_date"
    EMP_DEFERRAL_RATE = "employee_deferral_rate"
    EMP_TENURE = "employee_tenure"
    EMP_TENURE_BAND = "employee_tenure_band"
    EMP_ACTIVE = "active"
    EMP_LEVEL = "employee_level"
    EMP_LEVEL_SOURCE = "job_level_source"
    EMP_EXITED = "exited"

# Import additional column constants
from ...utils.columns import SIMULATION_YEAR

# Standard columns for snapshots, in preferred order
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
]

# Data types for snapshot columns
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
    EMP_LEVEL_SOURCE: pd.CategoricalDtype(
        categories=["hire", "promotion", "demotion", "manual"], ordered=True
    ),
    EMP_EXITED: pd.BooleanDtype(),
    SIMULATION_YEAR: pd.Int64Dtype(),  # Make nullable
}

# Default compensation values
DEFAULT_COMPENSATION = 50000.0

# Level-based default compensation
LEVEL_BASED_DEFAULTS = {1: 50000.0, 2: 75000.0, 3: 100000.0, 4: 150000.0}

# Column name mappings for standardization
COLUMN_MAPPING = {
    # Standard mappings from schema.py
    "ssn": EMP_ID,
    "employee_ssn": EMP_ID,
    "birth_date": EMP_BIRTH_DATE,
    "hire_date": EMP_HIRE_DATE,
    "termination_date": EMP_TERM_DATE,
    "gross_compensation": EMP_GROSS_COMP,
    # Additional mappings specific to CSV structure
    "employee_birth_date": EMP_BIRTH_DATE,
    "employee_hire_date": EMP_HIRE_DATE,
    "employee_termination_date": EMP_TERM_DATE,
    "employee_gross_compensation": EMP_GROSS_COMP,
    "employee_deferral_rate": EMP_DEFERRAL_RATE,
}

# Tenure band definitions
TENURE_BANDS = {
    "<1": (0, 1),
    "1-3": (1, 3),
    "3-5": (3, 5),
    "5-10": (5, 10),
    "10-15": (10, 15),
    "15+": (15, float("inf")),
}

# Tenure bins and labels for pd.cut
TENURE_BINS = [0, 1, 3, 5, 10, 15, float("inf")]
TENURE_LABELS = ["<1", "1-3", "3-5", "5-10", "10-15", "15+"]

# Columns to remove from consolidated snapshots
COLUMNS_TO_REMOVE = ["term_rate", "comp_raise_pct", "new_hire_term_rate", "cola_pct", "cfg"]

# Default employee status
DEFAULT_EMPLOYEE_STATUS = {"ACTIVE": "Active", "TERMINATED": "Terminated", "INACTIVE": "Inactive"}
