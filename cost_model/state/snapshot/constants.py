"""
Constants and schema definitions for workforce snapshots.
"""

import pandas as pd
from typing import Dict, List, Any

# Import event log constants
try:
    from ..event_log import EVT_HIRE, EVT_TERM, EVT_COMP, EVENT_COLS
except ImportError:
    # Fallbacks if running standalone
    EVT_HIRE, EVT_TERM, EVT_COMP = "EVT_HIRE", "EVT_TERM", "EVT_COMP"
    EVENT_COLS = [
        "event_id", "event_time", "employee_id", 
        "event_type", "value_num", "value_json", "meta"
    ]

# Import column name constants
try:
    from ...utils.columns import (
        EMP_ID, EMP_HIRE_DATE, EMP_BIRTH_DATE, EMP_ROLE,
        EMP_GROSS_COMP, EMP_TERM_DATE, EMP_DEFERRAL_RATE,
        EMP_TENURE, EMP_TENURE_BAND, EMP_ACTIVE,
        EMP_LEVEL, EMP_LEVEL_SOURCE, EMP_EXITED
    )
except ImportError:
    # Fallbacks if running standalone
    EMP_ID = "employee_id"
    EMP_HIRE_DATE = "employee_hire_date"
    EMP_BIRTH_DATE = "employee_birth_date"
    EMP_ROLE = "employee_role"
    EMP_GROSS_COMP = "employee_gross_compensation"
    EMP_TERM_DATE = "employee_termination_date"
    EMP_DEFERRAL_RATE = "employee_deferral_rate"
    EMP_TENURE = "employee_tenure"
    EMP_TENURE_BAND = "tenure_band"
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
    EMP_ROLE,
    EMP_GROSS_COMP,
    EMP_TERM_DATE,
    EMP_ACTIVE,
    EMP_DEFERRAL_RATE,
    EMP_TENURE_BAND,
    EMP_TENURE,
    EMP_LEVEL,
    EMP_LEVEL_SOURCE,
    EMP_EXITED,
    SIMULATION_YEAR
]

# Data types for snapshot columns
SNAPSHOT_DTYPES = {
    EMP_ID: pd.StringDtype(),
    EMP_HIRE_DATE: "datetime64[ns]",
    EMP_BIRTH_DATE: "datetime64[ns]",
    EMP_ROLE: pd.StringDtype(),
    EMP_GROSS_COMP: pd.Float64Dtype(),
    EMP_TERM_DATE: "datetime64[ns]",
    EMP_ACTIVE: pd.BooleanDtype(),
    EMP_DEFERRAL_RATE: pd.Float64Dtype(),
    EMP_TENURE_BAND: pd.StringDtype(),
    EMP_TENURE: "float64",
    EMP_LEVEL: pd.Int64Dtype(),
    EMP_LEVEL_SOURCE: pd.CategoricalDtype(categories=['hire', 'promotion', 'demotion', 'manual'], ordered=True),
    EMP_EXITED: pd.BooleanDtype(),
    SIMULATION_YEAR: pd.Int64Dtype()  # Make nullable
}
