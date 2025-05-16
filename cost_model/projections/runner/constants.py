"""
Constants and static mappings used throughout the projection runner.
"""

from typing import Dict, List, Any
from cost_model.utils.columns import (
    EMP_ID,
    SIMULATION_YEAR,
    EVENT_TYPE,
    EVT_HIRE,
    EVT_TERM,
    EVT_CONTRIB,
)

# Event priority mapping
EVENT_PRIORITY: Dict[str, int] = {
    EVT_HIRE: 1,
    EVT_TERM: 2,
    EVT_CONTRIB: 3,
}

# Column name constants
SNAPSHOT_COLUMNS: List[str] = [
    EMP_ID,
    SIMULATION_YEAR,
    EVENT_TYPE,
    # Add other required snapshot columns here
]

# Default event types for employee contributions
DEFAULT_EE_CONTRIB_EVENT_TYPES: List[str] = [EVT_CONTRIB]

# Default simulation parameters
DEFAULT_SIM_PARAMS: Dict[str, Any] = {
    "seed": 42,
    "num_years": 5,
}
