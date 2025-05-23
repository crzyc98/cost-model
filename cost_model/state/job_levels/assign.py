from functools import lru_cache
from typing import Dict, Optional, List, Union
import pandas as pd
import logging

from cost_model.state.schema import (
    EMP_LEVEL, 
    EMP_GROSS_COMP, 
    EMP_LEVEL_SOURCE,
    EMP_TENURE,
    EMP_TENURE_BAND
)
from cost_model.state.schema import SNAPSHOT_COLS, SNAPSHOT_DTYPES
from . import state
from .init import get_level_by_id, init_job_levels
from .models import JobLevel

init_job_levels(reset_warnings=True)

logger = logging.getLogger(__name__)

@lru_cache(maxsize=128)
def get_level_by_compensation(compensation: float) -> Optional['JobLevel']:
    """Get job level based on compensation."""
    if compensation < 0:
        raise ValueError("Compensation must be non-negative")
    if state._COMP_INTERVALS is None:
        # ensure intervals built
        init_job_levels(reset_warnings=False)
    idx = state._COMP_INTERVALS.get_indexer([compensation])[0]
    if idx == -1:
        # Snap
        if compensation < state._COMP_INTERVALS[0].left:
            return get_level_by_id(min(state.LEVEL_TAXONOMY.keys()))
        if compensation > state._COMP_INTERVALS[-1].right:
            return get_level_by_id(max(state.LEVEL_TAXONOMY.keys()))
        return None
    return sorted(state.LEVEL_TAXONOMY.values(), key=lambda lv: lv.level_id)[idx]


def get_level_distribution(df: pd.DataFrame) -> pd.Series:
    """Get counts of employees per level."""
    if EMP_LEVEL not in df.columns:
        raise ValueError(f"Level column '{EMP_LEVEL}' not found in dataframe")
    return df[EMP_LEVEL].value_counts().sort_index()


def get_warning_counts() -> Dict[str, int]:
    """Return the warning counts."""
    return state._WARNING_COUNTS.copy()
