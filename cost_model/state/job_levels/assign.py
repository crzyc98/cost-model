from functools import lru_cache
from typing import Dict, Optional
import pandas as pd
import logging

from cost_model.utils.columns import EMP_LEVEL, EMP_GROSS_COMP
from . import state
from .init import get_level_by_id, init_job_levels
init_job_levels(reset_warnings=True)

logger = logging.getLogger(__name__)

def assign_levels_to_dataframe(df: pd.DataFrame, comp_column: str = EMP_GROSS_COMP) -> pd.DataFrame:
    """Assign job levels to employees in a dataframe using vectorized pd.cut.

    Raises:
        ValueError: If compensation column is missing
    """
    if comp_column not in df.columns:
        raise ValueError(f"Compensation column '{comp_column}' not found in dataframe")
    # ensure job levels initialized
    if state._COMP_INTERVALS is None:
        init_job_levels(reset_warnings=False)
    # determine target level column: use existing 'level_id' or create EMP_LEVEL
    level_col = 'level_id' if 'level_id' in df.columns else EMP_LEVEL

    # assign levels via lookup per compensation
    df[level_col] = df[comp_column].apply(
        lambda c: (get_level_by_compensation(c).level_id if get_level_by_compensation(c) else pd.NA)
    ).astype('Int64')

    # --- NEW: Track source of assignment ---
    df["job_level_source"] = None
    mask = df[level_col].notna()
    df.loc[mask, "job_level_source"] = "salary-band"
    # ---------------------------------------
    return df


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
