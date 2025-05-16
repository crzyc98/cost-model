"""
Snapshot module for maintaining workforce state.

QuickStart: see docs/cost_model/state/snapshot.md
"""

import pandas as pd
import warnings
from ..schema import (
    EMP_ID, EMP_HIRE_DATE, EMP_GROSS_COMP, EMP_ACTIVE,
    EMP_BIRTH_DATE, EMP_TENURE, EMP_TENURE_BAND, EMP_LEVEL, EMP_LEVEL_SOURCE,
    EMP_EXITED
)
from .snapshot_build import build_full as _build_full
from .snapshot_update import update as _update
from .helpers import get_first_event as _get_first_event, get_last_event as _get_last_event, ensure_columns_and_types as _ensure_columns_and_types
from .tenure import assign_tenure_band as _assign_tenure_band
from .details import extract_hire_details as _extract_hire_details

def build_full(events: pd.DataFrame, snapshot_year: int) -> pd.DataFrame:
    """Compatibility wrapper for build_full. (Deprecated)"""
    warnings.warn(
        "build_full is deprecated. Use cost_model.state.snapshot_build.build_full instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _build_full(events, snapshot_year)

def update(prev_snapshot: pd.DataFrame, new_events: pd.DataFrame, snapshot_year: int) -> pd.DataFrame:
    """Compatibility wrapper for update. (Deprecated)"""
    warnings.warn(
        "update is deprecated. Use cost_model.state.snapshot_update.update instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _update(prev_snapshot, new_events, snapshot_year)

def get_first_event(events: pd.DataFrame, event_type: str) -> pd.DataFrame:
    """Compatibility wrapper for get_first_event. (Deprecated)"""
    warnings.warn(
        "get_first_event is deprecated. Use cost_model.state.snapshot.helpers.get_first_event instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _get_first_event(events, event_type)

def get_last_event(events: pd.DataFrame, event_type: str) -> pd.DataFrame:
    """Compatibility wrapper for get_last_event. (Deprecated)"""
    warnings.warn(
        "get_last_event is deprecated. Use cost_model.state.snapshot.helpers.get_last_event instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _get_last_event(events, event_type)

def ensure_columns_and_types(df: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper for ensure_columns_and_types. (Deprecated)"""
    warnings.warn(
        "ensure_columns_and_types is deprecated. Use cost_model.state.snapshot.helpers.ensure_columns_and_types instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _ensure_columns_and_types(df)

def assign_tenure_band(df: pd.DataFrame, snapshot_year: int) -> pd.DataFrame:
    """Compatibility wrapper for assign_tenure_band. (Deprecated)"""
    warnings.warn(
        "assign_tenure_band is deprecated. Use cost_model.state.snapshot.tenure.assign_tenure_band instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _assign_tenure_band(df, snapshot_year)

def extract_hire_details(df: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper for extract_hire_details. (Deprecated)"""
    warnings.warn(
        "extract_hire_details is deprecated. Use cost_model.state.snapshot.details.extract_hire_details instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _extract_hire_details(df)

__all__ = ["build_full", "update", "get_first_event", "get_last_event", "ensure_columns_and_types", "assign_tenure_band", "extract_hire_details"]
