"""
DEPRECATED: use the state/job_levels package instead.
This module re-exports the public API for backward compatibility.

QuickStart: see docs/cost_model/state/job_levels.md
"""

import os
from pathlib import Path

__path__ = [os.path.join(os.path.dirname(__file__), "job_levels")]

import cost_model.state.job_levels.state as _state
from cost_model.state.job_levels.assign import (
    assign_levels_to_dataframe,
    get_level_by_compensation,
    get_level_distribution,
)
from cost_model.state.job_levels.init import get_warning_counts, init_job_levels, refresh_job_levels
from cost_model.state.job_levels.loader import load_from_yaml, load_job_levels_from_config
from cost_model.state.job_levels.models import ConfigError, JobLevel
from cost_model.state.job_levels.sampling import (
    sample_mixed_new_hires,
    sample_new_hire_compensation,
    sample_new_hires_vectorized,
)

# Initialize module-level state
init_job_levels(reset_warnings=True)

__all__ = [
    "init_job_levels",
    "refresh_job_levels",
    "get_level_by_compensation",
    "assign_levels_to_dataframe",
    "get_level_distribution",
    "get_warning_counts",
    "sample_new_hire_compensation",
    "sample_new_hires_vectorized",
    "sample_mixed_new_hires",
    "JobLevel",
    "ConfigError",
    "load_from_yaml",
    "load_job_levels_from_config",
    "LEVEL_TAXONOMY",
    "_COMP_INTERVALS",
    "_WARNING_COUNTS",
    "MAX_WARNINGS",
]


def __getattr__(name):
    """
    Dynamically expose state variables from the underlying state module.
    """
    if name in ("LEVEL_TAXONOMY", "_COMP_INTERVALS", "_WARNING_COUNTS", "MAX_WARNINGS"):
        return getattr(_state, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
