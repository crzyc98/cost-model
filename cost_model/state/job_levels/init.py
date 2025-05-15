import os
import logging
from pathlib import Path
from typing import Optional, Dict

from .defaults import DEFAULT_LEVELS
from .loader import load_from_yaml
from .intervals import build_intervals
from .models import ConfigError
from . import state

logger = logging.getLogger(__name__)

def init_job_levels(
    config_path: Optional[str] = None,
    strict_validation: bool = True,
    reset_warnings: bool = False
) -> bool:
    """
    Initialize global job levels state.
    """
    path = config_path or os.getenv("COST_MODEL_JOB_LEVELS_CONFIG")
    levels = None
    if path:
        p = Path(path)
        if p.exists():
            try:
                levels = load_from_yaml(path, strict_validation=strict_validation)
            except ConfigError as e:
                if strict_validation:
                    raise
                logger.warning("Invalid job levels config at %s, falling back to defaults: %s", path, e)
        else:
            logger.info("Job levels config path %s does not exist, using defaults", path)
    if levels is None:
        levels = DEFAULT_LEVELS
    # Update global taxonomy
    state.LEVEL_TAXONOMY.clear()
    state.LEVEL_TAXONOMY.update(levels)
    # Build and set compensation intervals
    state._COMP_INTERVALS = build_intervals(state.LEVEL_TAXONOMY)
    # Reset warning counts if requested
    if reset_warnings:
        for key in state._WARNING_COUNTS:
            state._WARNING_COUNTS[key] = 0
    return True


def refresh_job_levels() -> bool:
    """
    Refresh job levels by re-initializing using existing env/config.
    """
    return init_job_levels(reset_warnings=True)


def get_level_by_id(level_id: int):
    """
    Get JobLevel by id.
    """
    try:
        return state.LEVEL_TAXONOMY[level_id]
    except KeyError:
        raise KeyError(f"Job level {level_id} not found")


def get_warning_counts() -> Dict[str, int]:
    """
    Return the warning counts.
    """
    return state._WARNING_COUNTS.copy()
