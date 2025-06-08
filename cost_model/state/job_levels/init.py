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
    config_dict: Optional[dict] = None,
    strict_validation: bool = True,
    reset_warnings: bool = False
) -> bool:
    """
    Initialize global job levels state.

    Args:
        config_path: Path to separate job levels YAML file
        config_dict: Main config dictionary containing job_levels section
        strict_validation: Whether to raise errors on validation failures
        reset_warnings: Whether to reset warning counts
    """
    levels = None

    # Priority 1: Use config_dict if provided (main config integration)
    if config_dict is not None:
        try:
            from .loader import load_job_levels_from_config
            levels = load_job_levels_from_config(config_dict, strict_validation=strict_validation)
            logger.info("Loaded job levels from main config dictionary")
        except ConfigError as e:
            if strict_validation:
                raise
            logger.warning("Invalid job levels in config dict, falling back to defaults: %s", e)

    # Priority 2: Use config_path if provided
    if levels is None:
        path = config_path or os.getenv("COST_MODEL_JOB_LEVELS_CONFIG")
        if path:
            p = Path(path)
            if p.exists():
                try:
                    levels = load_from_yaml(path, strict_validation=strict_validation)
                    logger.info("Loaded job levels from file: %s", path)
                except ConfigError as e:
                    if strict_validation:
                        raise
                    logger.warning("Invalid job levels config at %s, falling back to defaults: %s", path, e)
            else:
                logger.info("Job levels config path %s does not exist, using defaults", path)

    # Priority 3: Use defaults
    if levels is None:
        levels = DEFAULT_LEVELS
        logger.info("Using default job levels")

    # Update global taxonomy
    state.LEVEL_TAXONOMY.clear()
    state.LEVEL_TAXONOMY.update(levels)
    # Build and set compensation intervals
    state._COMP_INTERVALS = build_intervals(state.LEVEL_TAXONOMY)
    # Reset warning counts if requested
    if reset_warnings:
        for key in state._WARNING_COUNTS:
            state._WARNING_COUNTS[key] = 0

    logger.info("Initialized job levels: %s", list(levels.keys()))
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
