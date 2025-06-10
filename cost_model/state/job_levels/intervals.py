from typing import Dict, List, Tuple

import pandas as pd

from .models import ConfigError, JobLevel

__all__ = ["build_intervals", "check_for_overlapping_bands"]


def build_intervals(levels: Dict[int, JobLevel]) -> pd.IntervalIndex:
    """Build interval index for vectorized level assignment.

    Args:
        levels: Dictionary of job levels

    Returns:
        IntervalIndex for level assignment

    Raises:
        ConfigError: If levels dict is empty
    """
    if not levels:
        raise ConfigError("Cannot build intervals from empty levels dictionary")
    sorted_levels = sorted(levels.values(), key=lambda lv: lv.min_compensation)
    return pd.IntervalIndex.from_tuples(
        [(lv.min_compensation, lv.max_compensation) for lv in sorted_levels], closed="both"
    )


def check_for_overlapping_bands(
    levels: Dict[int, JobLevel], strict: bool = True
) -> List[Tuple[JobLevel, JobLevel]]:
    """Check for overlapping compensation bands between levels.

    Args:
        levels: Dictionary of job levels
        strict: If True, raise error on overlaps; if False, return overlaps list

    Returns:
        List of tuples of overlapping levels

    Raises:
        ConfigError: If strict and overlaps found
    """
    sorted_levels = sorted(levels.values(), key=lambda lv: lv.min_compensation)
    overlaps: List[Tuple[JobLevel, JobLevel]] = []
    for current, next_level in zip(sorted_levels, sorted_levels[1:]):
        if current.max_compensation >= next_level.min_compensation:
            overlaps.append((current, next_level))
            if strict:
                raise ConfigError(
                    f"Overlapping compensation bands: "
                    f"Level {current.level_id} ({current.name}) max {current.max_compensation} "
                    f"overlaps with Level {next_level.level_id} ({next_level.name}) min {next_level.min_compensation}"
                )
    return overlaps
