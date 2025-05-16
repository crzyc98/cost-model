from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any, Union
import pandas as pd

from cost_model.utils.columns import (
    EMP_LEVEL,
    EMP_GROSS_COMP,
    EMP_LEVEL_SOURCE,
    EMP_TENURE,
    EMP_TENURE_BAND
)
from cost_model.state.schema import SNAPSHOT_COLS, SNAPSHOT_DTYPES


class ConfigError(Exception):
    """Exception raised for errors in the job level configuration."""
    pass


@dataclass(frozen=True)
class JobLevel:
    """Represents a job level in the organization hierarchy.

    Args:
        level_id: Unique identifier for the level
        name: Human-readable name
        description: Description of the level's responsibilities
        min_compensation: Minimum compensation for this level
        max_compensation: Maximum compensation for this level
        comp_base_salary: Base salary for new hires at this level
        comp_age_factor: Age adjustment factor for compensation
        comp_stochastic_std_dev: Standard deviation for random compensation variation
        mid_compensation: Optional midpoint (auto-calculated if None)
        avg_annual_merit_increase: Average annual merit increase percentage
        promotion_probability: Annual probability of promotion
        target_bonus_percent: Target bonus percentage
        source: Method of assignment (band, title, ml)
        job_families: List of job families associated with this level
    """
    level_id: int
    name: str
    description: str
    min_compensation: float
    max_compensation: float
    comp_base_salary: float
    comp_age_factor: float
    comp_stochastic_std_dev: float
    mid_compensation: Optional[float] = None
    avg_annual_merit_increase: float = 0.03
    promotion_probability: float = 0.10
    target_bonus_percent: float = 0.0
    source: Literal["band", "title", "ml"] = "band"
    job_families: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Calculate midpoint if not provided and validate compensation bands."""
        if self.mid_compensation is None:
            object.__setattr__(self, "mid_compensation", (self.min_compensation + self.max_compensation) / 2)
        # Assert no overlap in bands (run once at startup)
        assert self.min_compensation <= self.max_compensation, (
            f"Level {self.level_id} ({self.name}): min_compensation > max_compensation!"
        )

    def is_in_range(self, compensation: float) -> bool:
        """Check if a compensation value is within this level's range."""
        return self.min_compensation <= compensation <= self.max_compensation

    def calculate_compa_ratio(self, compensation: float) -> float:
        """Calculate compa-ratio (compensation / midpoint)."""
        if self.mid_compensation == 0:
            return 1.0
        return compensation / self.mid_compensation
