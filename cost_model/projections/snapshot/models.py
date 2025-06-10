"""
Data models and schemas for snapshot processing.

Defines Pydantic models and TypedDicts for type safety and validation
throughout the snapshot creation and update process.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Union

import pandas as pd

from .types import (
    ColumnName,
    CompensationAmount,
)
from .types import CompensationExtractionResult as CompensationExtractionResultType
from .types import (
    PerformanceMetrics,
    SimulationYear,
)
from .types import ValidationResult as ValidationResultType


@dataclass
class SnapshotConfig:
    """Configuration for snapshot creation and processing."""

    start_year: SimulationYear
    enable_validation: bool = True
    enable_timing: bool = True
    enable_memory_tracking: bool = True
    log_level: str = "INFO"
    compensation_defaults: Optional[Dict[str, CompensationAmount]] = None
    required_columns: Optional[List[ColumnName]] = None
    column_mappings: Optional[Dict[str, str]] = None

    def __post_init__(self) -> None:
        """Initialize default values after dataclass creation."""
        if self.compensation_defaults is None:
            from .constants import LEVEL_BASED_DEFAULTS

            self.compensation_defaults = LEVEL_BASED_DEFAULTS.copy()

        if self.required_columns is None:
            from .constants import REQUIRED_CENSUS_COLUMNS

            self.required_columns = REQUIRED_CENSUS_COLUMNS.copy()

        if self.column_mappings is None:
            from .constants import CENSUS_COLUMN_MAPPINGS

            self.column_mappings = CENSUS_COLUMN_MAPPINGS.copy()


@dataclass
class ValidationResult:
    """Result of data validation operations."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        """Add an error message and mark validation as invalid."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message without affecting validity."""
        self.warnings.append(message)

    def has_issues(self) -> bool:
        """Check if there are any errors or warnings."""
        return bool(self.errors or self.warnings)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results."""
        return {
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "has_issues": self.has_issues(),
        }


@dataclass
class CompensationExtractionResult:
    """Result of compensation extraction for an employee."""

    compensation: Optional[CompensationAmount]
    source: str
    confidence: str  # 'high', 'medium', 'low'
    details: Dict[str, Any] = field(default_factory=dict)
    extraction_timestamp: datetime = field(default_factory=datetime.now)

    def is_successful(self) -> bool:
        """Check if compensation was successfully extracted."""
        return self.compensation is not None

    def get_confidence_score(self) -> float:
        """Get numeric confidence score (0.0 to 1.0)."""
        confidence_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
        return confidence_map.get(self.confidence, 0.0)


@dataclass
class SnapshotMetrics:
    """Comprehensive metrics about snapshot creation and processing."""

    total_employees: int
    active_employees: int
    terminated_employees: int
    missing_data_count: Dict[str, int]
    processing_time_seconds: float
    memory_usage_mb: Optional[float] = None
    operation_name: str = "unknown"
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    checkpoints: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set end time if not provided."""
        if self.end_time is None:
            self.end_time = datetime.now()

    def get_employee_distribution(self) -> Dict[str, float]:
        """Get percentage distribution of employee types."""
        if self.total_employees == 0:
            return {"active": 0.0, "terminated": 0.0}

        return {
            "active": (self.active_employees / self.total_employees) * 100,
            "terminated": (self.terminated_employees / self.total_employees) * 100,
        }

    def add_checkpoint(self, name: str, **metrics: Any) -> None:
        """Add a performance checkpoint."""
        self.checkpoints[name] = {"timestamp": datetime.now(), **metrics}

    def get_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of metrics."""
        return {
            "operation": self.operation_name,
            "duration_seconds": self.processing_time_seconds,
            "total_employees": self.total_employees,
            "employee_distribution": self.get_employee_distribution(),
            "memory_usage_mb": self.memory_usage_mb,
            "checkpoint_count": len(self.checkpoints),
            "missing_data_fields": len(self.missing_data_count),
        }
