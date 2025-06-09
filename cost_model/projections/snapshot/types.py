"""
Type definitions for snapshot processing.

Provides comprehensive type definitions, protocols, and type aliases
for the snapshot refactoring system to ensure type safety and clarity.
"""

from typing import TypedDict, Protocol, Union, Dict, List, Optional, Any, Callable, Set
from typing_extensions import NotRequired
import pandas as pd
from datetime import datetime
from pathlib import Path

# Type aliases for common types
FilePath = Union[str, Path]
EmployeeId = str
SimulationYear = int
CompensationAmount = float
TenureYears = float
AgeYears = float

# Column name type aliases
ColumnName = str
ColumnMapping = Dict[str, str]

# Event type definitions
class EmployeeEvent(TypedDict):
    """Structure for employee events in the system."""
    employee_id: EmployeeId
    event_type: str
    event_date: datetime
    compensation: NotRequired[Optional[CompensationAmount]]
    job_level: NotRequired[Optional[str]]
    payload: NotRequired[Optional[Dict[str, Any]]]

class SnapshotRow(TypedDict):
    """Structure for a single employee row in a snapshot."""
    employee_id: EmployeeId
    employee_hire_date: datetime
    employee_birth_date: NotRequired[Optional[datetime]]
    employee_gross_compensation: CompensationAmount
    employee_termination_date: NotRequired[Optional[datetime]]
    active: bool
    employee_deferral_rate: float
    employee_tenure: TenureYears
    employee_tenure_band: str
    employee_level: str
    job_level_source: str
    exited: bool
    employee_status_eoy: str
    simulation_year: SimulationYear
    employee_contribution: float
    employer_core_contribution: float
    employer_match_contribution: float
    is_eligible: bool

class ValidationResult(TypedDict):
    """Result of data validation operations."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]

class PerformanceMetrics(TypedDict):
    """Performance metrics for operations."""
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    memory_start_mb: float
    memory_end_mb: Optional[float]
    memory_delta_mb: Optional[float]
    checkpoints: Dict[str, Dict[str, Any]]
    final_metrics: Dict[str, Any]

class CompensationExtractionResult(TypedDict):
    """Result of compensation extraction for an employee."""
    compensation: Optional[CompensationAmount]
    source: str
    confidence: str  # 'high', 'medium', 'low'
    details: Dict[str, Any]

class StatusMetrics(TypedDict):
    """Metrics about employee status distribution."""
    status_counts: Dict[str, int]
    total_employees: int
    active_percentage: float
    terminated_percentage: float
    compensation_by_status: Dict[str, Dict[str, float]]

# Protocol definitions for interfaces
class SnapshotProcessor(Protocol):
    """Protocol for snapshot processing components."""
    
    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Process snapshot data and return updated DataFrame."""
        ...

class DataValidator(Protocol):
    """Protocol for data validation components."""
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate data and return validation result."""
        ...

class DataTransformer(Protocol):
    """Protocol for data transformation components."""
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data and return updated DataFrame."""
        ...

class EventProcessor(Protocol):
    """Protocol for event processing components."""
    
    def process_events(self, events: pd.DataFrame) -> pd.DataFrame:
        """Process events and return processed results."""
        ...

class Logger(Protocol):
    """Protocol for logging components."""
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with optional context."""
        ...
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with optional context."""
        ...
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with optional context."""
        ...
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message with optional context."""
        ...

# Configuration type definitions
class CensusLoaderConfig(TypedDict):
    """Configuration for census data loading."""
    required_columns: List[ColumnName]
    column_mappings: ColumnMapping
    validation_enabled: bool
    default_compensation: CompensationAmount

class TransformerConfig(TypedDict):
    """Configuration for data transformations."""
    start_year: SimulationYear
    tenure_bands: Dict[str, tuple]
    age_bands: Dict[str, tuple]
    compensation_defaults: Dict[str, CompensationAmount]

class ValidatorConfig(TypedDict):
    """Configuration for data validation."""
    min_compensation: CompensationAmount
    max_compensation: CompensationAmount
    min_age: AgeYears
    max_age: AgeYears
    max_tenure: TenureYears
    max_deferral_rate: float

class LoggingConfig(TypedDict):
    """Configuration for logging system."""
    enable_timing: bool
    enable_memory_tracking: bool
    log_level: str
    structured_format: bool

# Complex data structure types
EmployeeSet = Set[EmployeeId]
ColumnMissingData = Dict[ColumnName, int]
ProcessingCheckpoints = Dict[str, Dict[str, Any]]

# Function type definitions
TimingDecorator = Callable[[Callable], Callable]
ProgressTracker = Callable[[int, str], None]

# Result type definitions
class SnapshotCreationResult(TypedDict):
    """Result of snapshot creation operation."""
    snapshot: pd.DataFrame
    metrics: PerformanceMetrics
    validation_results: ValidationResult
    processing_time_seconds: float
    employee_count: int
    column_count: int

class YearlySnapshotResult(TypedDict):
    """Result of yearly snapshot building operation."""
    snapshot: pd.DataFrame
    active_employees: int
    terminated_employees: int
    reconstructed_employees: int
    status_distribution: Dict[str, int]
    performance_metrics: PerformanceMetrics

# Error context types
class ErrorContext(TypedDict):
    """Context information for error reporting."""
    operation: str
    employee_id: Optional[EmployeeId]
    simulation_year: Optional[SimulationYear]
    processing_step: Optional[str]
    data_shape: Optional[tuple]
    timestamp: datetime
    additional_context: Dict[str, Any]

# Utility type definitions
class DataFrameInfo(TypedDict):
    """Information about a DataFrame for logging."""
    name: str
    rows: int
    columns: int
    memory_mb: float
    dtypes_summary: Dict[str, int]
    null_counts: Dict[str, int]
    unique_counts: Dict[str, int]