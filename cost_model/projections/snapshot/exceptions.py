"""
Custom exception classes for snapshot processing.

Provides specific exception types for different failure modes in snapshot
creation and processing, enabling better error handling and debugging.
"""


class SnapshotError(Exception):
    """Base exception for all snapshot-related errors."""

    pass


class CensusDataError(SnapshotError):
    """Raised when there are issues with census data loading or validation."""

    pass


class ValidationError(SnapshotError):
    """Raised when data validation fails during snapshot processing."""

    pass


class EventProcessingError(SnapshotError):
    """Raised when there are issues processing events during snapshot updates."""

    pass


class CompensationExtractionError(SnapshotError):
    """Raised when compensation cannot be extracted for an employee."""

    pass


class SnapshotBuildError(SnapshotError):
    """Raised when snapshot building fails due to data or logic errors."""

    pass
