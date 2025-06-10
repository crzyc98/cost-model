# cost_model/state/snapshot/event_processors/base.py
"""
Base classes for event processors in the snapshot update system.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from cost_model.state.schema import EMP_ID


@dataclass
class EventProcessorResult:
    """Result object for event processing operations."""

    success: bool = True
    updated_snapshot: Optional[pd.DataFrame] = None
    errors: List[str] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None
    employees_affected: Set[str] = None

    def __post_init__(self):
        """Initialize mutable default values."""
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}
        if self.employees_affected is None:
            self.employees_affected = set()

    def add_error(self, error: str) -> None:
        """Add an error and mark result as failed."""
        self.errors.append(error)
        self.success = False

    def add_warning(self, warning: str) -> None:
        """Add a warning."""
        self.warnings.append(warning)

    def add_affected_employee(self, emp_id: str) -> None:
        """Add an employee ID to the affected set."""
        self.employees_affected.add(emp_id)


class BaseEventProcessor(ABC):
    """
    Base class for all event processors.

    Event processors handle specific types of events using the strategy pattern,
    allowing for clean separation of concerns and easier testing.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the event processor.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def get_event_types(self) -> List[str]:
        """
        Return the list of event types this processor handles.

        Returns:
            List of event type strings
        """
        pass

    @abstractmethod
    def process_events(
        self, snapshot: pd.DataFrame, events: pd.DataFrame, snapshot_year: int
    ) -> EventProcessorResult:
        """
        Process events of this processor's type.

        Args:
            snapshot: Current snapshot DataFrame
            events: Events DataFrame (filtered to this processor's types)
            snapshot_year: Current simulation year

        Returns:
            EventProcessorResult with updated snapshot and metadata
        """
        pass

    def validate_events(self, events: pd.DataFrame) -> List[str]:
        """
        Validate events for this processor.

        Args:
            events: Events DataFrame to validate

        Returns:
            List of validation error messages
        """
        errors = []

        if events.empty:
            return errors

        # Check for required columns
        required_columns = self.get_required_columns()
        missing_columns = [col for col in required_columns if col not in events.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")

        # Check for null employee IDs
        if EMP_ID in events.columns:
            null_emp_ids = events[EMP_ID].isna().sum()
            if null_emp_ids > 0:
                errors.append(f"Found {null_emp_ids} events with null employee IDs")

        return errors

    def get_required_columns(self) -> List[str]:
        """
        Return the list of required columns for this processor's events.

        Returns:
            List of required column names
        """
        return [EMP_ID]  # Default: all processors need employee ID

    def filter_events(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Filter events to only those handled by this processor.

        Args:
            events: All events DataFrame

        Returns:
            Filtered events DataFrame
        """
        if events.empty:
            return events

        event_types = self.get_event_types()
        if "event_type" in events.columns:
            return events[events["event_type"].isin(event_types)]
        else:
            # Fallback: assume all events are relevant
            return events

    def log_processing_start(self, events_count: int, snapshot_size: int) -> None:
        """Log the start of event processing."""
        self.logger.info(
            f"Processing {events_count} {self.__class__.__name__} events on snapshot of {snapshot_size} employees"
        )

    def log_processing_end(self, result: EventProcessorResult) -> None:
        """Log the end of event processing."""
        if result.success:
            self.logger.info(
                f"Successfully processed events. Affected {len(result.employees_affected)} employees"
            )
            if result.warnings:
                self.logger.warning(f"Processing completed with {len(result.warnings)} warnings")
        else:
            self.logger.error(f"Event processing failed with {len(result.errors)} errors")

    def ensure_snapshot_integrity(
        self, original_snapshot: pd.DataFrame, updated_snapshot: pd.DataFrame, events: pd.DataFrame
    ) -> List[str]:
        """
        Ensure snapshot integrity after processing.

        Args:
            original_snapshot: Original snapshot before processing
            updated_snapshot: Updated snapshot after processing
            events: Events that were processed

        Returns:
            List of integrity issues found
        """
        issues = []

        if original_snapshot.empty and updated_snapshot.empty:
            return issues

        # Check for unexpected employee loss
        if EMP_ID in original_snapshot.columns and EMP_ID in updated_snapshot.columns:
            original_ids = set(original_snapshot[EMP_ID].unique())
            updated_ids = set(updated_snapshot[EMP_ID].unique())

            lost_employees = original_ids - updated_ids
            if lost_employees:
                # Check if these employees had termination events
                if not events.empty and EMP_ID in events.columns:
                    event_employees = set(events[EMP_ID].unique())
                    unexpected_losses = lost_employees - event_employees
                    if unexpected_losses:
                        issues.append(
                            f"Unexpected employee loss: {len(unexpected_losses)} employees"
                        )
                else:
                    issues.append(f"Employee loss without events: {len(lost_employees)} employees")

        # Check for duplicate employees
        if EMP_ID in updated_snapshot.columns:
            duplicates = updated_snapshot[EMP_ID].duplicated().sum()
            if duplicates > 0:
                issues.append(f"Duplicate employees in updated snapshot: {duplicates}")

        return issues
