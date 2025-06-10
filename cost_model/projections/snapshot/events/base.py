"""
Base classes and interfaces for event processing.

This module defines the core abstractions for event processing, including
the base event processor interface and common functionality shared across
all event handlers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Protocol, runtime_checkable
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import logging

from cost_model.schema import EventTypes, SnapshotColumns, EventColumns

logger = logging.getLogger(__name__)


@dataclass
class EventContext:
    """Context information for event processing."""
    simulation_year: int
    reference_date: datetime
    config: Any  # SnapshotConfig type
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value with default."""
        return self.metadata.get(key, default)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value."""
        self.metadata[key] = value


@dataclass
class EventProcessingStats:
    """Statistics for event processing operations."""
    events_processed: int = 0
    events_successful: int = 0
    events_failed: int = 0
    processing_time_seconds: float = 0.0
    rows_affected: int = 0
    warnings_count: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.events_processed == 0:
            return 0.0
        return (self.events_successful / self.events_processed) * 100.0
    
    def add_processed_event(self, success: bool = True, rows_affected: int = 0):
        """Add a processed event to statistics."""
        self.events_processed += 1
        if success:
            self.events_successful += 1
        else:
            self.events_failed += 1
        self.rows_affected += rows_affected
    
    def add_warning(self):
        """Add a warning to statistics."""
        self.warnings_count += 1


@runtime_checkable
class EventProcessorProtocol(Protocol):
    """Protocol defining the interface for event processors."""
    
    def can_handle(self, event_type: str) -> bool:
        """Check if this processor can handle the given event type."""
        ...
    
    def process_events(self, 
                      snapshot: pd.DataFrame, 
                      events: pd.DataFrame,
                      context: EventContext) -> pd.DataFrame:
        """Process events and return updated snapshot."""
        ...
    
    def validate_events(self, 
                       events: pd.DataFrame, 
                       context: EventContext) -> List[str]:
        """Validate events and return list of error messages."""
        ...


class BaseEventProcessor(ABC):
    """
    Base class for all event processors.
    
    This class provides common functionality and enforces the interface
    that all event processors must implement.
    """
    
    def __init__(self, supported_event_types: List[str]):
        """
        Initialize the base event processor.
        
        Args:
            supported_event_types: List of event types this processor can handle
        """
        self.supported_event_types = set(supported_event_types)
        self.stats = EventProcessingStats()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def can_handle(self, event_type: str) -> bool:
        """
        Check if this processor can handle the given event type.
        
        Args:
            event_type: The event type to check
            
        Returns:
            True if this processor can handle the event type
        """
        return event_type in self.supported_event_types
    
    @abstractmethod
    def process_events(self, 
                      snapshot: pd.DataFrame, 
                      events: pd.DataFrame,
                      context: EventContext) -> pd.DataFrame:
        """
        Process events and return updated snapshot.
        
        Args:
            snapshot: Current snapshot DataFrame
            events: Events to process
            context: Processing context and configuration
            
        Returns:
            Updated snapshot DataFrame
        """
        pass
    
    @abstractmethod
    def validate_events(self, 
                       events: pd.DataFrame, 
                       context: EventContext) -> List[str]:
        """
        Validate events and return list of error messages.
        
        Args:
            events: Events to validate
            context: Processing context
            
        Returns:
            List of error messages (empty if no errors)
        """
        pass
    
    def get_processing_stats(self) -> EventProcessingStats:
        """Get processing statistics for this processor."""
        return self.stats
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.stats = EventProcessingStats()
    
    def _validate_common_fields(self, 
                               events: pd.DataFrame, 
                               context: EventContext) -> List[str]:
        """
        Validate common fields present in all event types.
        
        Args:
            events: Events to validate
            context: Processing context
            
        Returns:
            List of error messages
        """
        errors = []
        
        # Check required columns
        required_columns = [
            EventColumns.EVENT_TYPE,
            EventColumns.EVENT_DATE,
            EventColumns.EMP_ID
        ]
        
        for col in required_columns:
            if col not in events.columns:
                errors.append(f"Missing required column: {col}")
        
        if errors:
            return errors
        
        # Validate event types
        invalid_types = ~events[EventColumns.EVENT_TYPE].isin(self.supported_event_types)
        if invalid_types.any():
            invalid_values = events.loc[invalid_types, EventColumns.EVENT_TYPE].unique()
            errors.append(f"Invalid event types for this processor: {list(invalid_values)}")
        
        # Validate employee IDs
        missing_emp_ids = events[EventColumns.EMP_ID].isna()
        if missing_emp_ids.any():
            errors.append(f"Found {missing_emp_ids.sum()} events with missing employee IDs")
        
        # Validate event dates
        try:
            event_dates = pd.to_datetime(events[EventColumns.EVENT_DATE])
            invalid_dates = event_dates.isna()
            if invalid_dates.any():
                errors.append(f"Found {invalid_dates.sum()} events with invalid dates")
        except Exception as e:
            errors.append(f"Error parsing event dates: {str(e)}")
        
        return errors
    
    def _update_snapshot_safely(self, 
                               snapshot: pd.DataFrame,
                               updates: Dict[str, Any],
                               employee_mask: pd.Series,
                               operation_name: str) -> pd.DataFrame:
        """
        Safely update snapshot with validation and error handling.
        
        Args:
            snapshot: Current snapshot
            updates: Dictionary of column updates
            employee_mask: Boolean mask for affected employees
            operation_name: Name of operation for logging
            
        Returns:
            Updated snapshot
        """
        try:
            updated_snapshot = snapshot.copy()
            affected_count = employee_mask.sum()
            
            if affected_count == 0:
                self.logger.warning(f"No employees affected by {operation_name}")
                return updated_snapshot
            
            self.logger.debug(f"Applying {operation_name} to {affected_count} employees")
            
            # Apply updates
            for column, values in updates.items():
                if column not in updated_snapshot.columns:
                    # Add new column if it doesn't exist
                    updated_snapshot[column] = None
                
                if isinstance(values, (list, pd.Series)):
                    # Apply values to specific rows
                    if len(values) != affected_count:
                        raise ValueError(f"Mismatch between mask size ({affected_count}) and values size ({len(values)})")
                    updated_snapshot.loc[employee_mask, column] = values
                else:
                    # Apply single value to all affected rows
                    updated_snapshot.loc[employee_mask, column] = values
            
            self.stats.add_processed_event(success=True, rows_affected=affected_count)
            self.logger.debug(f"Successfully applied {operation_name} to {affected_count} employees")
            
            return updated_snapshot
            
        except Exception as e:
            self.logger.error(f"Error applying {operation_name}: {str(e)}")
            self.stats.add_processed_event(success=False)
            raise
    
    def _get_employee_mask(self, 
                          snapshot: pd.DataFrame,
                          employee_ids: List[str]) -> pd.Series:
        """
        Get boolean mask for employees in the snapshot.
        
        Args:
            snapshot: Snapshot DataFrame
            employee_ids: List of employee IDs to match
            
        Returns:
            Boolean mask for matching employees
        """
        if SnapshotColumns.EMP_ID not in snapshot.columns:
            raise ValueError(f"Snapshot missing required column: {SnapshotColumns.EMP_ID}")
        
        return snapshot[SnapshotColumns.EMP_ID].isin(employee_ids)
    
    def _log_processing_summary(self, 
                               event_type: str, 
                               events_count: int,
                               processing_time: float) -> None:
        """
        Log summary of processing operation.
        
        Args:
            event_type: Type of events processed
            events_count: Number of events processed
            processing_time: Time taken for processing
        """
        self.logger.info(
            f"Processed {events_count} {event_type} events in {processing_time:.2f}s "
            f"(Success rate: {self.stats.success_rate:.1f}%, "
            f"Rows affected: {self.stats.rows_affected})"
        )
        
        if self.stats.warnings_count > 0:
            self.logger.warning(f"Processing generated {self.stats.warnings_count} warnings")


class EventProcessingError(Exception):
    """Base exception for event processing errors."""
    
    def __init__(self, message: str, event_type: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.event_type = event_type
        self.context = context or {}


class EventValidationError(EventProcessingError):
    """Exception raised when event validation fails."""
    pass


class EventApplicationError(EventProcessingError):
    """Exception raised when applying events to snapshot fails."""
    pass