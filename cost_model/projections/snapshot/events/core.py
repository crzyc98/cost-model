"""
Core event processing orchestration.

This module provides the main EventProcessor class that coordinates
event processing across different event types and handlers.
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import logging
import time

from cost_model.schema import EventTypes, EventColumns, SnapshotColumns
from .base import (
    BaseEventProcessor, EventContext, EventProcessingStats, 
    EventProcessingError, EventValidationError
)
from .registry import EventHandlerRegistry
from .validation import EventValidator

logger = logging.getLogger(__name__)


@dataclass
class EventProcessingResult:
    """Result of event processing operation."""
    success: bool
    updated_snapshot: Optional[pd.DataFrame] = None
    events_processed: int = 0
    processing_time_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    handler_stats: Dict[str, EventProcessingStats] = field(default_factory=dict)
    
    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)
        self.success = False
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of processing results."""
        return {
            "success": self.success,
            "events_processed": self.events_processed,
            "processing_time": self.processing_time_seconds,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "handlers_used": list(self.handler_stats.keys())
        }


class EventProcessor:
    """
    Main event processor that orchestrates event handling.
    
    This class coordinates event processing across multiple event types,
    manages event ordering, and provides comprehensive error handling
    and reporting.
    """
    
    def __init__(self, config: Any = None):
        """
        Initialize the event processor.
        
        Args:
            config: Configuration object for event processing
        """
        self.config = config
        self.registry = EventHandlerRegistry()
        self.validator = EventValidator()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Register default handlers
        self._register_default_handlers()
    
    def process_events(self, 
                      snapshot: pd.DataFrame, 
                      events: pd.DataFrame,
                      simulation_year: int,
                      reference_date: Optional[datetime] = None) -> EventProcessingResult:
        """
        Process all events and return updated snapshot.
        
        Args:
            snapshot: Current snapshot DataFrame
            events: Events to process
            simulation_year: Current simulation year
            reference_date: Reference date for processing (defaults to year start)
            
        Returns:
            Event processing result with updated snapshot and statistics
        """
        start_time = time.time()
        
        if reference_date is None:
            reference_date = datetime(simulation_year, 1, 1)
        
        context = EventContext(
            simulation_year=simulation_year,
            reference_date=reference_date,
            config=self.config
        )
        
        result = EventProcessingResult(
            success=True,
            updated_snapshot=snapshot.copy()
        )
        
        try:
            # Validate input data
            self._validate_inputs(snapshot, events, result)
            if not result.success:
                return result
            
            # Pre-process events (sorting, grouping, validation)
            processed_events = self._preprocess_events(events, context, result)
            if not result.success:
                return result
            
            # Group events by type and process in order
            event_groups = self._group_events_by_type(processed_events)
            
            # Process each event type in priority order
            for event_type in self._get_processing_order():
                if event_type not in event_groups:
                    continue
                
                type_events = event_groups[event_type]
                self.logger.debug(f"Processing {len(type_events)} {event_type} events")
                
                # Get appropriate handler
                handler = self.registry.get_handler(event_type)
                if handler is None:
                    result.add_warning(f"No handler found for event type: {event_type}")
                    continue
                
                try:
                    # Process events with this handler
                    result.updated_snapshot = handler.process_events(
                        result.updated_snapshot, type_events, context
                    )
                    
                    # Collect statistics
                    handler_stats = handler.get_processing_stats()
                    result.handler_stats[event_type] = handler_stats
                    result.events_processed += handler_stats.events_processed
                    
                    self.logger.debug(
                        f"Completed {event_type} processing: "
                        f"{handler_stats.events_successful}/{handler_stats.events_processed} successful"
                    )
                    
                except Exception as e:
                    error_msg = f"Error processing {event_type} events: {str(e)}"
                    self.logger.error(error_msg, exc_info=True)
                    result.add_error(error_msg)
            
            # Post-processing validation
            self._validate_final_snapshot(result.updated_snapshot, context, result)
            
        except Exception as e:
            error_msg = f"Critical error in event processing: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            result.add_error(error_msg)
        
        finally:
            result.processing_time_seconds = time.time() - start_time
            self._log_processing_summary(result)
        
        return result
    
    def register_handler(self, event_type: str, handler: BaseEventProcessor):
        """
        Register a custom event handler.
        
        Args:
            event_type: Event type to handle
            handler: Event handler instance
        """
        self.registry.register_handler(event_type, handler)
        self.logger.info(f"Registered custom handler for {event_type}")
    
    def get_supported_event_types(self) -> Set[str]:
        """Get set of supported event types."""
        return self.registry.get_supported_types()
    
    def validate_events_only(self, 
                            events: pd.DataFrame,
                            simulation_year: int,
                            reference_date: Optional[datetime] = None) -> List[str]:
        """
        Validate events without processing them.
        
        Args:
            events: Events to validate
            simulation_year: Current simulation year
            reference_date: Reference date for validation
            
        Returns:
            List of validation error messages
        """
        if reference_date is None:
            reference_date = datetime(simulation_year, 1, 1)
        
        context = EventContext(
            simulation_year=simulation_year,
            reference_date=reference_date,
            config=self.config
        )
        
        return self.validator.validate_events(events, context)
    
    def _register_default_handlers(self):
        """Register default event handlers."""
        try:
            from .handlers import (
                HireEventHandler, TerminationEventHandler,
                PromotionEventHandler, CompensationEventHandler
            )
            
            # Register handlers for standard event types
            self.registry.register_handler(EventTypes.HIRE, HireEventHandler())
            self.registry.register_handler(EventTypes.TERMINATION, TerminationEventHandler())
            self.registry.register_handler(EventTypes.NEW_HIRE_TERMINATION, TerminationEventHandler())
            self.registry.register_handler(EventTypes.PROMOTION, PromotionEventHandler())
            self.registry.register_handler(EventTypes.COMPENSATION, CompensationEventHandler())
            
            self.logger.info("Registered default event handlers")
            
        except ImportError as e:
            self.logger.warning(f"Could not import default handlers: {e}")
    
    def _validate_inputs(self, 
                        snapshot: pd.DataFrame, 
                        events: pd.DataFrame,
                        result: EventProcessingResult):
        """Validate input data."""
        if snapshot is None or snapshot.empty:
            result.add_error("Snapshot is None or empty")
            return
        
        if events is None or events.empty:
            result.add_warning("No events to process")
            return
        
        # Validate required columns
        required_snapshot_cols = [SnapshotColumns.EMP_ID]
        missing_snapshot_cols = [col for col in required_snapshot_cols if col not in snapshot.columns]
        if missing_snapshot_cols:
            result.add_error(f"Snapshot missing required columns: {missing_snapshot_cols}")
        
        required_event_cols = [EventColumns.EVENT_TYPE, EventColumns.EMP_ID, EventColumns.EVENT_DATE]
        missing_event_cols = [col for col in required_event_cols if col not in events.columns]
        if missing_event_cols:
            result.add_error(f"Events missing required columns: {missing_event_cols}")
    
    def _preprocess_events(self, 
                          events: pd.DataFrame,
                          context: EventContext,
                          result: EventProcessingResult) -> pd.DataFrame:
        """Preprocess events (sort, validate, clean)."""
        try:
            # Validate events
            validation_errors = self.validator.validate_events(events, context)
            for error in validation_errors:
                result.add_error(error)
            
            if validation_errors and self.config and getattr(self.config, 'strict_validation', False):
                return pd.DataFrame()  # Return empty DataFrame on validation failure
            
            # Sort events by date and priority
            processed_events = events.copy()
            
            # Convert event dates
            processed_events[EventColumns.EVENT_DATE] = pd.to_datetime(
                processed_events[EventColumns.EVENT_DATE]
            )
            
            # Sort by event date, then by processing priority
            event_priority_map = self._get_event_priority_map()
            processed_events['_processing_priority'] = processed_events[EventColumns.EVENT_TYPE].map(
                event_priority_map
            ).fillna(99)  # Unknown events get low priority
            
            processed_events = processed_events.sort_values([
                EventColumns.EVENT_DATE,
                '_processing_priority',
                EventColumns.EMP_ID
            ]).drop(columns=['_processing_priority'])
            
            self.logger.debug(f"Preprocessed {len(processed_events)} events")
            return processed_events
            
        except Exception as e:
            result.add_error(f"Error preprocessing events: {str(e)}")
            return pd.DataFrame()
    
    def _group_events_by_type(self, events: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Group events by event type."""
        if events.empty:
            return {}
        
        return {
            event_type: group_events.reset_index(drop=True)
            for event_type, group_events in events.groupby(EventColumns.EVENT_TYPE)
        }
    
    def _get_processing_order(self) -> List[str]:
        """Get the order in which event types should be processed."""
        return [
            EventTypes.TERMINATION,           # Process terminations first
            EventTypes.NEW_HIRE_TERMINATION,  # Then new hire terminations
            EventTypes.HIRE,                  # Then hires
            EventTypes.PROMOTION,             # Then promotions
            EventTypes.COMPENSATION,          # Finally compensation changes
        ]
    
    def _get_event_priority_map(self) -> Dict[str, int]:
        """Get priority mapping for event types (lower number = higher priority)."""
        processing_order = self._get_processing_order()
        return {event_type: i for i, event_type in enumerate(processing_order)}
    
    def _validate_final_snapshot(self, 
                                snapshot: pd.DataFrame,
                                context: EventContext,
                                result: EventProcessingResult):
        """Validate the final snapshot after processing."""
        try:
            from cost_model.schema import validate_snapshot_schema
            
            validation_result = validate_snapshot_schema(
                snapshot, 
                strict_mode=False,
                check_data_quality=True
            )
            
            # Add validation messages to result
            for error in validation_result.errors:
                result.add_error(f"Final snapshot validation: {error}")
            
            for warning in validation_result.warnings:
                result.add_warning(f"Final snapshot validation: {warning}")
                
        except Exception as e:
            result.add_warning(f"Could not validate final snapshot: {str(e)}")
    
    def _log_processing_summary(self, result: EventProcessingResult):
        """Log summary of processing operation."""
        summary = result.get_summary()
        
        self.logger.info(
            f"Event processing completed: {summary['events_processed']} events in "
            f"{summary['processing_time']:.2f}s (Success: {summary['success']})"
        )
        
        if summary['error_count'] > 0:
            self.logger.error(f"Processing completed with {summary['error_count']} errors")
        
        if summary['warning_count'] > 0:
            self.logger.warning(f"Processing completed with {summary['warning_count']} warnings")
        
        # Log handler statistics
        for handler_type, stats in result.handler_stats.items():
            self.logger.debug(
                f"{handler_type} handler: {stats.events_successful}/{stats.events_processed} "
                f"successful ({stats.success_rate:.1f}%), {stats.rows_affected} rows affected"
            )