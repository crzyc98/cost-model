# cost_model/state/snapshot/event_processors/processor_registry.py
"""
Registry for managing event processors in the snapshot update system.
"""
import pandas as pd
import logging
from typing import Dict, List, Type

from .base import BaseEventProcessor, EventProcessorResult
from .hire_processor import HireEventProcessor
from .termination_processor import TerminationEventProcessor
from .compensation_processor import CompensationEventProcessor
from .promotion_processor import PromotionEventProcessor
from .contribution_processor import ContributionEventProcessor


class EventProcessorRegistry:
    """
    Registry that manages all event processors and coordinates their execution.
    
    This class implements the central coordination logic for the strategy pattern,
    routing events to appropriate processors and managing the overall update process.
    """
    
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize the registry with default processors.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.processors: Dict[str, BaseEventProcessor] = {}
        
        # Register default processors
        self._register_default_processors()
    
    def _register_default_processors(self) -> None:
        """Register the default set of event processors."""
        default_processors = [
            HireEventProcessor(self.logger),
            TerminationEventProcessor(self.logger),
            CompensationEventProcessor(self.logger),
            PromotionEventProcessor(self.logger),
            ContributionEventProcessor(self.logger)
        ]
        
        for processor in default_processors:
            self.register_processor(processor)
    
    def register_processor(self, processor: BaseEventProcessor) -> None:
        """
        Register a new event processor.
        
        Args:
            processor: Event processor instance to register
        """
        for event_type in processor.get_event_types():
            if event_type in self.processors:
                self.logger.warning(f"Overriding existing processor for event type: {event_type}")
            self.processors[event_type] = processor
            self.logger.debug(f"Registered processor {processor.__class__.__name__} for {event_type}")
    
    def get_processor(self, event_type: str) -> BaseEventProcessor:
        """
        Get the processor for a specific event type.
        
        Args:
            event_type: Type of event
        
        Returns:
            Event processor instance or None if not found
        """
        return self.processors.get(event_type)
    
    def process_all_events(
        self,
        snapshot: pd.DataFrame,
        events: pd.DataFrame,
        snapshot_year: int
    ) -> EventProcessorResult:
        """
        Process all events using appropriate processors.
        
        Args:
            snapshot: Current snapshot DataFrame
            events: All events DataFrame
            snapshot_year: Current simulation year
        
        Returns:
            Consolidated EventProcessorResult
        """
        self.logger.info(f"Processing {len(events)} events with {len(self.processors)} registered processors")
        
        # Initialize result
        consolidated_result = EventProcessorResult()
        consolidated_result.updated_snapshot = snapshot.copy()
        
        if events.empty:
            self.logger.info("No events to process")
            return consolidated_result
        
        # Group events by type
        event_groups = self._group_events_by_type(events)
        
        # Process events in order of priority
        processing_order = self._get_processing_order()
        
        for event_type in processing_order:
            if event_type not in event_groups:
                continue
            
            event_group = event_groups[event_type]
            processor = self.get_processor(event_type)
            
            if processor is None:
                consolidated_result.add_warning(f"No processor found for event type: {event_type}")
                continue
            
            # Process this group of events
            result = processor.process_events(
                consolidated_result.updated_snapshot,
                event_group,
                snapshot_year
            )
            
            # Consolidate results
            if result.success:
                consolidated_result.updated_snapshot = result.updated_snapshot
                consolidated_result.employees_affected.update(result.employees_affected)
            else:
                # If any processor fails, stop processing
                consolidated_result.success = False
                consolidated_result.errors.extend(result.errors)
                break
            
            # Add warnings and metadata
            consolidated_result.warnings.extend(result.warnings)
            consolidated_result.metadata.update(result.metadata)
        
        # Final integrity check
        if consolidated_result.success:
            integrity_issues = self._final_integrity_check(
                snapshot, consolidated_result.updated_snapshot, events
            )
            for issue in integrity_issues:
                consolidated_result.add_warning(f"Final integrity check: {issue}")
        
        self.logger.info(f"Event processing completed. Success: {consolidated_result.success}, "
                        f"Affected employees: {len(consolidated_result.employees_affected)}")
        
        return consolidated_result
    
    def _group_events_by_type(self, events: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Group events by their type.
        
        Args:
            events: All events DataFrame
        
        Returns:
            Dictionary mapping event types to event DataFrames
        """
        event_groups = {}
        
        if 'event_type' not in events.columns:
            self.logger.warning("No event_type column found, cannot group events")
            return event_groups
        
        for event_type in events['event_type'].unique():
            if pd.notna(event_type):
                event_groups[event_type] = events[events['event_type'] == event_type]
        
        return event_groups
    
    def _get_processing_order(self) -> List[str]:
        """
        Get the order in which event types should be processed.
        
        Returns:
            List of event types in processing order
        """
        # Define processing order (some events must come before others)
        return [
            'hire',           # Hires first (add employees)
            'promotion',      # Then promotions
            'compensation',   # Then compensation changes
            'cola',          # Then cost of living adjustments
            'raise',         # Then raises
            'contribution',   # Then contribution updates
            'termination',    # Finally terminations (remove employees)
            'new_hire_termination'
        ]
    
    def _final_integrity_check(
        self,
        original_snapshot: pd.DataFrame,
        final_snapshot: pd.DataFrame,
        events: pd.DataFrame
    ) -> List[str]:
        """
        Perform final integrity check on the updated snapshot.
        
        Args:
            original_snapshot: Original snapshot before processing
            final_snapshot: Final snapshot after all processing
            events: All events that were processed
        
        Returns:
            List of integrity issues found
        """
        issues = []
        
        # Check for unexpected data loss
        if len(final_snapshot) < len(original_snapshot):
            difference = len(original_snapshot) - len(final_snapshot)
            issues.append(f"Snapshot size decreased by {difference} employees")
        
        # Check for null employee IDs
        if 'employee_id' in final_snapshot.columns:
            null_ids = final_snapshot['employee_id'].isna().sum()
            if null_ids > 0:
                issues.append(f"Found {null_ids} employees with null IDs")
        
        # Check for duplicate employee IDs
        if 'employee_id' in final_snapshot.columns:
            duplicates = final_snapshot['employee_id'].duplicated().sum()
            if duplicates > 0:
                issues.append(f"Found {duplicates} duplicate employee IDs")
        
        return issues
    
    def get_registered_processors(self) -> Dict[str, str]:
        """
        Get information about registered processors.
        
        Returns:
            Dictionary mapping event types to processor class names
        """
        return {
            event_type: processor.__class__.__name__
            for event_type, processor in self.processors.items()
        }