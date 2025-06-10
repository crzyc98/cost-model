# cost_model/state/snapshot_update_refactored.py
"""
Refactored snapshot update module using strategy pattern for event processing.

This module replaces the monolithic update function with a clean, modular design
that uses separate event processors for different types of events.
"""
import logging
import pandas as pd
from typing import Optional

from cost_model.state.schema import SIMULATION_YEAR, SNAPSHOT_COLS, SNAPSHOT_DTYPES
from cost_model.state.snapshot.event_processors import EventProcessorRegistry
from cost_model.state.snapshot_utils import ensure_columns_and_types

logger = logging.getLogger(__name__)


def update(prev_snapshot: pd.DataFrame, new_events: pd.DataFrame, snapshot_year: int) -> pd.DataFrame:
    """
    Return a new snapshot by applying new_events to prev_snapshot using modular event processors.
    
    This function maintains the same interface as the original update function but uses
    the strategy pattern with separate event processors for better maintainability.

    Args:
        prev_snapshot: Previous snapshot DataFrame
        new_events: DataFrame containing new events to apply
        snapshot_year: Current simulation year

    Returns:
        New snapshot DataFrame with updates applied
    """
    logger.debug("Starting refactored snapshot update for year %d", snapshot_year)
    
    # Initialize the event processor registry
    processor_registry = EventProcessorRegistry(logger)
    
    # Validate inputs
    if prev_snapshot.empty and new_events.empty:
        logger.debug("Both snapshot and events are empty, returning empty snapshot")
        return _create_empty_snapshot(snapshot_year)
    
    # Handle case where there are no events
    if new_events.empty:
        logger.debug("No new events to apply, returning copy of previous snapshot")
        result_snapshot = prev_snapshot.copy()
        result_snapshot[SIMULATION_YEAR] = snapshot_year
        return _finalize_snapshot(result_snapshot)
    
    # Log processing start
    initial_employee_count = len(prev_snapshot)
    events_count = len(new_events)
    logger.info(f"Processing {events_count} events on snapshot with {initial_employee_count} employees")
    
    try:
        # Process all events using the registry
        processing_result = processor_registry.process_all_events(
            snapshot=prev_snapshot,
            events=new_events,
            snapshot_year=snapshot_year
        )
        
        if not processing_result.success:
            logger.error(f"Event processing failed: {processing_result.errors}")
            # Return original snapshot with updated year as fallback
            fallback_snapshot = prev_snapshot.copy()
            fallback_snapshot[SIMULATION_YEAR] = snapshot_year
            return _finalize_snapshot(fallback_snapshot)
        
        # Log warnings if any
        if processing_result.warnings:
            for warning in processing_result.warnings:
                logger.warning(f"Event processing warning: {warning}")
        
        # Finalize the snapshot
        final_snapshot = processing_result.updated_snapshot
        final_snapshot[SIMULATION_YEAR] = snapshot_year
        final_snapshot = _finalize_snapshot(final_snapshot)
        
        # Log completion
        final_employee_count = len(final_snapshot)
        employees_affected = len(processing_result.employees_affected)
        
        logger.info(f"Snapshot update completed successfully:")
        logger.info(f"  Initial employees: {initial_employee_count}")
        logger.info(f"  Final employees: {final_employee_count}")
        logger.info(f"  Employees affected: {employees_affected}")
        logger.info(f"  Events processed: {events_count}")
        
        return final_snapshot
        
    except Exception as e:
        logger.error(f"Error during snapshot update: {e}", exc_info=True)
        # Return original snapshot with updated year as fallback
        fallback_snapshot = prev_snapshot.copy()
        fallback_snapshot[SIMULATION_YEAR] = snapshot_year
        return _finalize_snapshot(fallback_snapshot)


def _create_empty_snapshot(snapshot_year: int) -> pd.DataFrame:
    """
    Create an empty snapshot with proper schema.
    
    Args:
        snapshot_year: Current simulation year
    
    Returns:
        Empty snapshot DataFrame with correct columns and types
    """
    empty_snapshot = pd.DataFrame(columns=SNAPSHOT_COLS)
    
    # Apply correct data types
    for col, dtype in SNAPSHOT_DTYPES.items():
        if col in empty_snapshot.columns:
            try:
                empty_snapshot[col] = empty_snapshot[col].astype(dtype)
            except Exception as e:
                logger.warning(f"Could not apply dtype {dtype} to column {col}: {e}")
    
    empty_snapshot[SIMULATION_YEAR] = snapshot_year
    return empty_snapshot


def _finalize_snapshot(snapshot: pd.DataFrame) -> pd.DataFrame:
    """
    Finalize snapshot by ensuring proper columns and types.
    
    Args:
        snapshot: Snapshot DataFrame to finalize
    
    Returns:
        Finalized snapshot DataFrame
    """
    try:
        # Ensure all required columns exist and have correct types
        finalized_snapshot = ensure_columns_and_types(snapshot)
        
        # Validate final result
        _validate_final_snapshot(finalized_snapshot)
        
        return finalized_snapshot
        
    except Exception as e:
        logger.error(f"Error finalizing snapshot: {e}", exc_info=True)
        # Return snapshot as-is if finalization fails
        return snapshot


def _validate_final_snapshot(snapshot: pd.DataFrame) -> None:
    """
    Validate the final snapshot for basic integrity.
    
    Args:
        snapshot: Snapshot DataFrame to validate
    """
    if snapshot.empty:
        return
    
    # Check for duplicate employee IDs
    if 'employee_id' in snapshot.columns:
        duplicates = snapshot['employee_id'].duplicated().sum()
        if duplicates > 0:
            logger.error(f"Final snapshot has {duplicates} duplicate employee IDs")
    
    # Check for null employee IDs
    if 'employee_id' in snapshot.columns:
        null_ids = snapshot['employee_id'].isna().sum()
        if null_ids > 0:
            logger.error(f"Final snapshot has {null_ids} null employee IDs")
    
    # Log final statistics
    logger.debug(f"Final snapshot validation: {len(snapshot)} employees")


# Additional utility functions for backward compatibility
def update_with_custom_processors(
    prev_snapshot: pd.DataFrame,
    new_events: pd.DataFrame,
    snapshot_year: int,
    custom_processors: Optional[dict] = None
) -> pd.DataFrame:
    """
    Update snapshot with custom event processors.
    
    Args:
        prev_snapshot: Previous snapshot DataFrame
        new_events: DataFrame containing new events to apply
        snapshot_year: Current simulation year
        custom_processors: Dictionary mapping event types to custom processor instances
    
    Returns:
        New snapshot DataFrame with updates applied
    """
    logger.debug("Starting snapshot update with custom processors")
    
    # Initialize registry
    processor_registry = EventProcessorRegistry(logger)
    
    # Register custom processors if provided
    if custom_processors:
        for event_type, processor in custom_processors.items():
            processor_registry.register_processor(processor)
            logger.info(f"Registered custom processor for {event_type}")
    
    # Process events
    processing_result = processor_registry.process_all_events(
        snapshot=prev_snapshot,
        events=new_events,
        snapshot_year=snapshot_year
    )
    
    if not processing_result.success:
        logger.error(f"Custom event processing failed: {processing_result.errors}")
        fallback_snapshot = prev_snapshot.copy()
        fallback_snapshot[SIMULATION_YEAR] = snapshot_year
        return _finalize_snapshot(fallback_snapshot)
    
    # Finalize and return
    final_snapshot = processing_result.updated_snapshot
    final_snapshot[SIMULATION_YEAR] = snapshot_year
    return _finalize_snapshot(final_snapshot)


# Backward compatibility - export the same interface
__all__ = ['update', 'update_with_custom_processors']