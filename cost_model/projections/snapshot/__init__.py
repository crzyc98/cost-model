"""
Snapshot module for creating and updating workforce snapshots during projections.

This module provides a refactored, modular approach to snapshot creation and management.
The main functions are available for backward compatibility.
"""

from .builder import create_initial_snapshot, build_enhanced_yearly_snapshot
from .event_processor import EventProcessor
from .consolidator import consolidate_snapshots_to_parquet

# Create a default event processor instance for backward compatibility
_default_event_processor = EventProcessor()

def update_snapshot_with_events(prev_snapshot, events, target_date):
    """Backward compatibility wrapper for update_snapshot_with_events."""
    return _default_event_processor.update_snapshot_with_events(prev_snapshot, events, target_date)

__all__ = [
    'create_initial_snapshot',
    'build_enhanced_yearly_snapshot', 
    'update_snapshot_with_events',
    'consolidate_snapshots_to_parquet'
]