"""
Event processing module for snapshot system.

This module provides a unified interface for handling all types of events
that affect workforce snapshots, including hiring, termination, promotion,
and compensation events.

Example Usage:
    >>> from cost_model.projections.snapshot.events import EventProcessor
    >>> processor = EventProcessor(config)
    >>> updated_snapshot = processor.process_events(snapshot, events)
"""

from .base import BaseEventProcessor, EventContext, EventProcessorProtocol
from .core import EventProcessingResult, EventProcessor
from .handlers import (
    CompensationEventHandler,
    HireEventHandler,
    PromotionEventHandler,
    TerminationEventHandler,
)
from .registry import EventHandlerRegistry
from .validation import EventValidator

__all__ = [
    "BaseEventProcessor",
    "EventProcessorProtocol",
    "EventContext",
    "EventProcessor",
    "EventProcessingResult",
    "HireEventHandler",
    "TerminationEventHandler",
    "PromotionEventHandler",
    "CompensationEventHandler",
    "EventHandlerRegistry",
    "EventValidator",
]
