# cost_model/state/snapshot/event_processors/__init__.py
"""
Event processor modules for snapshot updates.
Implements strategy pattern for different event types.
"""

from .base import BaseEventProcessor, EventProcessorResult
from .compensation_processor import CompensationEventProcessor
from .contribution_processor import ContributionEventProcessor
from .hire_processor import HireEventProcessor
from .processor_registry import EventProcessorRegistry
from .promotion_processor import PromotionEventProcessor
from .termination_processor import TerminationEventProcessor

__all__ = [
    "BaseEventProcessor",
    "EventProcessorResult",
    "HireEventProcessor",
    "TerminationEventProcessor",
    "CompensationEventProcessor",
    "PromotionEventProcessor",
    "ContributionEventProcessor",
    "EventProcessorRegistry",
]
