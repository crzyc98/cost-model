# cost_model/state/snapshot/event_processors/__init__.py
"""
Event processor modules for snapshot updates.
Implements strategy pattern for different event types.
"""

from .base import BaseEventProcessor, EventProcessorResult
from .hire_processor import HireEventProcessor
from .termination_processor import TerminationEventProcessor
from .compensation_processor import CompensationEventProcessor
from .promotion_processor import PromotionEventProcessor
from .contribution_processor import ContributionEventProcessor
from .processor_registry import EventProcessorRegistry

__all__ = [
    'BaseEventProcessor',
    'EventProcessorResult',
    'HireEventProcessor',
    'TerminationEventProcessor',
    'CompensationEventProcessor',
    'PromotionEventProcessor',
    'ContributionEventProcessor',
    'EventProcessorRegistry'
]