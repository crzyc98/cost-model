# cost_model/engines/run_one_year/processors/__init__.py
"""
Processor modules for the refactored orchestrator.
Each processor handles a specific aspect of the yearly simulation.
"""

from .contribution_processor import ContributionProcessor
from .diagnostic_logger import DiagnosticLogger
from .event_consolidator import EventConsolidator
from .hiring_processor import HiringProcessor
from .promotion_processor import PromotionProcessor
from .termination_processor import TerminationProcessor

__all__ = [
    "PromotionProcessor",
    "TerminationProcessor",
    "HiringProcessor",
    "ContributionProcessor",
    "EventConsolidator",
    "DiagnosticLogger",
]
