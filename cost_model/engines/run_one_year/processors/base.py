# cost_model/engines/run_one_year/processors/base.py
"""
Base classes for orchestrator processors.
"""
import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class BaseProcessor(ABC):
    """Base class for all orchestrator processors."""
    
    def __init__(self, logger_name: Optional[str] = None):
        self.logger = logging.getLogger(logger_name or self.__class__.__name__)
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Main processing method to be implemented by subclasses."""
        pass
    
    def validate_inputs(self, **inputs) -> bool:
        """Validate processor inputs. Override in subclasses as needed."""
        return True
    
    def log_step_start(self, step_name: str, **context):
        """Log the start of a processing step."""
        self.logger.info(f"[STEP] {step_name}")
        for key, value in context.items():
            if isinstance(value, pd.DataFrame):
                self.logger.debug(f"  {key}: DataFrame with {len(value)} rows")
            else:
                self.logger.debug(f"  {key}: {value}")
    
    def log_step_end(self, step_name: str, **results):
        """Log the end of a processing step."""
        self.logger.info(f"[STEP COMPLETE] {step_name}")
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                self.logger.debug(f"  {key}: DataFrame with {len(value)} rows")
            else:
                self.logger.debug(f"  {key}: {value}")


class ProcessorResult:
    """Standard result object for processor operations."""
    
    def __init__(self, success: bool = True, data: Any = None, 
                 events: Optional[pd.DataFrame] = None, 
                 errors: Optional[list] = None,
                 warnings: Optional[list] = None,
                 metadata: Optional[dict] = None):
        self.success = success
        self.data = data
        self.events = events if events is not None else pd.DataFrame()
        self.errors = errors if errors is not None else []
        self.warnings = warnings if warnings is not None else []
        self.metadata = metadata if metadata is not None else {}
    
    def add_error(self, error: str):
        """Add an error to the result."""
        self.errors.append(error)
        self.success = False
    
    def add_warning(self, warning: str):
        """Add a warning to the result."""
        self.warnings.append(warning)
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to the result."""
        self.metadata[key] = value
    
    def merge_events(self, new_events: pd.DataFrame):
        """Merge new events into the result events."""
        if not new_events.empty:
            if self.events.empty:
                self.events = new_events.copy()
            else:
                self.events = pd.concat([self.events, new_events], ignore_index=True)