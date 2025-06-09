"""
Logging utilities for snapshot processing.

Provides structured logging, timing decorators, progress indicators,
and performance monitoring for the snapshot refactoring system.
"""

import logging
import time
import functools
import os
from typing import Any, Callable, Dict, Optional, Union
from contextlib import contextmanager
from datetime import datetime

# Optional dependency for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Configure structured logging format
STRUCTURED_LOG_FORMAT = (
    "%(asctime)s | %(name)s | %(levelname)-8s | "
    "%(module)s.%(funcName)s:%(lineno)d | %(message)s"
)

class SnapshotLogger:
    """Enhanced logger for snapshot processing with timing and performance tracking."""
    
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Add structured formatter if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(STRUCTURED_LOG_FORMAT)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        self._log_with_context(logging.INFO, message, kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data."""
        self._log_with_context(logging.DEBUG, message, kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data."""
        self._log_with_context(logging.WARNING, message, kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional structured data."""
        self._log_with_context(logging.ERROR, message, kwargs)
    
    def _log_with_context(self, level: int, message: str, context: Dict[str, Any]):
        """Log message with structured context data."""
        if context:
            context_str = " | ".join([f"{k}={v}" for k, v in context.items()])
            full_message = f"{message} | {context_str}"
        else:
            full_message = message
        
        self.logger.log(level, full_message)


def get_snapshot_logger(name: str) -> SnapshotLogger:
    """Get a snapshot logger instance."""
    return SnapshotLogger(name)


def timing_decorator(logger: Optional[Union[logging.Logger, SnapshotLogger]] = None):
    """
    Decorator to measure and log function execution time.
    
    Args:
        logger: Optional logger instance. If None, uses function's module logger.
        
    Returns:
        Decorated function with timing logging
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger
            if logger is None:
                func_logger = get_snapshot_logger(f"{func.__module__}.{func.__name__}")
            else:
                func_logger = logger
            
            # Record start time and memory
            start_time = time.time()
            start_memory = _get_memory_usage()
            
            func_logger.debug(
                f"Starting {func.__name__}",
                start_time=datetime.now().isoformat(),
                memory_mb=start_memory
            )
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Calculate metrics
                end_time = time.time()
                end_memory = _get_memory_usage()
                duration = end_time - start_time
                memory_delta = end_memory - start_memory
                
                # Log completion
                func_logger.info(
                    f"Completed {func.__name__}",
                    duration_seconds=f"{duration:.3f}",
                    memory_delta_mb=f"{memory_delta:+.1f}",
                    final_memory_mb=end_memory
                )
                
                return result
                
            except Exception as e:
                # Log error with timing
                duration = time.time() - start_time
                func_logger.error(
                    f"Failed {func.__name__}",
                    duration_seconds=f"{duration:.3f}",
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise
        
        return wrapper
    return decorator


@contextmanager
def progress_context(logger: SnapshotLogger, operation: str, total_items: Optional[int] = None):
    """
    Context manager for tracking progress of long-running operations.
    
    Args:
        logger: Logger instance
        operation: Description of the operation
        total_items: Optional total number of items being processed
        
    Yields:
        Progress tracker function
    """
    start_time = time.time()
    start_memory = _get_memory_usage()
    
    logger.info(
        f"Starting {operation}",
        total_items=total_items,
        start_memory_mb=start_memory
    )
    
    def progress_tracker(current_item: int, message: str = ""):
        """Track progress of current item."""
        elapsed = time.time() - start_time
        current_memory = _get_memory_usage()
        
        if total_items and total_items > 0:
            percentage = (current_item / total_items) * 100
            estimated_total = elapsed * (total_items / current_item) if current_item > 0 else 0
            remaining = estimated_total - elapsed
            
            logger.debug(
                f"{operation} progress",
                current_item=current_item,
                total_items=total_items,
                percentage=f"{percentage:.1f}%",
                elapsed_seconds=f"{elapsed:.1f}",
                estimated_remaining_seconds=f"{remaining:.1f}",
                memory_mb=current_memory,
                message=message
            )
        else:
            logger.debug(
                f"{operation} progress",
                current_item=current_item,
                elapsed_seconds=f"{elapsed:.1f}",
                memory_mb=current_memory,
                message=message
            )
    
    try:
        yield progress_tracker
        
        # Log completion
        total_time = time.time() - start_time
        final_memory = _get_memory_usage()
        memory_delta = final_memory - start_memory
        
        logger.info(
            f"Completed {operation}",
            total_duration_seconds=f"{total_time:.3f}",
            memory_delta_mb=f"{memory_delta:+.1f}",
            final_memory_mb=final_memory
        )
        
    except Exception as e:
        # Log error
        total_time = time.time() - start_time
        logger.error(
            f"Failed {operation}",
            duration_seconds=f"{total_time:.3f}",
            error=str(e),
            error_type=type(e).__name__
        )
        raise


def log_dataframe_info(logger: SnapshotLogger, df, name: str, detailed: bool = False):
    """
    Log detailed information about a DataFrame.
    
    Args:
        logger: Logger instance
        df: DataFrame to analyze
        name: Name/description of the DataFrame
        detailed: Whether to include detailed column information
    """
    if df is None or df.empty:
        logger.warning(f"{name} is empty or None")
        return
    
    # Basic info
    memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
    
    logger.info(
        f"{name} analysis",
        rows=len(df),
        columns=len(df.columns),
        memory_mb=f"{memory_usage:.2f}",
        dtypes_summary=df.dtypes.value_counts().to_dict()
    )
    
    if detailed:
        # Null value analysis
        null_counts = df.isnull().sum()
        null_info = {col: count for col, count in null_counts.items() if count > 0}
        
        if null_info:
            logger.debug(f"{name} null values", null_counts=null_info)
        
        # Data type info
        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            
            logger.debug(
                f"{name} column: {col}",
                dtype=str(dtype),
                null_count=null_count,
                unique_values=unique_count,
                null_percentage=f"{(null_count/len(df)*100):.1f}%"
            )


def log_performance_metrics(logger: SnapshotLogger, operation: str, metrics: Dict[str, Any]):
    """
    Log performance metrics for an operation.
    
    Args:
        logger: Logger instance
        operation: Name of the operation
        metrics: Dictionary of metrics to log
    """
    logger.info(f"{operation} performance metrics", **metrics)


def _get_memory_usage() -> float:
    """Get current memory usage in MB."""
    if not PSUTIL_AVAILABLE:
        return 0.0
    
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    except Exception:
        return 0.0


class PerformanceMonitor:
    """Monitor performance metrics during snapshot processing."""
    
    def __init__(self, logger: SnapshotLogger):
        self.logger = logger
        self.metrics = {}
        self.start_time = None
        self.start_memory = None
    
    def start_monitoring(self, operation: str):
        """Start monitoring an operation."""
        self.operation = operation
        self.start_time = time.time()
        self.start_memory = _get_memory_usage()
        self.metrics = {
            'operation': operation,
            'start_time': datetime.now().isoformat(),
            'start_memory_mb': self.start_memory
        }
        
        self.logger.debug(f"Started monitoring {operation}")
    
    def add_checkpoint(self, name: str, **additional_metrics):
        """Add a performance checkpoint."""
        if self.start_time is None:
            return
        
        current_time = time.time()
        current_memory = _get_memory_usage()
        
        checkpoint_metrics = {
            'elapsed_seconds': current_time - self.start_time,
            'memory_mb': current_memory,
            'memory_delta_mb': current_memory - self.start_memory,
            **additional_metrics
        }
        
        self.metrics[f'checkpoint_{name}'] = checkpoint_metrics
        
        self.logger.debug(
            f"Checkpoint {name} for {self.operation}",
            **checkpoint_metrics
        )
    
    def finish_monitoring(self, **final_metrics):
        """Finish monitoring and log final metrics."""
        if self.start_time is None:
            return
        
        end_time = time.time()
        end_memory = _get_memory_usage()
        
        final_data = {
            'total_duration_seconds': end_time - self.start_time,
            'final_memory_mb': end_memory,
            'memory_delta_mb': end_memory - self.start_memory,
            'end_time': datetime.now().isoformat(),
            **final_metrics
        }
        
        self.metrics['final'] = final_data
        
        self.logger.info(f"Performance summary for {self.operation}", **final_data)
        
        return self.metrics