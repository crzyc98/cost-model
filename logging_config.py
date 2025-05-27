"""
Structured logging configuration for the cost-model project.

This module provides a centralized way to configure logging across the application
with different log levels and output files for different concerns.
"""

import logging
import logging.handlers
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Set
from datetime import datetime

# Define logger names for different concerns
PROJECTION_LOGGER = "cost_model.projection"
PERFORMANCE_LOGGER = "cost_model.performance"
ERROR_LOGGER = "cost_model.errors"
DEBUG_LOGGER = "cost_model.debug"

# Standard log format with module name and line number
LOG_FORMAT = "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Track if logging is already configured and log files
_LOGGING_CONFIGURED = False
_log_files_created: Set[Path] = set()
_combined_log_file: Optional[Path] = None


def clear_logs(log_dir: Path) -> None:
    """
    Clear all log files in the specified directory.
    
    Args:
        log_dir: Directory containing log files to clear
    """
    log_files = [
        log_dir / "projection_events.log",
        log_dir / "performance_metrics.log",
        log_dir / "warnings_errors.log",
        log_dir / "debug_detail.log",
        log_dir / "combined.log"
    ]
    
    for log_file in log_files:
        if log_file.exists():
            try:
                log_file.unlink()
            except Exception as e:
                print(f"Warning: Could not delete {log_file}: {e}")


def setup_logging(log_dir: Path, debug: bool = False, clear_existing: bool = True) -> None:
    """
    Configure structured logging for the application.

    Creates separate log files for different concerns:
    - projection_events.log: Main projection workflow events (INFO+)
    - performance_metrics.log: Performance-related metrics (INFO+)
    - warnings_errors.log: Warnings and errors (WARNING+)
    - debug_detail.log: Detailed debug information (DEBUG, only if debug=True)
    - combined.log: Combined log of all messages (INFO+)

    Args:
        log_dir: Directory where log files will be stored
        debug: If True, enables debug logging and creates debug_detail.log
        clear_existing: If True, clears existing log files before starting
    """
    global _LOGGING_CONFIGURED, _combined_log_file

    if _LOGGING_CONFIGURED:
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    if clear_existing:
        clear_logs(log_dir)

    _combined_log_file = log_dir / "combined.log"
    _log_files_created.add(_combined_log_file)

    # Remove all handlers from the root logger before setup
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Formatters
    file_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    console_formatter = logging.Formatter("%(levelname)-8s %(message)s")

    # Console handler (for warnings and above)
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(console_formatter)
    root_logger.addHandler(console)

    # Combined log (INFO+)
    combined_handler = logging.handlers.RotatingFileHandler(
        filename=_combined_log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8',
        mode='a'
    )
    combined_handler.setLevel(logging.INFO)
    combined_handler.setFormatter(file_formatter)
    root_logger.addHandler(combined_handler)

    # Warnings and errors log (WARNING+)
    warnings_errors_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "warnings_errors.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8',
        mode='a'
    )
    warnings_errors_handler.setLevel(logging.WARNING)
    warnings_errors_handler.setFormatter(file_formatter)
    root_logger.addHandler(warnings_errors_handler)

    # projection_events.log (INFO+) for projection logger
    proj_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "projection_events.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8',
        mode='a'
    )
    proj_handler.setLevel(logging.INFO)
    proj_handler.setFormatter(file_formatter)
    proj_logger = logging.getLogger(PROJECTION_LOGGER)
    for h in proj_logger.handlers[:]:
        proj_logger.removeHandler(h)
    proj_logger.setLevel(logging.INFO)
    proj_logger.addHandler(proj_handler)
    proj_logger.propagate = True  # Allow to bubble up to root

    # performance_metrics.log (INFO+) for performance logger
    perf_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "performance_metrics.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8',
        mode='a'
    )
    perf_handler.setLevel(logging.INFO)
    perf_handler.setFormatter(file_formatter)
    perf_logger = logging.getLogger(PERFORMANCE_LOGGER)
    for h in perf_logger.handlers[:]:
        perf_logger.removeHandler(h)
    perf_logger.setLevel(logging.INFO)
    perf_logger.addHandler(perf_handler)
    perf_logger.propagate = True

    if debug:
        debug_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / "debug_detail.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding='utf-8',
            mode='a'
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(file_formatter)
        debug_logger = logging.getLogger(DEBUG_LOGGER)
        for h in debug_logger.handlers[:]:
            debug_logger.removeHandler(h)
        debug_logger.setLevel(logging.DEBUG)
        debug_logger.addHandler(debug_handler)
        debug_logger.propagate = True

    _LOGGING_CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a properly configured logger instance.

    Args:
        name: Logger name (e.g., __name__). If None, returns root logger.

    Returns:
        Configured logger instance
    """
    if not _LOGGING_CONFIGURED:
        setup_logging(Path("output_dev/projection_logs"), debug=False)
    return logging.getLogger(name)


def test_logging():
    """Test function to verify logging configuration."""
    # Setup logging
    log_dir = Path("output_dev/projection_logs")
    
    # Clear existing logs and set up fresh logging
    clear_logs(log_dir)
    setup_logging(log_dir, debug=True, clear_existing=True)
    
    # Get loggers
    logger = get_logger(__name__)
    proj_logger = get_logger(PROJECTION_LOGGER)
    perf_logger = get_logger(PERFORMANCE_LOGGER)
    err_logger = get_logger(ERROR_LOGGER)
    debug_logger = get_logger(DEBUG_LOGGER)
    
    # Test log messages
    logger.info("Root logger test - INFO")
    logger.warning("Root logger test - WARNING")
    logger.error("Root logger test - ERROR")
    logger.debug("Root logger test - DEBUG")
    
    proj_logger.info("Projection logger test - INFO")
    perf_logger.info("Performance logger test - INFO")
    err_logger.warning("Error logger test - WARNING")
    debug_logger.debug("Debug logger test - DEBUG")
    
    # Test exception logging
    try:
        1 / 0
    except Exception as e:
        logger.exception("Test exception logging")
    
    # Verify combined log was created
    combined_log = log_dir / "combined.log"
    if combined_log.exists():
        print(f"✓ Combined log created: {combined_log.absolute()}")
    
    print(f"\nTest logs written to: {log_dir.absolute()}")
    print("\nLog files created:")
    for log_file in log_dir.glob("*.log"):
        print(f"- {log_file.name}")
    
    # Display first few lines of combined log
    if combined_log.exists():
        print("\nFirst few lines of combined.log:")
        with open(combined_log, 'r') as f:
            for i, line in enumerate(f):
                if i < 5:  # Show first 5 lines
                    print(f"  {line.strip()}")
                else:
                    print("  ...")
                    break
