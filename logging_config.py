"""
Structured logging configuration for the cost-model project.

This module provides a centralized way to configure logging across the application
with different log levels and output files for different concerns.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any

# Define logger names for different concerns
PROJECTION_LOGGER = "cost_model.projection"
PERFORMANCE_LOGGER = "cost_model.performance"
ERROR_LOGGER = "cost_model.errors"
DEBUG_LOGGER = "cost_model.debug"

# Standard log format with module name and line number
LOG_FORMAT = "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Track if logging is already configured
_LOGGING_CONFIGURED = False


def setup_logging(log_dir: Path, debug: bool = False) -> None:
    """
    Configure structured logging for the application.

    Creates separate log files for different concerns:
    - projection_events.log: Main projection workflow events (INFO+)
    - performance_metrics.log: Performance-related metrics (INFO+)
    - warnings_errors.log: Warnings and errors (WARNING+)
    - debug_detail.log: Detailed debug information (DEBUG, only if debug=True)

    Args:
        log_dir: Directory where log files will be stored
        debug: If True, enables debug logging and creates debug_detail.log
    """
    global _LOGGING_CONFIGURED
    
    if _LOGGING_CONFIGURED:
        return
        
    log_dir.mkdir(parents=True, exist_ok=True)

    # Clear any existing handlers from root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Create formatters
    file_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    console_formatter = logging.Formatter("%(levelname)-8s %(message)s")

    # Configure console handler (only warnings and above by default)
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(console_formatter)
    root_logger.addHandler(console)

    # Configure file handlers for different log types
    handlers = {
        "projection_events": {
            "level": logging.INFO,
            "filename": log_dir / "projection_events.log",
            "formatter": file_formatter,
            "logger": PROJECTION_LOGGER,
        },
        "performance_metrics": {
            "level": logging.INFO,
            "filename": log_dir / "performance_metrics.log",
            "formatter": file_formatter,
            "logger": PERFORMANCE_LOGGER,
        },
        "warnings_errors": {
            "level": logging.WARNING,
            "filename": log_dir / "warnings_errors.log",
            "formatter": file_formatter,
            "logger": ERROR_LOGGER,
        },
        # Add root logger handler to capture all warnings and errors
        "root_warnings_errors": {
            "level": logging.WARNING,
            "filename": log_dir / "warnings_errors.log",
            "formatter": file_formatter,
            "logger": "",  # Empty string means root logger
        },
    }

    if debug:
        handlers["debug_detail"] = {
            "level": logging.DEBUG,
            "filename": log_dir / "debug_detail.log",
            "formatter": file_formatter,
            "logger": DEBUG_LOGGER,
        }

    # Create and configure all handlers
    for name, config in handlers.items():
        # Create file handler with rotation (10MB per file, keep 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=config["filename"],
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8',
            mode='a'
        )
        file_handler.setLevel(config["level"])
        file_handler.setFormatter(config["formatter"])
        
        # Get or create the logger and add the handler
        logger = logging.getLogger(config["logger"])
        logger.setLevel(config["level"])
        
        # Remove any existing handlers to avoid duplicates
        for h in logger.handlers[:]:
            logger.removeHandler(h)
            
        logger.addHandler(file_handler)
        logger.propagate = False  # Prevent propagation to root logger

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
    setup_logging(log_dir, debug=True)
    
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
    
    print(f"Test logs written to: {log_dir.absolute()}")
