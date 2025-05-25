#!/usr/bin/env python3
"""
Test script to verify the logging configuration.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from logging_config import test_logging, setup_logging, get_logger

def main():
    print("Testing logging configuration...")
    
    # Test the logging configuration
    log_dir = project_root / "output_dev" / "projection_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the test function
    test_logging()
    
    # Test module-level logging
    logger = get_logger(__name__)
    logger.info("Test log from test_logging.py")
    
    print("\nLogging test complete. Check the log files in:")
    print(f"  - {log_dir / 'projection_events.log'}")
    print(f"  - {log_dir / 'warnings_errors.log'}")
    print(f"  - {log_dir / 'debug_detail.log'}")

if __name__ == "__main__":
    main()
