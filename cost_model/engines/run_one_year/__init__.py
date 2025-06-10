"""
Run One Year Package

This package contains the implementation of the run_one_year simulation engine,
which orchestrates a single year of workforce simulation.

The orchestrator has been refactored into a modular package structure for better
maintainability and single responsibility principles.
"""

print("Using run_one_year package implementation with modular orchestrator")

from .orchestrator import run_one_year

__all__ = ["run_one_year"]
