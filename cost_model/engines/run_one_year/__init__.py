"""
Run One Year package - orchestrates simulation for a single year.

This package breaks down the monolithic run_one_year.py into smaller,
focused modules with single responsibilities.
"""

from cost_model.engines.run_one_year_engine import run_one_year

__all__ = ["run_one_year"]
