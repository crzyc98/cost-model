# utils/simulation_utils.py
"""
High-level simulation entry points.

Re-exports the main stochastic utilities for:
  - sampling terminations
  - sampling new-hire compensation
  - applying ML-driven turnover
  - applying compensation bumps
"""

import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401

from cost_model.utils.compensation.bump import apply_comp_increase  # noqa: F401
from cost_model.utils.ml.turnover import apply_ml_turnover  # noqa: F401
from cost_model.utils.sampling.new_hires import sample_new_hire_compensation

# Orchestrator re-exporting domain helpers
from cost_model.utils.sampling.terminations import sample_terminations  # noqa: F401

__all__ = [
    "sample_terminations",
    "sample_new_hire_compensation",
    "apply_ml_turnover",
    "apply_comp_increase",
    "term",
    "bump",
]

# Convenience aliases
term = sample_terminations
bump = apply_comp_increase
