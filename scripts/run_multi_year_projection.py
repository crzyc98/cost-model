# scripts/run_multi_year_projection.py
# This script is now a thin wrapper for the projections CLI module.

import os
import sys

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cost_model.projections.cli import main

if __name__ == "__main__":
    # To run the projection, execute this script, or use:
    # python -m cost_model.projections.cli --config <config_path> --census <census_path>
    main()
