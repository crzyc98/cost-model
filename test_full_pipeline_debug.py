#!/usr/bin/env python3
"""
Test script to debug the full simulation pipeline and see where contribution values get corrupted.
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_full_pipeline():
    """Test the full simulation pipeline to see where contribution values get corrupted."""
    try:
        # Import required modules
        from cost_model.projections.cli import run_projection
        from cost_model.config.loaders import load_config_to_namespace
        from cost_model.state.schema import EMP_CONTR, EMPLOYER_MATCH_CONTRIB, EMPLOYER_CORE_CONTRIB

        logger.info("Testing full simulation pipeline...")

        # Load config
        config_path = "config/dev_tiny.yaml"
        logger.info(f"Loading config from: {config_path}")
        config_ns = load_config_to_namespace(config_path)

        # Set up minimal args
        class Args:
            def __init__(self):
                self.config = config_path
                self.census = "data/dev_tiny/census_2024.csv"
                self.output_dir = None
                self.debug = True
                self.scenario_name = "test_scenario"

        args = Args()
        output_path = Path("output/debug_test")
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("Running projection...")

        # Run the projection
        run_projection(args, config_ns, output_path)

        logger.info("Projection completed successfully!")

        # Check the output files
        snapshot_files = list(output_path.glob("**/snapshot_*.parquet"))
        if snapshot_files:
            logger.info(f"Found {len(snapshot_files)} snapshot files")

            # Load the first snapshot and check contribution columns
            snapshot_file = snapshot_files[0]
            logger.info(f"Examining snapshot file: {snapshot_file}")

            df = pd.read_parquet(snapshot_file)
            logger.info(f"Snapshot shape: {df.shape}")
            logger.info(f"Snapshot columns: {df.columns.tolist()}")

            # Check contribution columns
            contrib_cols = [EMP_CONTR, EMPLOYER_MATCH_CONTRIB, EMPLOYER_CORE_CONTRIB]
            for col in contrib_cols:
                if col in df.columns:
                    logger.info(f"Column '{col}' found:")
                    logger.info(f"  Sample values: {df[col].head().tolist()}")
                    logger.info(f"  Dtype: {df[col].dtype}")
                    logger.info(f"  Null count: {df[col].isnull().sum()}")
                    logger.info(f"  Non-zero count: {(df[col] != 0).sum()}")
                else:
                    logger.warning(f"Column '{col}' NOT FOUND")
        else:
            logger.warning("No snapshot files found in output")

        return True

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_pipeline()
    if success:
        logger.info("✅ Test completed successfully!")
    else:
        logger.error("❌ Test failed!")
        sys.exit(1)
