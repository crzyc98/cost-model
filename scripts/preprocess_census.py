#!/usr/bin/env python3
"""
Script to preprocess census data and save as parquet.
"""

import logging
import os
import sys
from pathlib import Path

from cost_model.data.readers import read_census_data

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

logger = logging.getLogger(__name__)


def preprocess_and_save(
    input_path: Path = Path("data/dev_tiny/census_2024.csv"),
    output_path: Path = Path("data/census_preprocessed.parquet"),
):
    """
    Read census data, preprocess it, and save as parquet.

    Args:
        input_path: Path to the input census file (CSV)
        output_path: Path where to save the preprocessed parquet file
    """
    try:
        # Read and preprocess the data
        logger.info(f"Reading census data from {input_path}")
        df = read_census_data(input_path)
        if df is None:
            raise ValueError(f"Failed to read census data from {input_path}")

        logger.info(f"Successfully loaded {len(df)} records")

        # Save as parquet
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving preprocessed data to {output_path}")
        df.to_parquet(output_path, index=False)

        logger.info("Preprocessing complete!")

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s",
    )

    preprocess_and_save()
