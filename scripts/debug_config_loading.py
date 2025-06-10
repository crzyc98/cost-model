#!/usr/bin/env python3
"""
Debug script to test configuration loading and flattening.
"""

import logging
from pathlib import Path

from cost_model.config.loaders import load_config_to_namespace
from cost_model.config.models import MainConfig

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_config_loading():
    """Test configuration loading and flattening."""
    config_path = Path("campaign_6_results/best_config.yaml")

    logger.info("=== Testing Configuration Loading ===")

    # 1. Load as namespace (with flattening)
    logger.info("1. Loading config as namespace...")
    namespace_config = load_config_to_namespace(config_path)

    # Check if attrition section exists
    gp = namespace_config.global_parameters
    logger.info(
        f"Global parameters attributes: {[attr for attr in dir(gp) if not attr.startswith('_')]}"
    )

    # Check for attrition section
    if hasattr(gp, "attrition"):
        logger.info(f"Found attrition section: {gp.attrition}")
    else:
        logger.info("No attrition section found (flattening worked)")

    # Check for flattened new_hire_termination_rate
    if hasattr(gp, "new_hire_termination_rate"):
        logger.info(f"Found flattened new_hire_termination_rate: {gp.new_hire_termination_rate}")
    else:
        logger.warning("new_hire_termination_rate NOT found at root level")

    # 2. Convert to Pydantic model
    logger.info("\n2. Converting to Pydantic model...")

    def namespace_to_dict(obj):
        if hasattr(obj, "__dict__"):
            return {k: namespace_to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [namespace_to_dict(item) for item in obj]
        else:
            return obj

    config_dict = namespace_to_dict(namespace_config)
    main_config = MainConfig(**config_dict)

    # Check the Pydantic model
    logger.info(
        f"Pydantic model new_hire_termination_rate: {main_config.global_parameters.new_hire_termination_rate}"
    )

    # 3. Test dynamic hazard table generation
    logger.info("\n3. Testing dynamic hazard table generation...")
    from cost_model.projections.dynamic_hazard import build_dynamic_hazard_table

    global_params = main_config.global_parameters
    simulation_years = [2025, 2026, 2027]
    job_levels = [1, 2, 3, 4]
    tenure_bands = ["<1", "1-3", "3-5", "5-10", "10-15", "15+"]

    hazard_table = build_dynamic_hazard_table(
        global_params=global_params,
        simulation_years=simulation_years,
        job_levels=job_levels,
        tenure_bands=tenure_bands,
    )

    logger.info(f"Dynamic hazard table columns: {hazard_table.columns.tolist()}")

    from cost_model.state.schema import NEW_HIRE_TERMINATION_RATE

    if NEW_HIRE_TERMINATION_RATE in hazard_table.columns:
        rate_value = hazard_table[NEW_HIRE_TERMINATION_RATE].iloc[0]
        logger.info(f"✓ Found {NEW_HIRE_TERMINATION_RATE} column with value: {rate_value}")
    else:
        logger.error(f"✗ {NEW_HIRE_TERMINATION_RATE} column missing from hazard table")

    return hazard_table


def test_hazard_slice_extraction():
    """Test the hazard slice extraction process."""
    logger.info("\n=== Testing Hazard Slice Extraction ===")

    # 1. Load config and build hazard table
    config_path = Path("campaign_6_results/best_config.yaml")
    namespace_config = load_config_to_namespace(config_path)

    def namespace_to_dict(obj):
        if hasattr(obj, "__dict__"):
            return {k: namespace_to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [namespace_to_dict(item) for item in obj]
        else:
            return obj

    config_dict = namespace_to_dict(namespace_config)
    main_config = MainConfig(**config_dict)

    from cost_model.projections.dynamic_hazard import build_dynamic_hazard_table

    global_params = main_config.global_parameters
    simulation_years = [2025, 2026, 2027]
    job_levels = [1, 2, 3, 4]
    tenure_bands = ["<1", "1-3", "3-5", "5-10", "10-15", "15+"]

    hazard_table = build_dynamic_hazard_table(
        global_params=global_params,
        simulation_years=simulation_years,
        job_levels=job_levels,
        tenure_bands=tenure_bands,
    )

    logger.info(f"Full hazard table columns: {hazard_table.columns.tolist()}")
    logger.info(f"Full hazard table shape: {hazard_table.shape}")

    # 2. Test hazard slice extraction
    from cost_model.engines.run_one_year.validation import validate_and_extract_hazard_slice
    from cost_model.state.schema import NEW_HIRE_TERMINATION_RATE

    year = 2025
    hazard_slice = validate_and_extract_hazard_slice(hazard_table, year)

    logger.info(f"Hazard slice columns: {hazard_slice.columns.tolist()}")
    logger.info(f"Hazard slice shape: {hazard_slice.shape}")

    if NEW_HIRE_TERMINATION_RATE in hazard_slice.columns:
        rate_value = hazard_slice[NEW_HIRE_TERMINATION_RATE].iloc[0]
        logger.info(f"✓ Hazard slice contains {NEW_HIRE_TERMINATION_RATE}: {rate_value}")
    else:
        logger.error(f"✗ Hazard slice missing {NEW_HIRE_TERMINATION_RATE}")
        logger.error(f"Available columns: {hazard_slice.columns.tolist()}")

    return hazard_slice


if __name__ == "__main__":
    test_config_loading()
    test_hazard_slice_extraction()
