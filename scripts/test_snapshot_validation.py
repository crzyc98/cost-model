#!/usr/bin/env python3
"""
Test script for snapshot validation.

This script demonstrates how to use the validate_eoy_snapshot function with sample data.
"""
import logging
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cost_model.engines.run_one_year.validation import validate_eoy_snapshot
from cost_model.state.schema import (
    EMP_ACTIVE,
    EMP_BIRTH_DATE,
    EMP_EXITED,
    EMP_GROSS_COMP,
    EMP_HIRE_DATE,
    EMP_ID,
    EMP_LEVEL,
    EMP_LEVEL_SOURCE,
    EMP_TENURE,
    EMP_TENURE_BAND,
    EMP_TERM_DATE,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def generate_sample_snapshot(num_employees: int = 100, year: int = 2025) -> pd.DataFrame:
    """Generate a sample snapshot DataFrame for testing."""
    np.random.seed(42)  # For reproducible results

    # Base date for calculations
    base_date = datetime(year, 1, 1)

    # Generate employee data
    data = {
        EMP_ID: [f"EMP{i:04d}" for i in range(1, num_employees + 1)],
        EMP_HIRE_DATE: [
            base_date - timedelta(days=np.random.randint(365, 365 * 10))
            for _ in range(num_employees)
        ],
        EMP_BIRTH_DATE: [
            base_date - timedelta(days=np.random.randint(25 * 365, 65 * 365))
            for _ in range(num_employees)
        ],
        EMP_GROSS_COMP: np.random.lognormal(10.5, 0.3, num_employees).round(2),
        EMP_LEVEL: np.random.randint(1, 8, num_employees),
        EMP_LEVEL_SOURCE: ["sample"] * num_employees,
        EMP_ACTIVE: [True] * (num_employees - 10) + [False] * 10,
        EMP_TERM_DATE: [pd.NaT] * (num_employees - 10)
        + [base_date - timedelta(days=np.random.randint(1, 365)) for _ in range(10)],
        EMP_EXITED: [False] * (num_employees - 10) + [True] * 10,
        EMP_TENURE: np.random.uniform(0, 20, num_employees).round(2),
        EMP_TENURE_BAND: [""] * num_employees,  # Will be filled based on tenure
    }

    df = pd.DataFrame(data)

    # Set tenure bands based on tenure
    conditions = [
        (df[EMP_TENURE] < 1),
        (df[EMP_TENURE] < 3),
        (df[EMP_TENURE] < 5),
        (df[EMP_TENURE] < 10),
        (df[EMP_TENURE] >= 10),
    ]
    choices = ["<1yr", "1-2yrs", "3-4yrs", "5-9yrs", "10+ yrs"]
    df[EMP_TENURE_BAND] = np.select(conditions, choices, default="<1yr")

    return df


def test_validation():
    """Test the snapshot validation with sample data."""
    logger.info("Generating sample snapshot...")
    snapshot = generate_sample_snapshot(num_employees=100)

    # Test 1: Valid snapshot
    logger.info("\n--- Testing valid snapshot ---")
    try:
        validate_eoy_snapshot(snapshot, target=90)  # 90 active employees expected
        logger.info("✅ Validation passed with valid snapshot")
    except ValueError as e:
        logger.error(f"❌ Validation failed: {e}")

    # Test 2: Duplicate employee IDs
    logger.info("\n--- Testing duplicate employee IDs ---")
    snapshot_dupe = snapshot.copy()
    snapshot_dupe.loc[0, EMP_ID] = snapshot_dupe.loc[1, EMP_ID]  # Create duplicate ID
    try:
        validate_eoy_snapshot(snapshot_dupe, target=90)
        logger.error("❌ Expected validation to fail with duplicate IDs")
    except ValueError as e:
        logger.info(f"✅ Correctly caught duplicate IDs: {e}")

    # Test 3: Invalid active/terminated state
    logger.info("\n--- Testing invalid active/terminated state ---")
    snapshot_invalid = snapshot.copy()
    # Make an active employee also have a termination date
    active_emp = snapshot_invalid[snapshot_invalid[EMP_ACTIVE]].iloc[0][EMP_ID]
    snapshot_invalid.loc[snapshot_invalid[EMP_ID] == active_emp, EMP_TERM_DATE] = datetime(
        2025, 6, 15
    )
    try:
        validate_eoy_snapshot(snapshot_invalid, target=90)
        logger.error("❌ Expected validation to fail with invalid state")
    except ValueError as e:
        logger.info(f"✅ Correctly caught invalid state: {e}")


if __name__ == "__main__":
    logger.info("Starting snapshot validation test...")
    test_validation()
    logger.info("Test completed.")
