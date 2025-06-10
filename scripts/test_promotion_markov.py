#!/usr/bin/env python3
"""
Test script to diagnose NaN values in EMP_LEVEL during apply_promotion_markov.
This script creates sample data and exercises just the promotion logic.
"""
import logging
import os
import sys

import numpy as np
import pandas as pd

# Set up logging immediately to avoid referencing before definition
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Add the root directory to the Python path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import our target functions and schema constants
from cost_model.state.job_levels.sampling import apply_promotion_markov

# Import schema constants with fallback values if import fails
try:
    from cost_model.state.schema import EMP_EXITED, EMP_HIRE_DATE, EMP_ID, EMP_LEVEL

    logger.info(
        f"Successfully imported schema constants: {EMP_LEVEL=}, {EMP_ID=}, {EMP_HIRE_DATE=}, {EMP_EXITED=}"
    )
except ImportError:
    # Define fallback values if import fails
    logger.warning("Could not import constants from schema.py, using fallbacks")
    EMP_LEVEL = "employee_level"
    EMP_ID = "employee_id"
    EMP_HIRE_DATE = "employee_hire_date"
    EMP_EXITED = "employee_exited"
    logger.info(
        f"Using fallback schema constants: {EMP_LEVEL=}, {EMP_ID=}, {EMP_HIRE_DATE=}, {EMP_EXITED=}"
    )

# Logger is already set up at the top of the file


def create_test_data():
    """Create a test DataFrame with employee data."""
    # Create a simple DataFrame with employee data
    today = pd.Timestamp.today()
    df = pd.DataFrame(
        {
            EMP_ID: [f"EMP-{i:04d}" for i in range(1, 101)],
            EMP_LEVEL: [np.random.randint(1, 8) for _ in range(100)],
            EMP_HIRE_DATE: [
                today - pd.Timedelta(days=np.random.randint(30, 1000)) for _ in range(100)
            ],
            EMP_EXITED: [False] * 100,
            "exited": [False] * 100,  # Duplicate column that exists in the real data
        }
    )

    # Intentionally introduce some missing levels to test our handling
    nan_indices = np.random.choice(range(100), 5, replace=False)
    df.loc[nan_indices, EMP_LEVEL] = np.nan

    # Mark some employees as exited
    exit_indices = np.random.choice(range(100), 10, replace=False)
    df.loc[exit_indices, EMP_EXITED] = True
    df.loc[exit_indices, "exited"] = True

    return df


def create_test_promotion_matrix():
    """Create a test promotion matrix for 7 levels."""
    # Create a simple diagonal-heavy matrix
    # Most employees stay at the same level, with a small chance of promotion
    matrix = np.eye(8, 8) * 0.9  # 90% stay at same level

    # Add 10% chance of promotion (except for highest level)
    for i in range(7):
        matrix[i, i + 1] = 0.1

    # Fix the highest level (level 8) to ensure it sums to 1.0
    # Employees at level 8 have 100% chance to stay at level 8
    matrix[7, 7] = 1.0

    # Convert to pandas DataFrame
    df_matrix = pd.DataFrame(matrix)

    # Set the indices and columns to be levels (1-8)
    df_matrix.index = range(1, 9)
    df_matrix.columns = range(1, 9)

    # Verify all rows sum to 1.0
    row_sums = df_matrix.sum(axis=1)
    for i, sum_val in row_sums.items():
        if not np.isclose(sum_val, 1.0):
            raise ValueError(f"Row {i} sum is {sum_val}, not 1.0")

    return df_matrix


def test_promotion_markov():
    """Run a test of the apply_promotion_markov function with diagnostic logging."""
    # Create test data
    logger.info("Creating test data with 100 employees (5 with NaN levels, 10 marked as exited)")
    df = create_test_data()

    # Create a test promotion matrix
    logger.info("Creating test promotion matrix")
    matrix = create_test_promotion_matrix()

    # Log initial state
    logger.info(f"Initial data: {len(df)} rows")
    logger.info(f"NaN count in {EMP_LEVEL} before: {pd.isna(df[EMP_LEVEL]).sum()}")

    # Apply the promotion markov function
    logger.info("Calling apply_promotion_markov...")
    result_df = apply_promotion_markov(
        df,
        level_col=EMP_LEVEL,
        rng=np.random.default_rng(seed=42),
        simulation_year=2025,
        matrix=matrix,
    )

    # Check the results
    logger.info(f"Result data: {len(result_df)} rows")
    logger.info(f"NaN count in {EMP_LEVEL} after: {pd.isna(result_df[EMP_LEVEL]).sum()}")

    # Find any employees where level changed from NaN to a value
    fixed_indices = df.index[pd.isna(df[EMP_LEVEL]) & ~pd.isna(result_df[EMP_LEVEL])].tolist()
    logger.info(f"Employees with fixed NaN levels: {len(fixed_indices)}")

    # Find any employees where level changed from a value to NaN
    broken_indices = df.index[~pd.isna(df[EMP_LEVEL]) & pd.isna(result_df[EMP_LEVEL])].tolist()
    if broken_indices:
        logger.warning(f"Employees with NEW NaN levels: {len(broken_indices)}")
        for idx in broken_indices[:5]:  # Show details for up to 5
            emp = df.loc[idx]
            logger.warning(f"  Employee {emp[EMP_ID]} had level {emp[EMP_LEVEL]} before processing")

    return result_df


if __name__ == "__main__":
    logger.info("Starting test_promotion_markov diagnostic script")
    test_df = test_promotion_markov()
    logger.info("Test completed")
