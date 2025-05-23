#!/usr/bin/env python3
"""
Standalone test script for snapshot validation.
This version doesn't depend on the project structure.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Schema constants
EMP_ID = 'employee_id'
EMP_HIRE_DATE = 'hire_date'
EMP_BIRTH_DATE = 'birth_date'
EMP_LEVEL = 'job_level'
EMP_GROSS_COMP = 'gross_compensation'
EMP_TERM_DATE = 'termination_date'
EMP_LEVEL_SOURCE = 'level_source'
EMP_TENURE_BAND = 'tenure_band'
EMP_ACTIVE = 'is_active'
EMP_EXITED = 'has_exited'
EMP_TENURE = 'tenure_years'

def validate_eoy_snapshot(final_snap: pd.DataFrame, target: Optional[int] = None) -> None:
    """
    Validates the end-of-year snapshot for consistency and correctness.
    
    Args:
        final_snap: The end-of-year snapshot DataFrame to validate
        target: Expected number of active employees. If None, skips headcount validation.
        
    Raises:
        ValueError: If any validation checks fail
    """
    logger.info("Validating end-of-year snapshot...")
    
    # Check for duplicate employee IDs
    duplicate_ids = final_snap[EMP_ID].duplicated()
    if duplicate_ids.any():
        duplicate_count = duplicate_ids.sum()
        duplicates = final_snap[final_snap[EMP_ID].duplicated(keep=False)].sort_values(EMP_ID)
        logger.error(f"Found {duplicate_count} duplicate EMP_IDs in snapshot")
        logger.error(f"Duplicate IDs:\n{duplicates[[EMP_ID, EMP_ACTIVE, EMP_TERM_DATE]].head(10) if not duplicates.empty else 'None'}")
        if len(duplicates) > 10:
            logger.error(f"... and {len(duplicates) - 10} more")
        raise ValueError(f"Found {duplicate_count} duplicate EMP_IDs in snapshot")
    
    # Check active employee count if target is provided
    if target is not None:
        active_count = final_snap[EMP_ACTIVE].sum()
        if active_count != target:
            logger.error(
                f"Active employee count mismatch. Expected: {target}, Actual: {active_count}"
            )
            raise ValueError(
                f"EOY headcount {active_count} does not match target {target}"
            )
    
    # Check for invalid active/terminated states
    invalid_active = final_snap[final_snap[EMP_ACTIVE] & ~pd.isna(final_snap[EMP_TERM_DATE])]
    if not invalid_active.empty:
        logger.error(
            f"Found {len(invalid_active)} employees marked as both active and terminated"
        )
        raise ValueError(
            f"Found {len(invalid_active)} employees with both active=True and a termination date"
        )
    
    logger.info("✅ End-of-year snapshot validation passed")

def generate_sample_snapshot(num_employees: int = 100, year: int = 2025) -> pd.DataFrame:
    """Generate a sample snapshot DataFrame for testing."""
    np.random.seed(42)  # For reproducible results
    
    # Base date for calculations
    base_date = datetime(year, 1, 1)
    
    # Generate employee data
    data = {
        EMP_ID: [f"EMP{i:04d}" for i in range(1, num_employees + 1)],
        EMP_HIRE_DATE: [
            base_date - timedelta(days=np.random.randint(365, 365*10)) 
            for _ in range(num_employees)
        ],
        EMP_BIRTH_DATE: [
            base_date - timedelta(days=np.random.randint(25*365, 65*365)) 
            for _ in range(num_employees)
        ],
        EMP_GROSS_COMP: np.random.lognormal(10.5, 0.3, num_employees).round(2),
        EMP_LEVEL: np.random.randint(1, 8, num_employees),
        EMP_LEVEL_SOURCE: ['sample'] * num_employees,
        EMP_ACTIVE: [True] * (num_employees - 10) + [False] * 10,
        EMP_TERM_DATE: [pd.NaT] * (num_employees - 10) + \
                      [base_date - timedelta(days=np.random.randint(1, 365)) 
                       for _ in range(10)],
        EMP_EXITED: [False] * (num_employees - 10) + [True] * 10,
        EMP_TENURE: np.random.uniform(0, 20, num_employees).round(2),
    }
    
    df = pd.DataFrame(data)
    
    # Set tenure bands based on tenure
    conditions = [
        (df[EMP_TENURE] < 1),
        (df[EMP_TENURE] < 3),
        (df[EMP_TENURE] < 5),
        (df[EMP_TENURE] < 10),
        (df[EMP_TENURE] >= 10)
    ]
    choices = ['<1yr', '1-2yrs', '3-4yrs', '5-9yrs', '10+ yrs']
    df[EMP_TENURE_BAND] = np.select(conditions, choices, default='<1yr')
    
    return df

def run_validation_tests():
    """Run validation tests with sample data."""
    logger.info("=== Starting Snapshot Validation Tests ===\n")
    
    # Test 1: Valid snapshot
    logger.info("1. Testing valid snapshot...")
    snapshot = generate_sample_snapshot(num_employees=100)
    try:
        validate_eoy_snapshot(snapshot, target=90)  # 90 active employees expected
        logger.info("✅ Test passed: Valid snapshot\n")
    except ValueError as e:
        logger.error(f"❌ Test failed: {e}\n")
    
    # Test 2: Duplicate employee IDs
    logger.info("2. Testing duplicate employee IDs...")
    snapshot_dupe = snapshot.copy()
    snapshot_dupe.loc[0, EMP_ID] = snapshot_dupe.loc[1, EMP_ID]  # Create duplicate ID
    try:
        validate_eoy_snapshot(snapshot_dupe, target=90)
        logger.error("❌ Test failed: Expected validation to fail with duplicate IDs\n")
    except ValueError as e:
        logger.info(f"✅ Test passed: Correctly caught duplicate IDs: {e}\n")
    
    # Test 3: Invalid active/terminated state
    logger.info("3. Testing invalid active/terminated state...")
    snapshot_invalid = snapshot.copy()
    # Make an active employee also have a termination date
    active_emp = snapshot_invalid[snapshot_invalid[EMP_ACTIVE]].iloc[0][EMP_ID]
    snapshot_invalid.loc[snapshot_invalid[EMP_ID] == active_emp, EMP_TERM_DATE] = datetime(2025, 6, 15)
    try:
        validate_eoy_snapshot(snapshot_invalid, target=90)
        logger.error("❌ Test failed: Expected validation to fail with invalid state\n")
    except ValueError as e:
        logger.info(f"✅ Test passed: Correctly caught invalid state: {e}\n")
    
    logger.info("=== All tests completed ===")

if __name__ == "__main__":
    run_validation_tests()
