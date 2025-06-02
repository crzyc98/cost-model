#!/usr/bin/env python3
"""
Test script to verify promotion age sensitivity implementation.
This script creates sample data with different age groups and verifies that
age multipliers are correctly applied to promotion probabilities.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our target functions and schema constants
from cost_model.engines.markov_promotion import apply_markov_promotions, _load_hazard_defaults, _apply_promotion_age_multipliers
from cost_model.state.schema import EMP_LEVEL, EMP_ID, EMP_HIRE_DATE, EMP_BIRTH_DATE, EMP_GROSS_COMP, EMP_EXITED

def create_test_data_with_ages():
    """Create test data with employees of different ages."""
    today = pd.Timestamp.today()
    
    # Create employees with specific age distributions
    employees = []
    
    # Young employees (<30): should have higher promotion rates (1.4x multiplier)
    for i in range(20):
        birth_date = today - pd.Timedelta(days=365 * (25 + i % 5))  # Ages 25-29
        employees.append({
            EMP_ID: f"YOUNG-{i:03d}",
            EMP_LEVEL: 1 + (i % 3),  # Levels 1-3
            EMP_BIRTH_DATE: birth_date,
            EMP_HIRE_DATE: today - pd.Timedelta(days=365 * (1 + i % 3)),
            EMP_GROSS_COMP: 50000 + (i * 1000),
            EMP_EXITED: False
        })
    
    # Middle-aged employees (40-49): should have lower promotion rates (0.9x multiplier)
    for i in range(20):
        birth_date = today - pd.Timedelta(days=365 * (40 + i % 10))  # Ages 40-49
        employees.append({
            EMP_ID: f"MIDDLE-{i:03d}",
            EMP_LEVEL: 2 + (i % 3),  # Levels 2-4
            EMP_BIRTH_DATE: birth_date,
            EMP_HIRE_DATE: today - pd.Timedelta(days=365 * (3 + i % 5)),
            EMP_GROSS_COMP: 70000 + (i * 1500),
            EMP_EXITED: False
        })
    
    # Older employees (50-59): should have much lower promotion rates (0.4x multiplier)
    for i in range(20):
        birth_date = today - pd.Timedelta(days=365 * (50 + i % 10))  # Ages 50-59
        employees.append({
            EMP_ID: f"OLDER-{i:03d}",
            EMP_LEVEL: 3 + (i % 3),  # Levels 3-5
            EMP_BIRTH_DATE: birth_date,
            EMP_HIRE_DATE: today - pd.Timedelta(days=365 * (5 + i % 10)),
            EMP_GROSS_COMP: 90000 + (i * 2000),
            EMP_EXITED: False
        })
    
    df = pd.DataFrame(employees)
    logger.info(f"Created test data with {len(df)} employees across 3 age groups")
    
    # Log age distribution
    ages = ((today - pd.to_datetime(df[EMP_BIRTH_DATE])).dt.days / 365.25).round()
    age_groups = pd.cut(ages, bins=[0, 30, 40, 50, 60, 100], labels=['<30', '30-39', '40-49', '50-59', '60+'])
    logger.info(f"Age distribution: {age_groups.value_counts().to_dict()}")
    
    return df

def create_simple_promotion_matrix():
    """Create a simple promotion matrix for testing."""
    # Create a matrix where employees have a 20% chance of promotion, 80% stay
    matrix_data = {
        1: [0.8, 0.2, 0.0, 0.0, 0.0],  # Level 1 -> 80% stay, 20% to level 2
        2: [0.0, 0.8, 0.2, 0.0, 0.0],  # Level 2 -> 80% stay, 20% to level 3
        3: [0.0, 0.0, 0.8, 0.2, 0.0],  # Level 3 -> 80% stay, 20% to level 4
        4: [0.0, 0.0, 0.0, 0.8, 0.2],  # Level 4 -> 80% stay, 20% to level 5
        5: [0.0, 0.0, 0.0, 0.0, 1.0],  # Level 5 -> 100% stay (top level)
    }
    
    df_matrix = pd.DataFrame(matrix_data).T
    df_matrix.columns = [1, 2, 3, 4, 5]
    
    # Verify all rows sum to 1.0
    row_sums = df_matrix.sum(axis=1)
    for i, sum_val in row_sums.items():
        if not np.isclose(sum_val, 1.0):
            raise ValueError(f"Row {i} sum is {sum_val}, not 1.0")
    
    logger.info("Created simple promotion matrix with 20% promotion probability per level")
    return df_matrix

def test_age_sensitivity():
    """Test that age multipliers are correctly applied to promotion probabilities."""
    logger.info("=== Testing Promotion Age Sensitivity ===")
    
    # Create test data
    df = create_test_data_with_ages()
    matrix = create_simple_promotion_matrix()
    
    # Set up test parameters
    promo_time = pd.Timestamp('2025-06-01')
    simulation_year = 2025
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    
    # Load hazard defaults to verify age multipliers are configured
    hazard_defaults = _load_hazard_defaults()
    if 'promotion' in hazard_defaults and 'age_multipliers' in hazard_defaults['promotion']:
        age_multipliers = hazard_defaults['promotion']['age_multipliers']
        logger.info(f"Loaded age multipliers: {age_multipliers}")
    else:
        logger.warning("No age multipliers found in hazard defaults!")
        return False
    
    # Test the age multiplier application function directly
    logger.info("Testing _apply_promotion_age_multipliers function...")
    df_with_multipliers = df.copy()
    df_with_multipliers = _apply_promotion_age_multipliers(df_with_multipliers, matrix, simulation_year, hazard_defaults)
    
    # Verify that age multipliers were added to the snapshot
    if '_promotion_age_multiplier' not in df_with_multipliers.columns:
        logger.error("Age multipliers were not added to the snapshot!")
        return False
    
    # Check that different age groups have different multipliers
    multipliers = df_with_multipliers['_promotion_age_multiplier'].unique()
    logger.info(f"Unique age multipliers applied: {sorted(multipliers)}")
    
    # Verify specific age groups have expected multipliers
    young_multipliers = df_with_multipliers[df_with_multipliers[EMP_ID].str.startswith('YOUNG')]['_promotion_age_multiplier'].unique()
    middle_multipliers = df_with_multipliers[df_with_multipliers[EMP_ID].str.startswith('MIDDLE')]['_promotion_age_multiplier'].unique()
    older_multipliers = df_with_multipliers[df_with_multipliers[EMP_ID].str.startswith('OLDER')]['_promotion_age_multiplier'].unique()
    
    logger.info(f"Young employees (<30) multipliers: {young_multipliers}")
    logger.info(f"Middle employees (40-49) multipliers: {middle_multipliers}")
    logger.info(f"Older employees (50-59) multipliers: {older_multipliers}")
    
    # Test the full promotion pipeline
    logger.info("Testing full promotion pipeline with age sensitivity...")
    promotions_df, raises_df, exits_df = apply_markov_promotions(
        df,
        promo_time=promo_time,
        rng=rng,
        simulation_year=simulation_year,
        promotion_matrix=matrix
    )
    
    logger.info(f"Promotion results: {len(promotions_df)} promotions, {len(raises_df)} raises, {len(exits_df)} exits")
    
    # Analyze promotion rates by age group
    if not promotions_df.empty:
        promoted_ids = promotions_df['employee_id'].tolist()
        young_promotions = len([id for id in promoted_ids if id.startswith('YOUNG')])
        middle_promotions = len([id for id in promoted_ids if id.startswith('MIDDLE')])
        older_promotions = len([id for id in promoted_ids if id.startswith('OLDER')])
        
        young_rate = young_promotions / 20 * 100
        middle_rate = middle_promotions / 20 * 100
        older_rate = older_promotions / 20 * 100
        
        logger.info(f"Promotion rates by age group:")
        logger.info(f"  Young (<30): {young_promotions}/20 = {young_rate:.1f}%")
        logger.info(f"  Middle (40-49): {middle_promotions}/20 = {middle_rate:.1f}%")
        logger.info(f"  Older (50-59): {older_promotions}/20 = {older_rate:.1f}%")
        
        # Verify that younger employees have higher promotion rates
        if young_rate > middle_rate > older_rate:
            logger.info("✓ Age sensitivity working correctly: younger employees have higher promotion rates")
            return True
        else:
            logger.warning("⚠ Age sensitivity may not be working as expected")
            return False
    else:
        logger.warning("No promotions occurred in test - cannot verify age sensitivity")
        return False

if __name__ == "__main__":
    logger.info("Starting promotion age sensitivity test")
    success = test_age_sensitivity()
    if success:
        logger.info("✓ Promotion age sensitivity test PASSED")
    else:
        logger.error("✗ Promotion age sensitivity test FAILED")
    sys.exit(0 if success else 1)
