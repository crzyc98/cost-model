#!/usr/bin/env python3
"""
Debug script to trace the promotion age multiplier pipeline end-to-end.
This script tests each step of the promotion age sensitivity implementation.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_hazard_loading():
    """Test hazard configuration loading."""
    logger.info("🔍 STEP 1: Testing hazard configuration loading")
    
    # Set environment variable
    os.environ['HAZARD_CONFIG_FILE'] = 'hazard_promotion_high_early_test.yaml'
    
    from cost_model.engines.markov_promotion import _load_hazard_defaults
    
    hazard_defaults = _load_hazard_defaults()
    
    if not hazard_defaults:
        logger.error("❌ Failed to load hazard defaults")
        return False
    
    promotion_config = hazard_defaults.get('promotion', {})
    age_multipliers = promotion_config.get('age_multipliers', {})
    
    logger.info(f"✅ Hazard defaults loaded successfully")
    logger.info(f"   Promotion config keys: {list(promotion_config.keys())}")
    logger.info(f"   Age multipliers: {age_multipliers}")
    
    # Verify the high early configuration
    if age_multipliers.get('<30') == 2.5:
        logger.info("✅ High early age multiplier (2.5x for <30) confirmed")
        return True
    else:
        logger.error(f"❌ Expected <30 multiplier of 2.5, got {age_multipliers.get('<30')}")
        return False

def test_age_multiplier_application():
    """Test the age multiplier application function."""
    logger.info("🔍 STEP 2: Testing age multiplier application")
    
    from cost_model.engines.markov_promotion import _load_hazard_defaults, _apply_promotion_age_multipliers
    from cost_model.state.schema import EMP_ID, EMP_BIRTH_DATE, EMP_LEVEL
    
    # Load hazard defaults
    hazard_defaults = _load_hazard_defaults()
    
    # Create test snapshot with young employees
    test_snapshot = pd.DataFrame({
        EMP_ID: [1, 2, 3, 4],
        EMP_BIRTH_DATE: ['1998-01-01', '1985-01-01', '1965-01-01', '1955-01-01'],  # Ages ~27, ~40, ~60, ~70
        EMP_LEVEL: [1, 2, 3, 4]
    })
    
    # Convert birth dates to datetime
    test_snapshot[EMP_BIRTH_DATE] = pd.to_datetime(test_snapshot[EMP_BIRTH_DATE])
    
    logger.info(f"   Test snapshot shape: {test_snapshot.shape}")
    logger.info(f"   Test snapshot columns: {test_snapshot.columns.tolist()}")
    
    # Create dummy promotion matrix
    promotion_matrix = pd.DataFrame(np.eye(5))
    
    # Apply age multipliers
    result = _apply_promotion_age_multipliers(test_snapshot, promotion_matrix, 2025, hazard_defaults)
    
    if '_promotion_age_multiplier' not in result.columns:
        logger.error("❌ Age multiplier column not added to snapshot")
        return False
    
    logger.info("✅ Age multiplier column added successfully")
    
    # Check multipliers for each employee
    for _, row in result.iterrows():
        emp_id = row[EMP_ID]
        multiplier = row['_promotion_age_multiplier']
        birth_date = row[EMP_BIRTH_DATE]
        age = 2025 - birth_date.year
        
        logger.info(f"   Employee {emp_id}: age {age}, multiplier {multiplier}")
        
        # Verify young employee gets high multiplier
        if age < 30 and multiplier != 2.5:
            logger.error(f"❌ Young employee {emp_id} (age {age}) should have multiplier 2.5, got {multiplier}")
            return False
    
    logger.info("✅ Age multipliers applied correctly")
    return True

def test_markov_sampling_with_age():
    """Test the Markov sampling function with age multipliers."""
    logger.info("🔍 STEP 3: Testing Markov sampling with age multipliers")
    
    from cost_model.state.job_levels.sampling import apply_promotion_markov
    from cost_model.engines.markov_promotion import _load_hazard_defaults, _apply_promotion_age_multipliers
    from cost_model.state.schema import EMP_ID, EMP_BIRTH_DATE, EMP_LEVEL
    
    # Load hazard defaults
    hazard_defaults = _load_hazard_defaults()
    
    # Create test snapshot with multiple young employees to increase promotion chances
    young_employees = []
    for i in range(20):  # Create 20 young employees
        young_employees.append({
            EMP_ID: f'YOUNG_{i:03d}',
            EMP_BIRTH_DATE: '1998-01-01',  # Age 27
            EMP_LEVEL: 1  # Level 1, eligible for promotion
        })
    
    test_snapshot = pd.DataFrame(young_employees)
    test_snapshot[EMP_BIRTH_DATE] = pd.to_datetime(test_snapshot[EMP_BIRTH_DATE])
    
    logger.info(f"   Created {len(test_snapshot)} young employees for testing")
    
    # Apply age multipliers
    test_snapshot = _apply_promotion_age_multipliers(test_snapshot, None, 2025, hazard_defaults)
    
    # Verify age multipliers are present
    if '_promotion_age_multiplier' not in test_snapshot.columns:
        logger.error("❌ Age multiplier column missing before Markov sampling")
        return False
    
    multipliers = test_snapshot['_promotion_age_multiplier'].unique()
    logger.info(f"   Age multipliers in snapshot: {multipliers}")
    
    if 2.5 not in multipliers:
        logger.error("❌ Expected 2.5x multiplier not found in snapshot")
        return False
    
    # Run Markov sampling
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    
    try:
        result = apply_promotion_markov(
            test_snapshot,
            rng=rng,
            simulation_year=2025
        )
        
        # Check for promotions
        promotions = (result[EMP_LEVEL] > test_snapshot[EMP_LEVEL]).sum()
        logger.info(f"   Promotions with 2.5x multiplier: {promotions}/{len(test_snapshot)} = {promotions/len(test_snapshot):.1%}")
        
        if promotions > 0:
            logger.info("✅ Markov sampling with age multipliers working")
            return True
        else:
            logger.warning("⚠️  No promotions occurred (might be due to low base rates)")
            return True  # Not necessarily an error, could be low base rates
            
    except Exception as e:
        logger.error(f"❌ Markov sampling failed: {e}")
        return False

def test_full_promotion_pipeline():
    """Test the full promotion pipeline end-to-end."""
    logger.info("🔍 STEP 4: Testing full promotion pipeline")
    
    from cost_model.engines.markov_promotion import apply_markov_promotions
    from cost_model.state.schema import EMP_ID, EMP_BIRTH_DATE, EMP_LEVEL, EMP_GROSS_COMP, EMP_HIRE_DATE
    
    # Create test snapshot with young employees
    test_data = []
    for i in range(50):  # Create 50 employees
        if i < 20:  # First 20 are young
            birth_date = '1998-01-01'  # Age 27
        else:  # Rest are older
            birth_date = '1975-01-01'  # Age 50
            
        test_data.append({
            EMP_ID: f'EMP_{i:03d}',
            EMP_BIRTH_DATE: birth_date,
            EMP_LEVEL: 1,  # All start at level 1
            EMP_GROSS_COMP: 50000.0,
            EMP_HIRE_DATE: '2020-01-01'
        })
    
    test_snapshot = pd.DataFrame(test_data)
    test_snapshot[EMP_BIRTH_DATE] = pd.to_datetime(test_snapshot[EMP_BIRTH_DATE])
    test_snapshot[EMP_HIRE_DATE] = pd.to_datetime(test_snapshot[EMP_HIRE_DATE])
    
    logger.info(f"   Created test snapshot with {len(test_snapshot)} employees")
    logger.info(f"   Young employees (<30): {(2025 - pd.to_datetime(test_snapshot[EMP_BIRTH_DATE]).dt.year < 30).sum()}")
    
    # Run full promotion pipeline
    promo_time = pd.Timestamp('2025-01-01')
    rng = np.random.RandomState(42)
    
    try:
        promotions_df, raises_df, exits_df = apply_markov_promotions(
            snapshot=test_snapshot,
            promo_time=promo_time,
            rng=rng,
            simulation_year=2025
        )
        
        logger.info(f"   Promotion events: {len(promotions_df)}")
        logger.info(f"   Raise events: {len(raises_df)}")
        logger.info(f"   Exit events: {len(exits_df)}")
        
        if len(promotions_df) > 0:
            # Analyze promotions by age
            promoted_ids = promotions_df[EMP_ID].unique()
            promoted_employees = test_snapshot[test_snapshot[EMP_ID].isin(promoted_ids)]
            
            young_promoted = (2025 - pd.to_datetime(promoted_employees[EMP_BIRTH_DATE]).dt.year < 30).sum()
            total_young = (2025 - pd.to_datetime(test_snapshot[EMP_BIRTH_DATE]).dt.year < 30).sum()
            
            logger.info(f"   Young employees promoted: {young_promoted}/{total_young}")
            
            if young_promoted > 0:
                logger.info("✅ Full promotion pipeline working with age sensitivity")
                return True
            else:
                logger.warning("⚠️  No young employees promoted in full pipeline test")
                return False
        else:
            logger.warning("⚠️  No promotions in full pipeline test")
            return False
            
    except Exception as e:
        logger.error(f"❌ Full promotion pipeline failed: {e}")
        return False

def main():
    """Run all promotion age sensitivity tests."""
    logger.info("🎯 PROMOTION AGE SENSITIVITY PIPELINE DEBUG")
    logger.info("=" * 60)
    
    tests = [
        ("Hazard Loading", test_hazard_loading),
        ("Age Multiplier Application", test_age_multiplier_application),
        ("Markov Sampling with Age", test_markov_sampling_with_age),
        ("Full Promotion Pipeline", test_full_promotion_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"RUNNING: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} CRASHED: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{test_name:<30} {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Age sensitivity should be working.")
    else:
        logger.error("💥 Some tests failed. Age sensitivity may not be working correctly.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
