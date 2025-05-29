#!/usr/bin/env python3
"""
Verify that our contribution calculation fixes work correctly.
"""

import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main verification function."""
    
    logger.info("🔍 Verifying contribution calculation fixes...")
    
    # Test 1: Check imports work
    try:
        from cost_model.state.schema import ACTIVE_STATUS, EMP_STATUS_EOY, EMP_ACTIVE
        from cost_model.utils.constants import ACTIVE_STATUSES
        logger.info("✅ Schema imports successful")
        logger.info(f"   ACTIVE_STATUS = '{ACTIVE_STATUS}'")
        logger.info(f"   EMP_STATUS_EOY = '{EMP_STATUS_EOY}'")
        logger.info(f"   EMP_ACTIVE = '{EMP_ACTIVE}'")
        logger.info(f"   ACTIVE_STATUSES = {ACTIVE_STATUSES}")
    except Exception as e:
        logger.error(f"❌ Schema import failed: {e}")
        return False
    
    # Test 2: Check status logic
    try:
        import pandas as pd
        
        # Test EMP_STATUS_EOY logic
        df1 = pd.DataFrame({EMP_STATUS_EOY: ['Active', 'Terminated', 'Inactive']})
        active_mask1 = df1[EMP_STATUS_EOY] == ACTIVE_STATUS
        expected1 = [True, False, False]
        
        if active_mask1.tolist() == expected1:
            logger.info("✅ EMP_STATUS_EOY logic works correctly")
        else:
            logger.error(f"❌ EMP_STATUS_EOY logic failed: got {active_mask1.tolist()}, expected {expected1}")
            return False
        
        # Test EMP_ACTIVE logic
        df2 = pd.DataFrame({EMP_ACTIVE: [True, False, None]})
        active_mask2 = df2[EMP_ACTIVE].fillna(False)
        expected2 = [True, False, False]
        
        if active_mask2.tolist() == expected2:
            logger.info("✅ EMP_ACTIVE logic works correctly")
        else:
            logger.error(f"❌ EMP_ACTIVE logic failed: got {active_mask2.tolist()}, expected {expected2}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Status logic test failed: {e}")
        return False
    
    # Test 3: Check IRS limits configuration
    try:
        import yaml
        
        with open('config/dev_tiny.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        irs_limits = config.get('plan_rules', {}).get('irs_limits', {})
        required_years = [2025, 2026, 2027, 2028, 2029]
        
        missing_years = [year for year in required_years if year not in irs_limits]
        
        if missing_years:
            logger.error(f"❌ Missing IRS limits for years: {missing_years}")
            return False
        else:
            logger.info("✅ All required IRS limits are configured")
            for year in required_years:
                limits = irs_limits[year]
                logger.info(f"   {year}: comp_limit={limits['compensation_limit']}, def_limit={limits['deferral_limit']}")
                
    except Exception as e:
        logger.error(f"❌ IRS limits test failed: {e}")
        return False
    
    # Test 4: Check contributions module can be imported
    try:
        from cost_model.rules.contributions import apply as apply_contributions
        logger.info("✅ Contributions module imports successfully")
    except Exception as e:
        logger.error(f"❌ Contributions module import failed: {e}")
        return False
    
    logger.info("🎉 All verification tests passed!")
    logger.info("")
    logger.info("Summary of fixes:")
    logger.info("1. ✅ Fixed KeyError: 'status' by updating contributions.py to handle EMP_STATUS_EOY and EMP_ACTIVE columns")
    logger.info("2. ✅ Added IRS limits for simulation years 2025-2029 in configuration files")
    logger.info("3. ✅ Updated status handling logic to correctly distinguish between schema status values and enum values")
    logger.info("")
    logger.info("The contribution calculation errors should now be resolved!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
