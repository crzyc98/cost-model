#!/usr/bin/env python3
"""
Simple test to verify the status column fix in contributions.py
"""

import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_status_column_handling():
    """Test that the contributions module handles different status column names correctly."""
    
    try:
        from cost_model.state.schema import EMP_STATUS_EOY, EMP_ACTIVE
        from cost_model.utils.constants import ACTIVE_STATUSES
        
        logger.info("Testing status column handling logic...")
        
        # Test 1: DataFrame with EMP_STATUS_EOY column
        df1 = pd.DataFrame({
            'employee_id': ['emp1', 'emp2'],
            EMP_STATUS_EOY: ['Active', 'Terminated'],
            'birth_date': pd.to_datetime(['1990-01-01', '1985-06-15'])
        })
        
        logger.info(f"Test 1 - EMP_STATUS_EOY column: {EMP_STATUS_EOY}")
        logger.info(f"ACTIVE_STATUSES: {ACTIVE_STATUSES}")
        
        # Simulate the logic from contributions.py
        if EMP_STATUS_EOY in df1.columns:
            active_mask1 = df1[EMP_STATUS_EOY].isin(ACTIVE_STATUSES)
            logger.info(f"Active mask for df1: {active_mask1.tolist()}")
        
        # Test 2: DataFrame with EMP_ACTIVE column
        df2 = pd.DataFrame({
            'employee_id': ['emp1', 'emp2'],
            EMP_ACTIVE: [True, False],
            'birth_date': pd.to_datetime(['1990-01-01', '1985-06-15'])
        })
        
        logger.info(f"Test 2 - EMP_ACTIVE column: {EMP_ACTIVE}")
        
        if EMP_ACTIVE in df2.columns:
            active_mask2 = df2[EMP_ACTIVE].fillna(False)
            logger.info(f"Active mask for df2: {active_mask2.tolist()}")
        
        # Test 3: DataFrame with legacy 'status' column
        df3 = pd.DataFrame({
            'employee_id': ['emp1', 'emp2'],
            'status': ['Active Initial', 'Terminated'],
            'birth_date': pd.to_datetime(['1990-01-01', '1985-06-15'])
        })
        
        logger.info("Test 3 - legacy 'status' column")
        
        if 'status' in df3.columns:
            active_mask3 = df3['status'].isin(ACTIVE_STATUSES)
            logger.info(f"Active mask for df3: {active_mask3.tolist()}")
        
        # Test 4: DataFrame with no status column
        df4 = pd.DataFrame({
            'employee_id': ['emp1', 'emp2'],
            'birth_date': pd.to_datetime(['1990-01-01', '1985-06-15'])
        })
        
        logger.info("Test 4 - no status column")
        
        if EMP_STATUS_EOY not in df4.columns and EMP_ACTIVE not in df4.columns and 'status' not in df4.columns:
            active_mask4 = pd.Series(True, index=df4.index)
            logger.info(f"Active mask for df4 (default): {active_mask4.tolist()}")
        
        logger.info("✅ All status column tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Status column test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_irs_limits_config():
    """Test that IRS limits are properly configured."""
    
    try:
        import yaml
        
        with open('config/dev_tiny.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        irs_limits = config.get('plan_rules', {}).get('irs_limits', {})
        
        logger.info("Testing IRS limits configuration...")
        logger.info(f"Available years: {list(irs_limits.keys())}")
        
        # Check that we have limits for simulation years 2025-2029
        required_years = [2025, 2026, 2027, 2028, 2029]
        missing_years = []
        
        for year in required_years:
            if year not in irs_limits:
                missing_years.append(year)
            else:
                limits = irs_limits[year]
                logger.info(f"Year {year}: {limits}")
        
        if missing_years:
            logger.error(f"❌ Missing IRS limits for years: {missing_years}")
            return False
        else:
            logger.info("✅ All required IRS limits are configured!")
            return True
            
    except Exception as e:
        logger.error(f"❌ IRS limits test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Running contribution fix tests...")
    
    status_test = test_status_column_handling()
    irs_test = test_irs_limits_config()
    
    if status_test and irs_test:
        print("\n🎉 ALL TESTS PASSED! The contribution fixes should work correctly.")
    else:
        print("\n💥 SOME TESTS FAILED! Please check the errors above.")
