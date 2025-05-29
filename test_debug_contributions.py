#!/usr/bin/env python3
"""
Test script to debug apply_contributions function with detailed logging.
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

def test_apply_contributions():
    """Test apply_contributions with realistic data."""
    try:
        # Import required modules
        from cost_model.rules.contributions import apply as apply_contributions
        from cost_model.rules.validators import ContributionsRule, MatchRule, NonElectiveRule, Tier
        from cost_model.state.schema import (
            EMP_ID, EMP_GROSS_COMP, EMP_DEFERRAL_RATE, EMP_CONTR, 
            EMPLOYER_MATCH, EMPLOYER_CORE, EMP_STATUS_EOY, EMP_ACTIVE
        )
        
        logger.info("Creating test DataFrame...")
        
        # Create test data that mimics real simulation data
        test_data = {
            EMP_ID: ['EMP001', 'EMP002', 'EMP003', 'EMP004', 'EMP005'],
            EMP_GROSS_COMP: [75000.0, 120000.0, 50000.0, 200000.0, 85000.0],
            EMP_DEFERRAL_RATE: [0.06, 0.10, 0.03, 0.15, 0.08],
            EMP_STATUS_EOY: ['Active', 'Active', 'Terminated', 'Active', 'Active'],
            EMP_ACTIVE: [True, True, False, True, True],
            'birth_date': pd.to_datetime(['1980-01-01', '1975-06-15', '1990-03-20', '1970-12-10', '1985-08-05']),
            'employee_hire_date': pd.to_datetime(['2020-01-01', '2018-03-01', '2022-06-01', '2015-01-01', '2019-09-01']),
            'employee_termination_date': [pd.NaT, pd.NaT, pd.to_datetime('2025-06-30'), pd.NaT, pd.NaT]
        }
        
        test_df = pd.DataFrame(test_data)
        logger.info(f"Test DataFrame created with shape: {test_df.shape}")
        logger.info(f"Test DataFrame columns: {test_df.columns.tolist()}")
        logger.info(f"Test DataFrame:\n{test_df}")
        
        # Create rule objects
        logger.info("Creating rule objects...")
        contrib_rules = ContributionsRule(enabled=True)
        
        # Create match tiers
        tier1 = Tier(match_rate=0.5, cap_deferral_pct=0.06)  # 50% match up to 6%
        match_rules = MatchRule(tiers=[tier1], dollar_cap=None)
        
        nec_rules = NonElectiveRule(rate=0.01)  # 1% non-elective
        
        # Create IRS limits
        irs_limits = {
            2025: {
                'compensation_limit': 350000,
                'deferral_limit': 23500,
                'catchup_limit': 7500,
                'catchup_eligibility_age': 50
            }
        }
        
        logger.info("Rule objects created successfully")
        logger.info(f"contrib_rules: {contrib_rules}")
        logger.info(f"match_rules: {match_rules}")
        logger.info(f"nec_rules: {nec_rules}")
        
        # Apply contributions
        logger.info("Calling apply_contributions...")
        result_df = apply_contributions(
            df=test_df,
            contrib_rules=contrib_rules,
            match_rules=match_rules,
            nec_rules=nec_rules,
            irs_limits=irs_limits,
            simulation_year=2025,
            year_start=pd.Timestamp('2025-01-01'),
            year_end=pd.Timestamp('2025-12-31')
        )
        
        logger.info("apply_contributions completed successfully!")
        logger.info(f"Result DataFrame shape: {result_df.shape}")
        
        # Check contribution columns
        contrib_cols = [EMP_CONTR, EMPLOYER_MATCH, EMPLOYER_CORE]
        logger.info("Final contribution values:")
        for col in contrib_cols:
            if col in result_df.columns:
                values = result_df[col].tolist()
                logger.info(f"  {col}: {values}")
                logger.info(f"  {col} dtype: {result_df[col].dtype}")
                logger.info(f"  {col} null count: {result_df[col].isnull().sum()}")
            else:
                logger.error(f"  {col}: MISSING")
        
        # Show final result
        display_cols = [EMP_ID, EMP_GROSS_COMP, EMP_DEFERRAL_RATE, EMP_CONTR, EMPLOYER_MATCH, EMPLOYER_CORE]
        available_cols = [col for col in display_cols if col in result_df.columns]
        logger.info(f"Final result:\n{result_df[available_cols]}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_apply_contributions()
    if success:
        logger.info("✅ Test completed successfully!")
    else:
        logger.error("❌ Test failed!")
        sys.exit(1)
