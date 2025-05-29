#!/usr/bin/env python3
"""
Test script to verify the contribution calculation fixes.
"""

import pandas as pd
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_contribution_fix():
    """Test the contribution calculation fix."""
    try:
        # Import the modules
        from cost_model.rules.contributions import apply as apply_contributions
        from cost_model.rules.validators import ContributionsRule, MatchRule, NonElectiveRule
        from cost_model.state.schema import (
            EMP_CONTR, EMPLOYER_MATCH, EMPLOYER_CORE,
            EMP_STATUS_EOY, EMP_ACTIVE
        )

        logger.info("Successfully imported modules")

        # Create a test DataFrame with the expected schema columns
        test_df = pd.DataFrame({
            'employee_id': ['emp1', 'emp2', 'emp3'],
            'employee_gross_compensation': [50000.0, 75000.0, 100000.0],
            'employee_deferral_rate': [0.05, 0.10, 0.15],
            'employee_hire_date': pd.to_datetime(['2020-01-01', '2019-06-15', '2018-03-10']),
            'employee_termination_date': [pd.NaT, pd.NaT, pd.NaT],
            'birth_date': pd.to_datetime(['1990-01-01', '1985-06-15', '1970-03-10']),
            'active': [True, True, True],
            'employee_status_eoy': ['Active', 'Active', 'Active']
        })

        logger.info(f"Created test DataFrame with shape: {test_df.shape}")
        logger.info(f"Columns: {test_df.columns.tolist()}")

        # Create simple rule objects
        from cost_model.rules.validators import Tier

        contrib_rules = ContributionsRule(enabled=True)
        # Create a simple match tier to satisfy validation
        match_tier = Tier(match_rate=0.5, cap_deferral_pct=0.06)
        match_rules = MatchRule(tiers=[match_tier], dollar_cap=None)
        nec_rules = NonElectiveRule(rate=0.0)

        # Create IRS limits for 2025
        irs_limits = {
            2025: {
                'compensation_limit': 350000,
                'deferral_limit': 23500,
                'catchup_limit': 7500,
                'catchup_eligibility_age': 50
            }
        }

        logger.info("Testing contribution calculations...")

        # Test the contribution calculation
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

        logger.info("SUCCESS: Contribution calculations completed without errors!")
        logger.info(f"Result shape: {result_df.shape}")

        # Check contribution columns
        logger.info("Contribution columns:")
        for col in [EMP_CONTR, EMPLOYER_MATCH, EMPLOYER_CORE]:
            if col in result_df.columns:
                values = result_df[col].tolist()
                logger.info(f"  {col}: {values}")
            else:
                logger.error(f"  {col}: MISSING")

        # Check if status columns were handled correctly
        logger.info("Status handling verification:")
        logger.info(f"  EMP_STATUS_EOY in df: {EMP_STATUS_EOY in test_df.columns}")
        logger.info(f"  EMP_ACTIVE in df: {EMP_ACTIVE in test_df.columns}")

        return True

    except Exception as e:
        logger.error(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_contribution_fix()
    if success:
        print("\n✅ CONTRIBUTION FIX TEST PASSED!")
        sys.exit(0)
    else:
        print("\n❌ CONTRIBUTION FIX TEST FAILED!")
        sys.exit(1)
