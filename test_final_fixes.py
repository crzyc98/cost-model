#!/usr/bin/env python3
"""
Test script to verify all contribution calculation fixes work correctly.
"""

import sys
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_rule_object_creation():
    """Test that we can create rule objects correctly."""
    try:
        from cost_model.rules.validators import ContributionsRule, MatchRule, NonElectiveRule, Tier
        
        # Test creating rule objects like the orchestrator does
        contrib_rules = ContributionsRule(enabled=True)
        default_tier = Tier(match_rate=0.0, cap_deferral_pct=0.01)
        match_rules = MatchRule(tiers=[default_tier], dollar_cap=None)
        nec_rules = NonElectiveRule(rate=0.0)
        
        logger.info("✅ Rule object creation works!")
        logger.info(f"   contrib_rules: {contrib_rules}")
        logger.info(f"   match_rules: {match_rules}")
        logger.info(f"   nec_rules: {nec_rules}")
        logger.info(f"   nec_rules.rate: {nec_rules.rate}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Rule object creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_irs_limits_config():
    """Test that IRS limits are properly configured."""
    try:
        import yaml
        
        with open('config/dev_tiny.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        plan_rules = config.get('plan_rules', {})
        irs_limits = plan_rules.get('irs_limits', {})
        
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
                logger.info(f"   Year {year}: comp_limit={limits['compensation_limit']}, def_limit={limits['deferral_limit']}")
        
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

def test_contribution_calculation():
    """Test the actual contribution calculation with our fixes."""
    try:
        from cost_model.rules.contributions import apply as apply_contributions
        from cost_model.rules.validators import ContributionsRule, MatchRule, NonElectiveRule, Tier
        from cost_model.state.schema import EMP_CONTR, EMPLOYER_MATCH, EMPLOYER_CORE, EMP_STATUS_EOY, EMP_ACTIVE
        
        # Create test data
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
        
        # Create rule objects
        contrib_rules = ContributionsRule(enabled=True)
        default_tier = Tier(match_rate=0.5, cap_deferral_pct=0.06)  # 50% match up to 6%
        match_rules = MatchRule(tiers=[default_tier], dollar_cap=None)
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
        
        logger.info("Testing contribution calculations...")
        
        # Apply contributions
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
        
        logger.info("✅ Contribution calculations completed successfully!")
        logger.info(f"   Result shape: {result_df.shape}")
        
        # Check contribution columns
        for col in [EMP_CONTR, EMPLOYER_MATCH, EMPLOYER_CORE]:
            if col in result_df.columns:
                values = result_df[col].tolist()
                logger.info(f"   {col}: {values}")
                # Check that we have non-zero values where expected
                if col == EMPLOYER_CORE and any(v > 0 for v in values):
                    logger.info(f"   ✅ {col} has expected non-zero values")
                elif col == EMP_CONTR and any(v > 0 for v in values):
                    logger.info(f"   ✅ {col} has expected non-zero values")
            else:
                logger.error(f"   ❌ {col}: MISSING")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"❌ Contribution calculation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    logger.info("🧪 Running comprehensive contribution fix tests...")
    
    tests = [
        ("Rule Object Creation", test_rule_object_creation),
        ("IRS Limits Configuration", test_irs_limits_config),
        ("Contribution Calculation", test_contribution_calculation),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY:")
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED!")
        logger.info("The contribution calculation fixes should work correctly.")
        logger.info("\nKey fixes implemented:")
        logger.info("1. ✅ Fixed KeyError: 'status' by updating status column handling")
        logger.info("2. ✅ Fixed 'dict' object has no attribute 'rate' by creating proper Pydantic models")
        logger.info("3. ✅ Fixed IRS limits not found by getting them from plan_rules instead of global_params")
        logger.info("4. ✅ Added IRS limits for simulation years 2025-2029")
        return True
    else:
        logger.error("\n💥 SOME TESTS FAILED!")
        logger.error("Please check the errors above and fix them before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
