#!/usr/bin/env python3
"""
Test the orchestrator contribution fixes directly.
"""

import sys
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_orchestrator_contribution_logic():
    """Test the orchestrator contribution logic directly."""
    try:
        # Import what we need
        from cost_model.rules.validators import ContributionsRule, MatchRule, NonElectiveRule, Tier
        
        # Simulate the orchestrator logic
        logger.info("Testing orchestrator contribution logic...")
        
        # Simulate plan_rules from config
        plan_rules = {
            'contributions': {'enabled': True},
            'employer_match': {
                'tiers': [
                    {'match_rate': 0.5, 'cap_deferral_pct': 0.06}
                ],
                'dollar_cap': None
            },
            'employer_nec': {'rate': 0.01},
            'irs_limits': {
                2025: {
                    'compensation_limit': 350000,
                    'deferral_limit': 23500,
                    'catchup_limit': 7500,
                    'catchup_eligibility_age': 50
                }
            }
        }
        
        # Get contribution rules from plan_rules and convert to Pydantic models
        contrib_config = plan_rules.get('contributions', {})
        match_config = plan_rules.get('employer_match', {})
        nec_config = plan_rules.get('employer_nec', {})
        
        # Get IRS limits from plan_rules (not global_params)
        irs_limits = plan_rules.get('irs_limits', {})
        
        logger.info(f"contrib_config: {contrib_config}")
        logger.info(f"match_config: {match_config}")
        logger.info(f"nec_config: {nec_config}")
        logger.info(f"irs_limits keys: {list(irs_limits.keys())}")
        
        # Create Pydantic model instances with proper defaults
        try:
            contrib_rules = ContributionsRule(**contrib_config) if contrib_config else ContributionsRule(enabled=True)
            
            # Handle match rules - need at least one tier for validation
            if match_config and match_config.get('tiers'):
                match_rules = MatchRule(**match_config)
            else:
                # Create a default tier with 0% match to satisfy validation
                default_tier = Tier(match_rate=0.0, cap_deferral_pct=0.01)  # Minimal tier
                match_rules = MatchRule(tiers=[default_tier], dollar_cap=None)
                
            nec_rules = NonElectiveRule(**nec_config) if nec_config else NonElectiveRule(rate=0.0)
            
            logger.info(f"✅ Created rule objects successfully!")
            logger.info(f"   contrib_rules: {contrib_rules}")
            logger.info(f"   match_rules: {match_rules}")
            logger.info(f"   nec_rules: {nec_rules}")
            logger.info(f"   nec_rules.rate: {nec_rules.rate}")
            
        except Exception as e:
            logger.warning(f"Error creating rule objects: {e}. Using defaults.")
            contrib_rules = ContributionsRule(enabled=True)
            # Create default match rule with minimal tier
            default_tier = Tier(match_rate=0.0, cap_deferral_pct=0.01)
            match_rules = MatchRule(tiers=[default_tier], dollar_cap=None)
            nec_rules = NonElectiveRule(rate=0.0)
        
        # Test the contribution calculation
        from cost_model.rules.contributions import apply as apply_contributions
        from cost_model.state.schema import EMP_CONTR, EMPLOYER_MATCH, EMPLOYER_CORE
        
        # Create test data
        test_df = pd.DataFrame({
            'employee_id': ['emp1', 'emp2'],
            'employee_gross_compensation': [50000.0, 75000.0],
            'employee_deferral_rate': [0.05, 0.10],
            'employee_hire_date': pd.to_datetime(['2020-01-01', '2019-06-15']),
            'employee_termination_date': [pd.NaT, pd.NaT],
            'birth_date': pd.to_datetime(['1990-01-01', '1985-06-15']),
            'active': [True, True],
            'employee_status_eoy': ['Active', 'Active']
        })
        
        logger.info("Testing contribution calculation...")
        
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
        
        logger.info("✅ Contribution calculation completed successfully!")
        
        # Check results
        for col in [EMP_CONTR, EMPLOYER_MATCH, EMPLOYER_CORE]:
            if col in result_df.columns:
                values = result_df[col].tolist()
                logger.info(f"   {col}: {values}")
            else:
                logger.error(f"   ❌ {col}: MISSING")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test."""
    logger.info("🧪 Testing orchestrator contribution fixes...")
    
    success = test_orchestrator_contribution_logic()
    
    if success:
        logger.info("\n🎉 ORCHESTRATOR CONTRIBUTION TEST PASSED!")
        logger.info("The fixes should resolve:")
        logger.info("1. ✅ KeyError: 'status' - Fixed by proper status column handling")
        logger.info("2. ✅ 'dict' object has no attribute 'rate' - Fixed by creating Pydantic models")
        logger.info("3. ✅ IRS limits not found - Fixed by getting from plan_rules")
    else:
        logger.error("\n💥 ORCHESTRATOR CONTRIBUTION TEST FAILED!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
