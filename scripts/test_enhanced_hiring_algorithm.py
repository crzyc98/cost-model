#!/usr/bin/env python3
"""
Test the Enhanced Hiring Algorithm Fix

This script tests the fix for the hiring algorithm that now accounts for
expected experienced employee terminations throughout the year, not just
those that have already occurred.

The fix addresses the core issue where:
- Old logic: hire based only on employees who have already left
- New logic: hire based on ALL expected experienced employee exits for the year

Expected outcome: Headcount growth should now match the target growth rate.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_enhanced_hiring_test():
    """Run a test simulation with the enhanced hiring algorithm."""
    logger.info("🚀 Testing Enhanced Hiring Algorithm Fix")
    logger.info("=" * 60)
    
    try:
        # Import required modules
        from cost_model.simulation import run_simulation
        from cost_model.config.loader import load_main_config
        
        # Load configuration
        config_path = "configs/main_config.yaml"
        main_config = load_main_config(config_path)
        
        # Set test parameters for clear results
        main_config.global_parameters.target_growth = 0.05  # 5% growth for clear signal
        main_config.global_parameters.new_hire_termination_rate = 0.25  # 25% NH termination
        main_config.global_parameters.projection_years = 2  # Shorter test
        main_config.global_parameters.random_seed = 123  # Reproducible results
        
        # Set up output directory
        output_dir = Path("output_dev/enhanced_hiring_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Census path
        census_path = "data/census_template.parquet"
        
        logger.info("Configuration:")
        logger.info(f"  Target Growth: {main_config.global_parameters.target_growth:.1%}")
        logger.info(f"  NH Termination Rate: {main_config.global_parameters.new_hire_termination_rate:.1%}")
        logger.info(f"  Projection Years: {main_config.global_parameters.projection_years}")
        logger.info(f"  Output Directory: {output_dir}")
        
        # Run simulation
        logger.info("\n🔄 Running simulation with enhanced hiring algorithm...")
        run_simulation(
            main_config=main_config,
            scenario_name="enhanced_hiring_test",
            input_census_path=census_path,
            output_dir_base=output_dir,
            save_detailed_snapshots=True,
            save_summary_metrics=True,
            random_seed=123
        )
        
        # Analyze results
        logger.info("\n📊 Analyzing results...")
        scenario_output_dir = output_dir / "enhanced_hiring_test"
        
        # Load results
        snapshots_path = scenario_output_dir / "consolidated_snapshots.parquet"
        events_path = scenario_output_dir / "enhanced_hiring_test_event_log.parquet"
        
        if snapshots_path.exists() and events_path.exists():
            snapshots_df = pd.read_parquet(snapshots_path)
            events_df = pd.read_parquet(events_path)
            
            # Analyze headcount progression
            analyze_headcount_results(snapshots_df, events_df, main_config.global_parameters.target_growth)
            
            return True
        else:
            logger.error("❌ Output files not found")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def analyze_headcount_results(snapshots_df: pd.DataFrame, events_df: pd.DataFrame, target_growth: float):
    """Analyze the headcount results from the enhanced hiring algorithm."""
    logger.info("\n📈 HEADCOUNT ANALYSIS")
    logger.info("=" * 40)
    
    # Analyze headcount progression
    years = sorted(snapshots_df['simulation_year'].unique())
    headcount_progression = {}
    
    for year in years:
        year_data = snapshots_df[snapshots_df['simulation_year'] == year]
        active_count = year_data['employee_active'].sum() if 'employee_active' in year_data.columns else len(year_data)
        headcount_progression[year] = active_count
        logger.info(f"  Year {year}: {active_count} active employees")
    
    # Calculate actual growth
    start_year = min(years)
    end_year = max(years)
    start_count = headcount_progression[start_year]
    end_count = headcount_progression[end_year]
    
    total_growth = (end_count - start_count) / start_count
    annual_growth = total_growth / (len(years) - 1)
    
    logger.info(f"\n📊 GROWTH ANALYSIS:")
    logger.info(f"  Start Count: {start_count}")
    logger.info(f"  End Count: {end_count}")
    logger.info(f"  Total Growth: {total_growth:.2%}")
    logger.info(f"  Annual Growth: {annual_growth:.2%}")
    logger.info(f"  Target Growth: {target_growth:.2%}")
    logger.info(f"  Growth Gap: {annual_growth - target_growth:.2%}")
    
    # Analyze hiring vs termination balance
    logger.info(f"\n🔄 HIRING VS TERMINATION ANALYSIS:")
    
    # Count events by type
    hire_events = events_df[events_df['event_type'] == 'EVT_HIRE']
    term_events = events_df[events_df['event_type'] == 'EVT_TERM']
    
    # Classify terminations
    nh_by_json = term_events['value_json'].str.contains('new_hire_termination', na=False)
    nh_by_meta = term_events['meta'].str.contains('New-hire termination', na=False)
    nh_terms = term_events[nh_by_json | nh_by_meta]
    exp_terms = term_events[~(nh_by_json | nh_by_meta)]
    
    total_hires = len(hire_events)
    total_nh_terms = len(nh_terms)
    total_exp_terms = len(exp_terms)
    
    logger.info(f"  Total Hires: {total_hires}")
    logger.info(f"  New Hire Terminations: {total_nh_terms}")
    logger.info(f"  Experienced Terminations: {total_exp_terms}")
    logger.info(f"  Net Change: {total_hires - total_nh_terms - total_exp_terms:+d}")
    
    # Calculate rates
    if total_hires > 0:
        actual_nh_rate = total_nh_terms / total_hires
        logger.info(f"  Actual NH Termination Rate: {actual_nh_rate:.1%}")
    
    # Success criteria
    logger.info(f"\n✅ SUCCESS CRITERIA:")
    growth_success = abs(annual_growth - target_growth) < 0.01  # Within 1%
    nh_rate_success = abs(actual_nh_rate - 0.25) < 0.05 if total_hires > 0 else True  # Within 5%
    
    logger.info(f"  Growth Target Met: {'✅' if growth_success else '❌'} "
                f"({annual_growth:.1%} vs {target_growth:.1%})")
    logger.info(f"  NH Rate Accurate: {'✅' if nh_rate_success else '❌'} "
                f"({actual_nh_rate:.1%} vs 25.0%)" if total_hires > 0 else "  NH Rate: N/A")
    
    overall_success = growth_success and nh_rate_success
    logger.info(f"  Overall Success: {'✅' if overall_success else '❌'}")
    
    if overall_success:
        logger.info("\n🎉 ENHANCED HIRING ALGORITHM FIX SUCCESSFUL!")
        logger.info("The hiring algorithm now correctly accounts for expected experienced attrition.")
    else:
        logger.info("\n⚠️  Enhanced hiring algorithm needs further adjustment.")
        if not growth_success:
            logger.info("   - Growth target not met")
        if not nh_rate_success:
            logger.info("   - New hire termination rate inaccurate")

def main():
    """Main test function."""
    success = run_enhanced_hiring_test()
    
    if success:
        logger.info("\n🎯 Enhanced hiring algorithm test completed successfully!")
    else:
        logger.error("\n❌ Enhanced hiring algorithm test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
