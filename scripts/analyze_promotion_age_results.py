#!/usr/bin/env python3
"""
Detailed analysis script for promotion age sensitivity verification results.
This script examines the actual promotion events and rates by age group across scenarios.
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def analyze_promotion_events(scenario_name, output_dir):
    """Analyze promotion events from a scenario's output."""
    logger.info(f"📊 Analyzing promotion events for {scenario_name}")
    
    # Look for event log file
    event_log_file = os.path.join(output_dir, 'final_event_log.parquet')
    if not os.path.exists(event_log_file):
        logger.warning(f"   ⚠️  No event log found for {scenario_name}")
        return None
    
    try:
        # Load event log
        events_df = pd.read_parquet(event_log_file)
        logger.info(f"   Loaded {len(events_df)} events")
        
        # Filter for promotion events
        promotion_events = events_df[events_df['event_type'] == 'EVT_PROMOTION'].copy()
        logger.info(f"   Found {len(promotion_events)} promotion events")
        
        if promotion_events.empty:
            logger.warning(f"   ⚠️  No promotion events found for {scenario_name}")
            return None
        
        # Group by year and count promotions
        promotion_summary = promotion_events.groupby('year').size().reset_index(name='promotion_count')
        logger.info(f"   Promotion counts by year:")
        for _, row in promotion_summary.iterrows():
            logger.info(f"     {row['year']}: {row['promotion_count']} promotions")
        
        return {
            'total_promotions': len(promotion_events),
            'promotions_by_year': promotion_summary.set_index('year')['promotion_count'].to_dict(),
            'promotion_events': promotion_events
        }
        
    except Exception as e:
        logger.error(f"   ❌ Failed to analyze {scenario_name}: {e}")
        return None

def analyze_snapshots_with_age(scenario_name, output_dir):
    """Analyze snapshots to understand age distribution and level progression."""
    logger.info(f"📈 Analyzing snapshots with age data for {scenario_name}")
    
    # Look for consolidated snapshots
    snapshots_file = os.path.join(output_dir, 'consolidated_snapshots.parquet')
    if not os.path.exists(snapshots_file):
        logger.warning(f"   ⚠️  No consolidated snapshots found for {scenario_name}")
        return None
    
    try:
        # Load snapshots
        df = pd.read_parquet(snapshots_file)
        logger.info(f"   Loaded {len(df)} snapshot records")
        
        # Check for age-related columns
        age_cols = [col for col in df.columns if 'age' in col.lower()]
        logger.info(f"   Age-related columns: {age_cols}")
        
        # Analyze by year and age band
        if 'employee_age_band' in df.columns:
            analysis = {}
            for year in sorted(df['year'].unique()):
                year_data = df[df['year'] == year].copy()
                
                # Calculate level progression by age band
                age_level_summary = year_data.groupby('employee_age_band')['employee_level'].agg(['count', 'mean']).round(2)
                analysis[year] = age_level_summary
                
                logger.info(f"   Year {year} - Level by age band:")
                for age_band, stats in age_level_summary.iterrows():
                    logger.info(f"     {age_band}: {stats['count']} employees, avg level {stats['mean']}")
            
            return analysis
        else:
            logger.warning(f"   ⚠️  No employee_age_band column found")
            return None
            
    except Exception as e:
        logger.error(f"   ❌ Failed to analyze snapshots for {scenario_name}: {e}")
        return None

def compare_scenarios():
    """Compare promotion results across all scenarios."""
    logger.info("🔍 DETAILED PROMOTION AGE SENSITIVITY ANALYSIS")
    logger.info("=" * 80)
    
    # Define scenarios
    scenarios = {
        'promotion_baseline': 'output_dev/promotion_age_test/baseline',
        'promotion_high_early': 'output_dev/promotion_age_test/high_early',
        'promotion_high_mid': 'output_dev/promotion_age_test/high_mid',
        'promotion_low_late': 'output_dev/promotion_age_test/low_late'
    }
    
    # Analyze each scenario
    scenario_results = {}
    for scenario_name, output_dir in scenarios.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"ANALYZING SCENARIO: {scenario_name}")
        logger.info(f"{'='*60}")
        
        # Analyze promotion events
        promotion_analysis = analyze_promotion_events(scenario_name, output_dir)
        
        # Analyze snapshots
        snapshot_analysis = analyze_snapshots_with_age(scenario_name, output_dir)
        
        scenario_results[scenario_name] = {
            'promotions': promotion_analysis,
            'snapshots': snapshot_analysis
        }
    
    # Compare promotion totals across scenarios
    logger.info(f"\n🎯 PROMOTION COMPARISON SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Scenario':<25} {'Total Promotions':<20} {'Expected Impact'}")
    logger.info("-" * 70)
    
    for scenario_name, results in scenario_results.items():
        if results['promotions']:
            total_promotions = results['promotions']['total_promotions']
            
            # Determine expected impact
            if 'baseline' in scenario_name:
                expected = "Neutral (control)"
            elif 'high_early' in scenario_name:
                expected = "Higher (young boost)"
            elif 'high_mid' in scenario_name:
                expected = "Higher (mid boost)"
            elif 'low_late' in scenario_name:
                expected = "Lower (late penalty)"
            else:
                expected = "Unknown"
            
            logger.info(f"{scenario_name:<25} {total_promotions:<20} {expected}")
        else:
            logger.info(f"{scenario_name:<25} {'No data':<20} {'N/A'}")
    
    # Check if age multipliers are actually being applied
    logger.info(f"\n🔍 INVESTIGATION FINDINGS")
    logger.info("-" * 40)
    
    # Compare baseline vs high early scenario
    baseline_promotions = scenario_results.get('promotion_baseline', {}).get('promotions', {}).get('total_promotions', 0)
    high_early_promotions = scenario_results.get('promotion_high_early', {}).get('promotions', {}).get('total_promotions', 0)
    
    if baseline_promotions > 0 and high_early_promotions > 0:
        if high_early_promotions > baseline_promotions:
            logger.info("✅ High early scenario shows more promotions than baseline")
        elif high_early_promotions == baseline_promotions:
            logger.warning("⚠️  High early scenario shows SAME promotions as baseline")
            logger.warning("    This suggests age multipliers may not be working correctly")
        else:
            logger.warning("⚠️  High early scenario shows FEWER promotions than baseline")
            logger.warning("    This is unexpected and suggests a configuration issue")
    
    # Check for identical results across scenarios
    promotion_counts = [
        results.get('promotions', {}).get('total_promotions', 0) 
        for results in scenario_results.values()
    ]
    
    if len(set(promotion_counts)) == 1 and promotion_counts[0] > 0:
        logger.warning("⚠️  ALL scenarios show identical promotion counts!")
        logger.warning("    This strongly suggests age multipliers are not being applied")
        logger.warning("    Possible causes:")
        logger.warning("    1. HAZARD_CONFIG_FILE environment variable not being used")
        logger.warning("    2. Age multiplier logic not integrated into Markov promotion")
        logger.warning("    3. Configuration files not being loaded correctly")
    
    return scenario_results

def main():
    """Main analysis function."""
    try:
        results = compare_scenarios()
        
        logger.info(f"\n💡 RECOMMENDATIONS")
        logger.info("-" * 40)
        logger.info("1. Check promotion engine logs for age multiplier application messages")
        logger.info("2. Verify HAZARD_CONFIG_FILE environment variable is being used")
        logger.info("3. Examine individual promotion events for age-based patterns")
        logger.info("4. Consider increasing age multiplier values for clearer test signals")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
