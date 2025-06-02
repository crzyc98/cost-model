#!/usr/bin/env python3
"""
Comprehensive test script for User Story 1.4: Promotion Age Sensitivity Verification.

This script runs targeted simulations with different promotion age multiplier configurations
to verify that the promotion engine correctly applies age sensitivity.

Test Scenarios:
1. P-Baseline: All promotion age multipliers = 1.0 (neutral control)
2. P-High Early: <30 age group with very high multiplier (2.5x)
3. P-High Mid: 30-39 age group with very high multiplier (2.5x)
4. P-Low Late: 50-59 and 60-65 age groups with very low multipliers (0.1x, 0.05x)
"""

import os
import sys
import logging
import subprocess
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

def run_simulation_scenario(scenario_name, hazard_config_file, output_dir):
    """Run a single simulation scenario with the specified hazard configuration."""
    logger.info(f"🚀 Running scenario: {scenario_name}")
    logger.info(f"   Hazard config: {hazard_config_file}")
    logger.info(f"   Output dir: {output_dir}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set up environment variables
    env = os.environ.copy()
    env['HAZARD_CONFIG_FILE'] = hazard_config_file

    # Run the simulation using the correct CLI format
    cmd = [
        sys.executable, '-m', 'cost_model.projections.cli',
        '--config', 'config/test_age_sensitivity_baseline.yaml',
        '--census', 'data/census_template.parquet',
        '--output-dir', output_dir,
        '--scenario-name', scenario_name
    ]

    logger.info(f"   Command: {' '.join(cmd)}")
    logger.info(f"   Environment: HAZARD_CONFIG_FILE={hazard_config_file}")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            logger.info(f"✅ Scenario {scenario_name} completed successfully")
            return True
        else:
            logger.error(f"❌ Scenario {scenario_name} failed with return code {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"❌ Scenario {scenario_name} timed out after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"❌ Scenario {scenario_name} failed with exception: {e}")
        return False

def analyze_promotion_results(scenarios_results):
    """Analyze promotion results across all scenarios."""
    logger.info("📊 Analyzing promotion results across scenarios...")
    
    analysis_results = {}
    
    for scenario_name, output_dir in scenarios_results.items():
        logger.info(f"   Analyzing scenario: {scenario_name}")
        
        # Look for consolidated snapshots file
        snapshots_file = os.path.join(output_dir, 'consolidated_snapshots.parquet')
        if not os.path.exists(snapshots_file):
            logger.warning(f"   ⚠️  No consolidated snapshots found for {scenario_name}")
            continue
            
        try:
            # Load the snapshots data
            df = pd.read_parquet(snapshots_file)
            logger.info(f"   Loaded {len(df)} snapshot records")
            
            # Calculate promotion metrics by year and age band
            scenario_analysis = {}
            
            for year in sorted(df['year'].unique()):
                year_data = df[df['year'] == year].copy()
                
                if 'employee_age_band' in year_data.columns:
                    # Group by age band and calculate promotion rates
                    age_analysis = {}
                    for age_band in year_data['employee_age_band'].unique():
                        if pd.isna(age_band):
                            continue
                            
                        age_group_data = year_data[year_data['employee_age_band'] == age_band]
                        
                        # Calculate basic metrics
                        total_employees = len(age_group_data)
                        avg_level = age_group_data['employee_level'].mean() if total_employees > 0 else 0
                        
                        age_analysis[age_band] = {
                            'total_employees': total_employees,
                            'avg_level': avg_level
                        }
                    
                    scenario_analysis[year] = age_analysis
                else:
                    logger.warning(f"   ⚠️  No age band data found for {scenario_name} year {year}")
            
            analysis_results[scenario_name] = scenario_analysis
            
        except Exception as e:
            logger.error(f"   ❌ Failed to analyze {scenario_name}: {e}")
            continue
    
    return analysis_results

def print_promotion_analysis(analysis_results):
    """Print a comprehensive analysis of promotion results."""
    logger.info("📈 PROMOTION AGE SENSITIVITY VERIFICATION RESULTS")
    logger.info("=" * 80)
    
    if not analysis_results:
        logger.error("❌ No analysis results available")
        return
    
    # Print summary table
    logger.info("\n📊 SCENARIO SUMMARY")
    logger.info("-" * 60)
    logger.info(f"{'Scenario':<25} {'Years':<10} {'Status'}")
    logger.info("-" * 60)
    
    for scenario_name, data in analysis_results.items():
        years_analyzed = len(data) if data else 0
        status = "✅ ANALYZED" if years_analyzed > 0 else "❌ NO DATA"
        logger.info(f"{scenario_name:<25} {years_analyzed:<10} {status}")
    
    # Print detailed analysis for each scenario
    for scenario_name, scenario_data in analysis_results.items():
        if not scenario_data:
            continue
            
        logger.info(f"\n🔍 DETAILED ANALYSIS: {scenario_name}")
        logger.info("-" * 50)
        
        for year, year_data in scenario_data.items():
            logger.info(f"\n  Year {year}:")
            
            if year_data:
                for age_band, metrics in year_data.items():
                    total = metrics['total_employees']
                    avg_level = metrics['avg_level']
                    logger.info(f"    {age_band:<8}: {total:>3} employees, avg level {avg_level:.2f}")
            else:
                logger.info("    No age band data available")
    
    # Print comparative analysis
    logger.info(f"\n🎯 COMPARATIVE ANALYSIS")
    logger.info("-" * 50)
    
    # Compare average levels across scenarios for the final year
    final_year = 2029
    logger.info(f"\nAverage employee levels by age band in {final_year}:")
    logger.info(f"{'Age Band':<10} {'Baseline':<10} {'High Early':<12} {'High Mid':<10} {'Low Late':<10}")
    logger.info("-" * 60)
    
    # Get all age bands from all scenarios
    all_age_bands = set()
    for scenario_data in analysis_results.values():
        if final_year in scenario_data:
            all_age_bands.update(scenario_data[final_year].keys())
    
    for age_band in sorted(all_age_bands):
        row = f"{age_band:<10}"
        
        for scenario in ['promotion_baseline', 'promotion_high_early', 'promotion_high_mid', 'promotion_low_late']:
            if scenario in analysis_results and final_year in analysis_results[scenario]:
                if age_band in analysis_results[scenario][final_year]:
                    avg_level = analysis_results[scenario][final_year][age_band]['avg_level']
                    row += f" {avg_level:<9.2f}"
                else:
                    row += f" {'N/A':<9}"
            else:
                row += f" {'N/A':<9}"
        
        logger.info(row)

def main():
    """Main function to run all promotion age sensitivity verification tests."""
    logger.info("🎯 STARTING PROMOTION AGE SENSITIVITY VERIFICATION")
    logger.info("=" * 80)
    
    # Define test scenarios
    scenarios = {
        'promotion_baseline': {
            'name': 'P-Baseline (All 1.0)',
            'config': 'hazard_promotion_baseline_test.yaml',  # Just filename, not full path
            'output_dir': 'output_dev/promotion_age_test/baseline'
        },
        'promotion_high_early': {
            'name': 'P-High Early (<30 = 2.5x)',
            'config': 'hazard_promotion_high_early_test.yaml',  # Just filename, not full path
            'output_dir': 'output_dev/promotion_age_test/high_early'
        },
        'promotion_high_mid': {
            'name': 'P-High Mid (30-39 = 2.5x)',
            'config': 'hazard_promotion_high_mid_test.yaml',  # Just filename, not full path
            'output_dir': 'output_dev/promotion_age_test/high_mid'
        },
        'promotion_low_late': {
            'name': 'P-Low Late (50+ = 0.1x)',
            'config': 'hazard_promotion_low_late_test.yaml',  # Just filename, not full path
            'output_dir': 'output_dev/promotion_age_test/low_late'
        }
    }
    
    # Run all scenarios
    successful_scenarios = {}
    failed_scenarios = []
    
    for scenario_id, scenario_config in scenarios.items():
        success = run_simulation_scenario(
            scenario_config['name'],
            scenario_config['config'],
            scenario_config['output_dir']
        )
        
        if success:
            successful_scenarios[scenario_id] = scenario_config['output_dir']
        else:
            failed_scenarios.append(scenario_id)
    
    # Report simulation results
    logger.info(f"\n📋 SIMULATION RESULTS SUMMARY")
    logger.info("-" * 40)
    logger.info(f"✅ Successful scenarios: {len(successful_scenarios)}")
    logger.info(f"❌ Failed scenarios: {len(failed_scenarios)}")
    
    if failed_scenarios:
        logger.warning(f"Failed scenarios: {', '.join(failed_scenarios)}")
    
    # Analyze results if we have successful scenarios
    if successful_scenarios:
        analysis_results = analyze_promotion_results(successful_scenarios)
        print_promotion_analysis(analysis_results)
        
        logger.info(f"\n🎯 VERIFICATION COMPLETE")
        logger.info("=" * 40)
        logger.info("✅ Promotion age sensitivity verification completed")
        logger.info(f"📁 Results saved in: output_dev/promotion_age_test/")
        logger.info("📊 Review the analysis above for directional impact verification")
        
        return len(failed_scenarios) == 0
    else:
        logger.error("❌ No successful scenarios - cannot perform verification")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
