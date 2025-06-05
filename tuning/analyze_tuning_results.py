#!/usr/bin/env python3
"""
Analyze auto-tuning results for Epic 3 User Story 3.3.

This script provides detailed analysis of tuning campaign results to understand:
1. Why the tuner made specific trade-offs
2. Which parameters had the most impact
3. How to refine weights and search spaces for better results

Usage:
    python tuning/analyze_tuning_results.py [--results-file tuned/tuning_results.json]
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd

def load_tuning_results(results_file: Path) -> List[Dict[str, Any]]:
    """Load tuning results from JSON file."""
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        return json.load(f)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def analyze_score_components(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Analyze score components across all configurations."""
    data = []
    
    for result in results:
        summary = result.get('summary', {})
        config_path = result.get('config_path', '')
        
        # Extract key metrics
        row = {
            'config': Path(config_path).stem,
            'total_score': result.get('score', float('inf')),
            'hc_growth': summary.get('hc_growth', 0.0),
            'pay_growth': summary.get('pay_growth', 0.0),
            'age_hist': summary.get('age_hist', {}),
            'tenure_hist': summary.get('tenure_hist', {}),
            'final_headcount': summary.get('final_headcount', 0),
            'config_path': config_path
        }
        
        # Calculate individual error components (matching tune_configs.py logic)
        TARGET_HC_GROWTH = 0.03
        TARGET_PAY_GROWTH = 0.03
        
        row['hc_growth_err'] = abs(row['hc_growth'] - TARGET_HC_GROWTH)
        row['pay_growth_err'] = abs(row['pay_growth'] - TARGET_PAY_GROWTH)
        
        data.append(row)
    
    return pd.DataFrame(data)

def analyze_best_config(best_result: Dict[str, Any]) -> Dict[str, Any]:
    """Detailed analysis of the best configuration."""
    config_path = best_result['config_path']
    config = load_config(config_path)
    summary = best_result['summary']
    
    analysis = {
        'config_path': config_path,
        'score': best_result['score'],
        'performance': {
            'hc_growth': {
                'achieved': summary.get('hc_growth', 0.0),
                'target': 0.03,
                'error': abs(summary.get('hc_growth', 0.0) - 0.03)
            },
            'pay_growth': {
                'achieved': summary.get('pay_growth', 0.0),
                'target': 0.03,
                'error': abs(summary.get('pay_growth', 0.0) - 0.03)
            },
            'age_distribution': summary.get('age_hist', {}),
            'tenure_distribution': summary.get('tenure_hist', {})
        },
        'key_parameters': extract_key_parameters(config)
    }
    
    return analysis

def extract_key_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key parameters from configuration for analysis."""
    global_params = config.get('global_parameters', {})
    
    key_params = {
        # Growth and hiring
        'target_growth': global_params.get('target_growth'),
        'new_hire_rate': global_params.get('new_hires', {}).get('new_hire_rate'),
        'new_hire_average_age': global_params.get('new_hire_average_age'),
        'max_working_age': global_params.get('max_working_age'),
        
        # Termination
        'base_rate_for_new_hire': global_params.get('termination_hazard', {}).get('base_rate_for_new_hire'),
        'term_age_mult_under_30': global_params.get('termination_hazard', {}).get('age_multipliers', {}).get('<30'),
        'term_tenure_mult_under_1': global_params.get('termination_hazard', {}).get('tenure_multipliers', {}).get('<1'),
        
        # Compensation
        'annual_compensation_increase_rate': global_params.get('annual_compensation_increase_rate'),
        'merit_base': global_params.get('raises_hazard', {}).get('merit_base'),
        'cola_2025': global_params.get('cola_hazard', {}).get('by_year', {}).get('2025'),
    }
    
    return {k: v for k, v in key_params.items() if v is not None}

def print_analysis_report(results: List[Dict[str, Any]]):
    """Print comprehensive analysis report."""
    if not results:
        print("No results to analyze!")
        return
    
    # Sort by score (best first)
    results_sorted = sorted(results, key=lambda x: x.get('score', float('inf')))
    best_result = results_sorted[0]
    
    print("=" * 80)
    print("AUTO-TUNING RESULTS ANALYSIS - Epic 3 User Story 3.3")
    print("=" * 80)
    
    # Overall campaign summary
    print(f"\nCAMPAIGN SUMMARY:")
    print(f"Total configurations tested: {len(results)}")
    print(f"Best score achieved: {best_result.get('score', 'N/A'):.4f}")
    print(f"Score range: {best_result.get('score', 0):.4f} - {results_sorted[-1].get('score', 0):.4f}")
    
    # Best configuration analysis
    print(f"\nBEST CONFIGURATION ANALYSIS:")
    best_analysis = analyze_best_config(best_result)
    
    perf = best_analysis['performance']
    print(f"Configuration: {Path(best_analysis['config_path']).name}")
    
    print(f"\nPERFORMANCE vs TARGETS:")
    print(f"  Headcount Growth: {perf['hc_growth']['achieved']:.1%} vs {perf['hc_growth']['target']:.1%} target")
    print(f"    → Error: {perf['hc_growth']['error']:.1%} ({'MISS' if perf['hc_growth']['error'] > 0.005 else 'OK'})")
    
    print(f"  Pay Growth: {perf['pay_growth']['achieved']:.1%} vs {perf['pay_growth']['target']:.1%} target")
    print(f"    → Error: {perf['pay_growth']['error']:.1%} ({'OVERSHOOT' if perf['pay_growth']['achieved'] > perf['pay_growth']['target'] else 'OK'})")
    
    # Age distribution analysis
    age_dist = perf['age_distribution']
    if age_dist:
        print(f"\nAGE DISTRIBUTION:")
        target_age = {"<30": 0.109, "30-39": 0.210, "40-49": 0.336, "50-59": 0.210, "60-65": 0.050, "65+": 0.084}
        for band in ["<30", "30-39", "40-49", "50-59", "60-65", "65+"]:
            achieved = age_dist.get(band, 0.0)
            target = target_age.get(band, 0.0)
            error = abs(achieved - target)
            status = "MAJOR MISS" if error > 0.05 else "MISS" if error > 0.02 else "OK"
            print(f"  {band:>5}: {achieved:.1%} vs {target:.1%} target → {error:.1%} error ({status})")
    
    # Tenure distribution analysis
    tenure_dist = perf['tenure_distribution']
    if tenure_dist:
        print(f"\nTENURE DISTRIBUTION:")
        target_tenure = {"<1": 0.20, "1-3": 0.30, "3-5": 0.25, "5-10": 0.15, "10-15": 0.07, "15+": 0.03}
        for band in ["<1", "1-3", "3-5", "5-10", "10-15", "15+"]:
            achieved = tenure_dist.get(band, 0.0)
            target = target_tenure.get(band, 0.0)
            error = abs(achieved - target)
            status = "MAJOR MISS" if error > 0.1 else "MISS" if error > 0.05 else "OK"
            print(f"  {band:>5}: {achieved:.1%} vs {target:.1%} target → {error:.1%} error ({status})")
    
    # Key parameters
    print(f"\nKEY OPTIMIZED PARAMETERS:")
    key_params = best_analysis['key_parameters']
    for param, value in key_params.items():
        print(f"  {param}: {value}")

def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Analyze auto-tuning results")
    parser.add_argument("--results-file", type=str, default="test_tuning/tuning_results.json",
                       help="Path to tuning results JSON file")
    
    args = parser.parse_args()
    
    results_file = Path(args.results_file)
    
    try:
        results = load_tuning_results(results_file)
        print_analysis_report(results)
        
        # Generate recommendations
        print(f"\nSTRATEGIC RECOMMENDATIONS FOR NEXT CAMPAIGN:")
        print(f"1. Score Weight Adjustments:")
        print(f"   - Current weights appear to need rebalancing based on results")
        print(f"   - Consider increasing HC_GROWTH and TENURE weights")
        print(f"   - Consider decreasing AGE weight slightly for more flexibility")
        
        print(f"\n2. Search Space Refinements:")
        print(f"   - Increase minimum new_hire_rate ranges for better growth")
        print(f"   - Lower termination rates for <1 year tenure employees")
        print(f"   - Constrain compensation parameters to control pay growth")
        print(f"   - Focus new hire age ranges on younger demographics")
        
        print(f"\n3. Next Steps:")
        print(f"   - Run refined campaign with updated weights and search space")
        print(f"   - Consider 100-200 iterations for comprehensive search")
        print(f"   - Monitor specific error components during tuning")
        
    except Exception as e:
        print(f"Error analyzing results: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
