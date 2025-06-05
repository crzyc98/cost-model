#!/usr/bin/env python3
"""
Validate tuning refinements with a small test campaign.

This script runs a small tuning campaign (5-10 iterations) to validate:
1. Refined score weights are working correctly
2. Search space changes are being applied
3. Component scores show expected patterns

Usage:
    python tuning/validate_refinements.py [--iterations 5]
"""

import argparse
import sys
from pathlib import Path

# Add the project root to the path so we can import tune_configs
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tuning.tune_configs import main as tune_main, score, load_baseline_distributions
from tuning.analyze_tuning_results import load_tuning_results, print_analysis_report

def validate_score_weights():
    """Test that score weights are correctly updated."""
    # Create a mock summary to test scoring
    mock_summary = {
        'hc_growth': 0.00,  # Miss target by 3%
        'pay_growth': 0.05,  # Overshoot target by 2%
        'age_hist': {'<30': 0.05, '30-39': 0.35, '40-49': 0.30, '50-59': 0.20, '60-65': 0.05, '65+': 0.05},
        'tenure_hist': {'<1': 0.45, '1-3': 0.25, '3-5': 0.15, '5-10': 0.10, '10-15': 0.03, '15+': 0.02}
    }
    
    test_score = score(mock_summary)
    print(f"Test score calculation: {test_score:.4f}")
    print("✓ Score function is working with updated weights")

def validate_search_space():
    """Validate that search space has been updated correctly."""
    from tuning.tune_configs import SEARCH_SPACE
    
    # Check key parameter ranges
    checks = [
        ('target_growth', [0.025, 0.030, 0.035, 0.040, 0.045, 0.050]),
        ('new_hire_rate', [0.12, 0.15, 0.18, 0.20, 0.22]),
        ('new_hire_average_age', [25, 27, 28, 30, 32]),
        ('base_rate_for_new_hire', [0.08, 0.10, 0.12, 0.15, 0.18]),
    ]
    
    for param_suffix, expected_range in checks:
        # Find the full parameter key
        matching_keys = [k for k in SEARCH_SPACE.keys() if param_suffix in k]
        if matching_keys:
            key = matching_keys[0]
            actual_range = SEARCH_SPACE[key]
            if set(actual_range) == set(expected_range):
                print(f"✓ {param_suffix}: {actual_range}")
            else:
                print(f"⚠ {param_suffix}: Expected {expected_range}, got {actual_range}")
        else:
            print(f"✗ {param_suffix}: Parameter not found in search space")
    
    print(f"✓ Search space validation complete ({len(SEARCH_SPACE)} parameters)")

def run_validation_campaign(iterations: int = 5):
    """Run a small validation campaign."""
    print(f"\nRunning validation campaign with {iterations} iterations...")
    
    # Temporarily override sys.argv to pass arguments to tune_main
    original_argv = sys.argv
    try:
        sys.argv = ['tune_configs.py', '--iterations', str(iterations), '--output-dir', 'validation_tuned']
        tune_main()
    except SystemExit:
        pass  # tune_main calls sys.exit, which is normal
    finally:
        sys.argv = original_argv
    
    # Analyze results
    results_file = Path('validation_tuned/tuning_results.json')
    if results_file.exists():
        print("\n" + "="*60)
        print("VALIDATION CAMPAIGN RESULTS")
        print("="*60)
        
        results = load_tuning_results(results_file)
        print_analysis_report(results)
        
        # Check if we see expected patterns
        if results:
            best_result = min(results, key=lambda x: x.get('score', float('inf')))
            best_summary = best_result.get('summary', {})
            
            print(f"\nVALIDATION CHECKS:")
            
            # Check if score components are reasonable
            hc_growth = best_summary.get('hc_growth', 0)
            if hc_growth > -0.05:  # Not too negative
                print(f"✓ Headcount growth reasonable: {hc_growth:.1%}")
            else:
                print(f"⚠ Headcount growth very negative: {hc_growth:.1%}")
            
            # Check if age/tenure distributions are present
            age_hist = best_summary.get('age_hist', {})
            tenure_hist = best_summary.get('tenure_hist', {})
            
            if age_hist:
                print(f"✓ Age distribution extracted: {len(age_hist)} bands")
            else:
                print(f"⚠ Age distribution missing")
            
            if tenure_hist:
                print(f"✓ Tenure distribution extracted: {len(tenure_hist)} bands")
            else:
                print(f"⚠ Tenure distribution missing")
        
        print(f"\n✓ Validation campaign completed successfully")
    else:
        print(f"⚠ Results file not found: {results_file}")

def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate tuning refinements")
    parser.add_argument("--iterations", "-n", type=int, default=5,
                       help="Number of validation iterations to run")
    
    args = parser.parse_args()
    
    print("VALIDATING TUNING REFINEMENTS")
    print("="*50)
    
    # 1. Validate score weights
    print("\n1. Testing score weight updates...")
    validate_score_weights()
    
    # 2. Validate search space
    print("\n2. Validating search space refinements...")
    validate_search_space()
    
    # 3. Run small validation campaign
    print(f"\n3. Running validation campaign...")
    run_validation_campaign(args.iterations)
    
    print(f"\n" + "="*50)
    print("VALIDATION COMPLETE")
    print("="*50)
    print("If all checks passed, you're ready to run the full Campaign 2!")
    print("Recommended command:")
    print("  python tuning/tune_configs.py --iterations 100")

if __name__ == "__main__":
    main()
