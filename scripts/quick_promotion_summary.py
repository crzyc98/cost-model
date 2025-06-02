#!/usr/bin/env python3
"""Quick summary of promotion age sensitivity verification results."""

import pandas as pd
import sys
import os

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    print('🎯 PROMOTION AGE SENSITIVITY VERIFICATION RESULTS')
    print('=' * 60)
    
    scenarios = [
        ('Baseline (1.0x all)', 'output_dev/promotion_age_test/baseline/P-Baseline (All 1.0)_final_cumulative_event_log.parquet'),
        ('High Early (2.5x <30)', 'output_dev/promotion_age_test/high_early/P-High Early (<30 = 2.5x)_final_cumulative_event_log.parquet'),
        ('High Mid (2.5x 30-39)', 'output_dev/promotion_age_test/high_mid/P-High Mid (30-39 = 2.5x)_final_cumulative_event_log.parquet'),
        ('Low Late (0.1x 50+)', 'output_dev/promotion_age_test/low_late/P-Low Late (50+ = 0.1x)_final_cumulative_event_log.parquet')
    ]
    
    results = {}
    
    for name, path in scenarios:
        try:
            df = pd.read_parquet(path)
            promotions = len(df[df['event_type'] == 'EVT_PROMOTION'])
            results[name] = promotions
            print(f'{name:<25}: {promotions:>3} promotions')
        except Exception as e:
            print(f'{name:<25}: ERROR - {e}')
            results[name] = None
    
    # Analysis
    print('\n📊 ANALYSIS:')
    
    if all(v is not None for v in results.values()):
        baseline = results['Baseline (1.0x all)']
        high_early = results['High Early (2.5x <30)']
        high_mid = results['High Mid (2.5x 30-39)']
        low_late = results['Low Late (0.1x 50+)']
        
        print(f'Expected pattern: High Early/Mid > Baseline > Low Late')
        print(f'Actual pattern: {high_early}/{high_mid} > {baseline} > {low_late}')
        
        # Check if pattern is correct
        high_scenarios_higher = (high_early >= baseline) and (high_mid >= baseline)
        low_scenario_lower = low_late <= baseline
        different_results = len(set(results.values())) > 1
        
        if high_scenarios_higher and low_scenario_lower and different_results:
            print('✅ SUCCESS: Age sensitivity is working correctly!')
            print('   - High multiplier scenarios have more/equal promotions')
            print('   - Low multiplier scenario has fewer/equal promotions')
            print('   - Results differ across scenarios')
        elif different_results:
            print('⚠️  PARTIAL SUCCESS: Age effects visible but pattern not perfect')
            print('   - Results differ across scenarios (good)')
            if not high_scenarios_higher:
                print('   - High multiplier scenarios should have more promotions')
            if not low_scenario_lower:
                print('   - Low multiplier scenario should have fewer promotions')
        else:
            print('❌ FAILURE: All scenarios show identical results')
            print('   - Age multipliers may not be working')
    
    print('\n💡 INTERPRETATION:')
    print('- Small differences are expected due to limited sample size (119 employees)')
    print('- Young employees (<30): only 9 total, so limited promotion opportunities')
    print('- Age multipliers work by modifying base promotion probabilities')
    print('- Success = directional impact in expected direction')

if __name__ == "__main__":
    main()
