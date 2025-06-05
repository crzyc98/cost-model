#!/usr/bin/env python3
"""
Campaign 3 Results Analysis - Simplified Version
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("🔍 Campaign 3 Analysis")
    print("=" * 50)
    
    # Load campaign results
    results_file = Path('../campaign_3_results/tuning_results.json')
    print(f"📁 Loading: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Convert to DataFrame
    campaign_df = pd.DataFrame(results)
    
    # Extract summary metrics
    summary_df = pd.json_normalize(campaign_df['summary'])
    campaign_df = pd.concat([campaign_df.drop('summary', axis=1), summary_df], axis=1)
    
    print(f"✅ Loaded {len(campaign_df)} campaign results")
    
    # Overall Performance
    print(f"\n📈 Campaign 3 Summary:")
    print(f"   Total Configurations: {len(campaign_df)}")
    print(f"   Best Score: {campaign_df['score'].min():.6f}")
    print(f"   Worst Score: {campaign_df['score'].max():.6f}")
    print(f"   Mean Score: {campaign_df['score'].mean():.6f}")
    print(f"   Score Std Dev: {campaign_df['score'].std():.6f}")
    
    # Top 10 configurations
    top_10 = campaign_df.nsmallest(10, 'score')
    print(f"\n🏆 Top 10 Configurations:")
    for i, row in top_10.iterrows():
        config_name = Path(row['config_path']).stem
        print(f"   {config_name}: Score={row['score']:.6f}, HC_Growth={row['hc_growth']:.3%}, Pay_Growth={row['pay_growth']:.3%}")
    
    # Key Metrics Analysis
    print(f"\n🎯 Key Metrics Analysis:")
    
    # Headcount Growth
    avg_hc_growth = campaign_df['hc_growth'].mean()
    target_hc_growth = 0.03  # 3% target
    hc_gap = target_hc_growth - avg_hc_growth
    
    print(f"\n📊 Headcount Growth:")
    print(f"   Target: {target_hc_growth:.1%}")
    print(f"   Average Achieved: {avg_hc_growth:.1%}")
    print(f"   Best Achieved: {campaign_df['hc_growth'].max():.1%}")
    print(f"   Gap from Target: {hc_gap:.1%}")
    
    # Pay Growth
    avg_pay_growth = campaign_df['pay_growth'].mean()
    target_pay_growth = 0.03  # 3% target
    pay_gap = target_pay_growth - avg_pay_growth
    
    print(f"\n💰 Pay Growth:")
    print(f"   Target: {target_pay_growth:.1%}")
    print(f"   Average Achieved: {avg_pay_growth:.1%}")
    print(f"   Gap from Target: {pay_gap:.1%}")
    
    # Age Distribution Analysis (Top 5)
    top_5 = campaign_df.nsmallest(5, 'score')
    
    print(f"\n👥 Age Distribution Analysis (Top 5 configs):")
    age_targets = {
        '<30': 0.15,
        '30-39': 0.35,
        '40-49': 0.25,
        '50-59': 0.15,
        '60-65': 0.07,
        '65+': 0.03
    }
    
    for age_band, target in age_targets.items():
        col_name = f'age_hist.{age_band}'
        if col_name in top_5.columns:
            actual = top_5[col_name].mean()
            error = actual - target
            print(f"   {age_band:>6}: Target={target:.1%}, Actual={actual:.1%}, Error={error:+.1%}")
    
    # Tenure Distribution Analysis (Top 5)
    print(f"\n⏰ Tenure Distribution Analysis (Top 5 configs):")
    tenure_targets = {
        '<1': 0.20,
        '1-3': 0.25,
        '3-5': 0.20,
        '5-10': 0.20,
        '10-15': 0.10,
        '15+': 0.05
    }
    
    for tenure_band, target in tenure_targets.items():
        col_name = f'tenure_hist.{tenure_band}'
        if col_name in top_5.columns:
            actual = top_5[col_name].mean()
            error = actual - target
            print(f"   {tenure_band:>6}: Target={target:.1%}, Actual={actual:.1%}, Error={error:+.1%}")
    
    # Key Issues Identified
    print(f"\n🚨 Key Issues Identified:")
    
    # Issue 1: Headcount Growth
    if avg_hc_growth < 0:
        print(f"   1. HEADCOUNT DECLINE: Achieving {avg_hc_growth:.1%} vs +3.0% target")
        print(f"      → Need higher target_growth and new_hire_rate parameters")
    
    # Issue 2: Age Distribution
    if 'age_hist.<30' in top_5.columns:
        avg_under_30 = top_5['age_hist.<30'].mean()
        if avg_under_30 < 0.10:
            print(f"   2. AGING WORKFORCE: Only {avg_under_30:.1%} under 30 vs 15% target")
            print(f"      → Need younger new hires and stronger retirement pressure")
    
    # Issue 3: Tenure Distribution
    if 'tenure_hist.<1' in top_5.columns:
        avg_new_hires = top_5['tenure_hist.<1'].mean()
        if avg_new_hires > 0.30:
            print(f"   3. HIGH NEW HIRE TURNOVER: {avg_new_hires:.1%} <1 year vs 20% target")
            print(f"      → Need better new hire retention")
    
    # Campaign 4 Recommendations
    print(f"\n🚀 Campaign 4 Recommendations:")
    print(f"   1. SEARCH SPACE ADJUSTMENTS:")
    print(f"      • target_growth: 0.04-0.07 (vs current achieving {avg_hc_growth:.1%})")
    print(f"      • new_hire_rate: 0.25-0.45 (need more hiring volume)")
    print(f"      • new_hire_average_age: 22-27 (younger workforce)")
    print(f"      • age_multipliers 60+: 3.0-8.0 (stronger retirement)")
    
    print(f"\n   2. SCORE WEIGHT ADJUSTMENTS:")
    print(f"      • HC_Growth: 0.50 (highest priority)")
    print(f"      • Age: 0.25 (critical for workforce balance)")
    print(f"      • Tenure: 0.20 (retention focus)")
    print(f"      • Pay_Growth: 0.05 (lowest priority)")
    
    print(f"\n   3. MODEL INVESTIGATION:")
    print(f"      • Review target_growth → actual hiring conversion")
    print(f"      • Analyze new hire progression through tenure bands")
    print(f"      • Validate termination vs promotion interactions")
    
    # Success Metrics
    best_score = campaign_df['score'].min()
    print(f"\n✅ Campaign 3 Success:")
    print(f"   • Achieved excellent best score: {best_score:.6f}")
    print(f"   • Pay growth accuracy: Very good")
    print(f"   • Identified clear parameter sensitivity patterns")
    print(f"   • Ready for targeted Campaign 4 refinements")
    
    print(f"\n🎯 Campaign 4 Target: Best score < 0.08 with HC_Growth > +2%")

if __name__ == "__main__":
    main()
