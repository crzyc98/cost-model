#!/usr/bin/env python3
"""
Quick test to verify campaign data structure and provide working analysis code.
"""

import pandas as pd
import json
from pathlib import Path

# Load campaign results
results_file = Path('../campaign_2_results/tuning_results.json')

with open(results_file, 'r') as f:
    results = json.load(f)

# Convert to DataFrame
campaign_df = pd.DataFrame(results)

# Extract summary metrics into separate columns
summary_df = pd.json_normalize(campaign_df['summary'])
campaign_df = pd.concat([campaign_df.drop('summary', axis=1), summary_df], axis=1)

print(f"✅ Loaded {len(campaign_df)} campaign results")
print(f"📊 Available columns: {list(campaign_df.columns)}")

# Check if age_hist and tenure_hist are available
if 'age_hist' in campaign_df.columns:
    print("✅ age_hist column found")
    print(f"Sample age_hist: {campaign_df['age_hist'].iloc[0]}")
else:
    print("❌ age_hist column not found")

if 'tenure_hist' in campaign_df.columns:
    print("✅ tenure_hist column found")
    print(f"Sample tenure_hist: {campaign_df['tenure_hist'].iloc[0]}")
else:
    print("❌ tenure_hist column not found")

# Get best configuration
best_config = campaign_df.loc[campaign_df['score'].idxmin()]
print(f"\n🏆 Best Configuration:")
print(f"   Config: {best_config['config_path']}")
print(f"   Score: {best_config['score']:.6f}")
print(f"   HC Growth: {best_config['hc_growth']*100:.2f}%")
print(f"   Pay Growth: {best_config['pay_growth']*100:.2f}%")

# Test demographics analysis
age_hist = best_config['age_hist']
tenure_hist = best_config['tenure_hist']

print(f"\n👥 Demographics Analysis:")
print(f"   Age hist type: {type(age_hist)}")
print(f"   Age hist: {age_hist}")
print(f"   Tenure hist type: {type(tenure_hist)}")
print(f"   Tenure hist: {tenure_hist}")

# Target distributions
target_age = {'<30': 0.109, '30-39': 0.210, '40-49': 0.336, '50-59': 0.210, '60-65': 0.050, '65+': 0.084}
target_tenure = {'<1': 0.20, '1-3': 0.30, '3-5': 0.25, '5-10': 0.15, '10-15': 0.07, '15+': 0.03}

print(f"\n🎂 Age Distribution Comparison:")
for band in target_age:
    actual = age_hist.get(band, 0) * 100
    target = target_age[band] * 100
    error = abs(actual - target)
    status = "✅" if error < 2 else "⚠️" if error < 5 else "❌"
    print(f"   {band}: {actual:.1f}% vs {target:.1f}% (error: {error:.1f}pp) {status}")

print(f"\n⏱️ Tenure Distribution Comparison:")
for band in target_tenure:
    actual = tenure_hist.get(band, 0) * 100
    target = target_tenure[band] * 100
    error = abs(actual - target)
    status = "✅" if error < 2 else "⚠️" if error < 5 else "❌"
    print(f"   {band}: {actual:.1f}% vs {target:.1f}% (error: {error:.1f}pp) {status}")

print("\n✅ Data structure test complete!")
