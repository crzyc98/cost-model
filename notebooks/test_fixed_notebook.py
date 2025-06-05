#!/usr/bin/env python3
"""
Test the fixed notebook data loading approach.
"""

import pandas as pd
import json
from pathlib import Path

def load_campaign_results(campaign_dir):
    """Load and process campaign results from tuning_results.json"""
    results_file = Path(campaign_dir) / 'tuning_results.json'
    
    if not results_file.exists():
        print(f"❌ Campaign results not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Convert to DataFrame for easier analysis
    campaign_df = pd.DataFrame(results)
    
    # Extract summary metrics into separate columns
    summary_df = pd.json_normalize(campaign_df['summary'])
    campaign_df = pd.concat([campaign_df.drop('summary', axis=1), summary_df], axis=1)
    
    print(f"✅ Loaded {len(campaign_df)} campaign results")
    print(f"📊 Available columns: {list(campaign_df.columns)}")
    return campaign_df

def test_demographics_analysis(campaign_results):
    """Test the fixed demographics analysis"""
    if campaign_results is None:
        print("❌ No campaign results loaded")
        return
    
    # Get best configuration
    config_data = campaign_results.loc[campaign_results['score'].idxmin()]
    
    # Extract age and tenure distributions from flattened columns
    age_hist = {}
    tenure_hist = {}
    
    # Reconstruct age_hist dictionary from flattened columns
    for col in config_data.index:
        if col.startswith('age_hist.'):
            band = col.replace('age_hist.', '')
            age_hist[band] = config_data[col]
        elif col.startswith('tenure_hist.'):
            band = col.replace('tenure_hist.', '')
            tenure_hist[band] = config_data[col]
    
    print(f"\n🏆 Best Configuration Analysis:")
    print(f"   Config: {config_data['config_path']}")
    print(f"   Score: {config_data['score']:.6f}")
    print(f"   HC Growth: {config_data['hc_growth']*100:.2f}%")
    print(f"   Pay Growth: {config_data['pay_growth']*100:.2f}%")
    
    print(f"\n🎂 Age Distribution:")
    print(f"   Reconstructed age_hist: {age_hist}")
    
    print(f"\n⏱️ Tenure Distribution:")
    print(f"   Reconstructed tenure_hist: {tenure_hist}")
    
    # Target distributions
    target_age = {'<30': 0.109, '30-39': 0.210, '40-49': 0.336, '50-59': 0.210, '60-65': 0.050, '65+': 0.084}
    target_tenure = {'<1': 0.20, '1-3': 0.30, '3-5': 0.25, '5-10': 0.15, '10-15': 0.07, '15+': 0.03}
    
    print(f"\n📊 Age Distribution Comparison:")
    for band in target_age:
        actual = age_hist.get(band, 0) * 100
        target = target_age[band] * 100
        error = abs(actual - target)
        status = "✅" if error < 2 else "⚠️" if error < 5 else "❌"
        print(f"   {band}: {actual:.1f}% vs {target:.1f}% (error: {error:.1f}pp) {status}")
    
    print(f"\n📊 Tenure Distribution Comparison:")
    for band in target_tenure:
        actual = tenure_hist.get(band, 0) * 100
        target = target_tenure[band] * 100
        error = abs(actual - target)
        status = "✅" if error < 2 else "⚠️" if error < 5 else "❌"
        print(f"   {band}: {actual:.1f}% vs {target:.1f}% (error: {error:.1f}pp) {status}")
    
    return True

if __name__ == "__main__":
    print("🔧 Testing Fixed Notebook Data Loading...")
    
    # Load campaign results
    campaign_results = load_campaign_results('campaign_2_results')
    
    # Test demographics analysis
    if campaign_results is not None:
        success = test_demographics_analysis(campaign_results)
        if success:
            print("\n✅ Fixed notebook approach works correctly!")
            print("🎯 The Jupyter notebook should now work without errors.")
        else:
            print("\n❌ Still having issues with the fix.")
    else:
        print("\n❌ Could not load campaign results.")
