#!/usr/bin/env python3
"""
Quick fix script to test and debug the notebook data loading issue.
This will help us understand the exact structure of the campaign results data.
"""

import pandas as pd
import json
from pathlib import Path

def test_campaign_data_loading():
    """Test loading campaign results and show the data structure"""
    
    # Load the campaign results
    results_file = Path('../campaign_2_results/tuning_results.json')
    
    if not results_file.exists():
        print(f"❌ Campaign results not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"📊 Raw data structure:")
    print(f"   Type: {type(results)}")
    print(f"   Length: {len(results)}")
    print(f"   First item keys: {list(results[0].keys())}")
    print(f"   Summary keys: {list(results[0]['summary'].keys())}")
    
    # Convert to DataFrame
    campaign_df = pd.DataFrame(results)
    print(f"\n📈 Initial DataFrame:")
    print(f"   Shape: {campaign_df.shape}")
    print(f"   Columns: {list(campaign_df.columns)}")
    
    # Extract summary metrics
    summary_df = pd.json_normalize(campaign_df['summary'])
    print(f"\n📋 Summary DataFrame:")
    print(f"   Shape: {summary_df.shape}")
    print(f"   Columns: {list(summary_df.columns)}")
    
    # Combine DataFrames
    final_df = pd.concat([campaign_df.drop('summary', axis=1), summary_df], axis=1)
    print(f"\n✅ Final DataFrame:")
    print(f"   Shape: {final_df.shape}")
    print(f"   Columns: {list(final_df.columns)}")
    
    # Check for age_hist and tenure_hist
    if 'age_hist' in final_df.columns:
        print(f"\n🎂 Age hist sample: {final_df['age_hist'].iloc[0]}")
    else:
        print(f"\n❌ age_hist not found in columns")
    
    if 'tenure_hist' in final_df.columns:
        print(f"\n⏱️ Tenure hist sample: {final_df['tenure_hist'].iloc[0]}")
    else:
        print(f"\n❌ tenure_hist not found in columns")
    
    # Show best configuration
    best_idx = final_df['score'].idxmin()
    best_config = final_df.loc[best_idx]
    print(f"\n🏆 Best Configuration:")
    print(f"   Config: {best_config['config_path']}")
    print(f"   Score: {best_config['score']:.6f}")
    print(f"   HC Growth: {best_config['hc_growth']*100:.2f}%")
    print(f"   Pay Growth: {best_config['pay_growth']*100:.2f}%")
    
    return final_df

def test_demographics_analysis(campaign_df):
    """Test the demographics analysis function"""
    
    if campaign_df is None:
        print("❌ No campaign data to analyze")
        return
    
    # Get best configuration
    best_config = campaign_df.loc[campaign_df['score'].idxmin()]
    
    print(f"\n👥 Demographics Analysis Test:")
    print(f"   Config: {best_config['config_path']}")
    
    # Check age_hist structure
    if 'age_hist' in best_config:
        age_hist = best_config['age_hist']
        print(f"   Age hist type: {type(age_hist)}")
        print(f"   Age hist: {age_hist}")
        
        # Target distributions
        target_age = {
            '<30': 0.109, '30-39': 0.210, '40-49': 0.336, 
            '50-59': 0.210, '60-65': 0.050, '65+': 0.084
        }
        
        print(f"\n   Age Distribution Comparison:")
        for band in target_age:
            actual = age_hist.get(band, 0) * 100
            target = target_age[band] * 100
            error = abs(actual - target)
            status = "✅" if error < 2 else "⚠️" if error < 5 else "❌"
            print(f"     {band}: {actual:.1f}% vs {target:.1f}% (error: {error:.1f}pp) {status}")
    
    # Check tenure_hist structure
    if 'tenure_hist' in best_config:
        tenure_hist = best_config['tenure_hist']
        print(f"\n   Tenure hist type: {type(tenure_hist)}")
        print(f"   Tenure hist: {tenure_hist}")
        
        # Target distributions
        target_tenure = {
            '<1': 0.20, '1-3': 0.30, '3-5': 0.25, 
            '5-10': 0.15, '10-15': 0.07, '15+': 0.03
        }
        
        print(f"\n   Tenure Distribution Comparison:")
        for band in target_tenure:
            actual = tenure_hist.get(band, 0) * 100
            target = target_tenure[band] * 100
            error = abs(actual - target)
            status = "✅" if error < 2 else "⚠️" if error < 5 else "❌"
            print(f"     {band}: {actual:.1f}% vs {target:.1f}% (error: {error:.1f}pp) {status}")

if __name__ == "__main__":
    print("🔧 Testing Campaign Data Loading...")
    campaign_data = test_campaign_data_loading()
    
    print("\n" + "="*60)
    test_demographics_analysis(campaign_data)
    
    print("\n" + "="*60)
    print("✅ Testing complete! Use this information to fix the notebook.")
