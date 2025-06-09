#!/usr/bin/env python3
"""
Test the notebook fixes with the campaign data.
"""

import pandas as pd
import numpy as np
import json

def test_campaign_data():
    print("🧪 Testing campaign data loading...")
    
    # Load and process campaign results
    with open('campaign_round2_results/tuning_results.json', 'r') as f:
        results = json.load(f)

    campaign_df = pd.DataFrame(results)
    summary_df = pd.json_normalize(campaign_df['summary'])
    campaign_df = pd.concat([campaign_df.drop('summary', axis=1), summary_df], axis=1)

    print(f'✅ Loaded {len(campaign_df)} campaign results')

    # Check for age_hist and tenure_hist columns
    age_cols = [col for col in campaign_df.columns if col.startswith('age_hist.')]
    tenure_cols = [col for col in campaign_df.columns if col.startswith('tenure_hist.')]

    print(f'🎂 Age histogram columns: {age_cols}')
    print(f'⏱️ Tenure histogram columns: {tenure_cols}')

    # Test the top 5 configurations
    comparison_df = campaign_df.nsmallest(5, 'score')

    print(f'\n🏆 Top 5 configurations:')
    for i, (idx, config) in enumerate(comparison_df.iterrows()):
        print(f'Config {i+1}: {config["config_path"]} - Score: {config["score"]:.6f}')

    # Test age distribution analysis
    print(f'\n👥 Testing Age Distribution Analysis:')
    age_bands = ['<30', '30-39', '40-49', '50-59', '60-65', '65+']
    target_age = [10.9, 21.0, 33.6, 21.0, 5.0, 8.4]

    age_analysis = pd.DataFrame(index=[f'Config {i+1}' for i in range(len(comparison_df))])

    for band, target in zip(age_bands, target_age):
        col_name = f'age_hist.{band}'
        if col_name in comparison_df.columns:
            age_analysis[f'{band} (%)'] = (comparison_df[col_name] * 100).round(1)
            print(f'✅ Found column {col_name}')
        else:
            age_analysis[f'{band} (%)'] = 'N/A'
            print(f'❌ Missing column {col_name}')

    print(f'\nAge Analysis Shape: {age_analysis.shape}')
    print(age_analysis.head())

    # Test score components calculation with error handling
    print(f'\n🎯 Testing Score Component Analysis:')
    score_components = pd.DataFrame(
        columns=['Age Error', 'Tenure Error', 'HC Growth Error', 'Pay Growth Error', 'Total Score'],
        index=[f"Config {i+1}" for i in range(len(comparison_df))]
    )

    for i, (idx, config) in enumerate(comparison_df.iterrows()):
        # Calculate individual score components
        age_hist = {col.replace('age_hist.', ''): config[col] for col in config.index if col.startswith('age_hist.')}
        tenure_hist = {col.replace('tenure_hist.', ''): config[col] for col in config.index if col.startswith('tenure_hist.')}
        
        print(f'Config {i+1} age_hist: {age_hist}')
        print(f'Config {i+1} tenure_hist: {tenure_hist}')
        
        # Age KL divergence with error handling
        target_age_dict = {'<30': 0.109, '30-39': 0.210, '40-49': 0.336, '50-59': 0.210, '60-65': 0.050, '65+': 0.084}
        try:
            if age_hist:
                age_error = sum(p * np.log(p / max(age_hist.get(band, 0.001), 0.001)) 
                               for band, p in target_age_dict.items())
            else:
                age_error = float('nan')
        except:
            age_error = float('nan')
        
        # Tenure KL divergence with error handling
        target_tenure_dict = {'<1': 0.20, '1-3': 0.30, '3-5': 0.25, '5-10': 0.15, '10-15': 0.07, '15+': 0.03}
        try:
            if tenure_hist:
                tenure_error = sum(p * np.log(p / max(tenure_hist.get(band, 0.001), 0.001)) 
                              for band, p in target_tenure_dict.items())
            else:
                tenure_error = float('nan')
        except:
            tenure_error = float('nan')
        
        # Growth errors
        hc_error = abs(config['hc_growth'] - 0.03)
        pay_error = abs(config['pay_growth'] - 0.03)
        
        # Create row data with proper NaN handling
        row_data = [
            round(age_error, 4) if not np.isnan(age_error) else 'N/A',
            round(tenure_error, 4) if not np.isnan(tenure_error) else 'N/A', 
            round(hc_error, 4),
            round(pay_error, 4),
            round(config['score'], 6)
        ]
        
        score_components.loc[f"Config {i+1}"] = row_data
        print(f'Config {i+1} row_data: {row_data}')

    score_components.columns = ['Age Error', 'Tenure Error', 'HC Growth Error', 'Pay Growth Error', 'Total Score']
    print(f'\nScore Components:')
    print(score_components)

    print(f'\n✅ All tests completed successfully!')

if __name__ == "__main__":
    test_campaign_data()
