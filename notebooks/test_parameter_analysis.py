#!/usr/bin/env python3
"""
Test the parameter sensitivity analysis functionality.
"""

import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def test_parameter_sensitivity_analysis():
    """Test the parameter sensitivity analysis with campaign results"""
    
    # Load campaign results
    with open('../campaign_round2_results/tuning_results.json', 'r') as f:
        results = json.load(f)

    campaign_df = pd.DataFrame(results)
    summary_df = pd.json_normalize(campaign_df['summary'])
    campaign_results = pd.concat([campaign_df.drop('summary', axis=1), summary_df], axis=1)

    print(f'✅ Loaded {len(campaign_results)} campaign results')

    # Load configuration files to extract parameters
    config_params = []
    
    # Get the base directory (project root)
    base_dir = Path('../')  # Go up from notebooks/ to project root
    
    for idx, row in campaign_results.iterrows():
        config_path = base_dir / row['config_path']
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Extract key parameters
                params = {
                    'config_path': row['config_path'],
                    'score': row['score'],
                    'hc_growth': row['hc_growth'],
                    'pay_growth': row['pay_growth'],
                    'target_growth': config['global_parameters'].get('target_growth', None),
                    'new_hire_rate': config['global_parameters'].get('new_hires', {}).get('new_hire_rate', None),
                    'new_hire_avg_age': config['global_parameters'].get('new_hire_average_age', None),
                    'max_working_age': config['global_parameters'].get('max_working_age', None),
                    'base_term_rate': config['global_parameters'].get('termination_hazard', {}).get('base_rate_for_new_hire', None),
                    'term_mult_under_1': config['global_parameters'].get('termination_hazard', {}).get('tenure_multipliers', {}).get('<1', None),
                    'term_mult_under_30': config['global_parameters'].get('termination_hazard', {}).get('age_multipliers', {}).get('<30', None),
                }
                config_params.append(params)
            except Exception as e:
                print(f"⚠️ Could not load config {config_path}: {e}")
    
    if not config_params:
        print("❌ No configuration parameters could be loaded")
        return
    
    params_df = pd.DataFrame(config_params)
    print(f"✅ Successfully loaded {len(config_params)} configuration parameters")
    
    # Show sample data
    print(f"\n📊 Sample Parameters:")
    print(params_df.head())
    
    # Correlation analysis
    numeric_cols = params_df.select_dtypes(include=[np.number]).columns
    print(f"\n🔢 Numeric columns: {list(numeric_cols)}")
    
    correlation_matrix = params_df[numeric_cols].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
                square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
    plt.title('⚙️ Parameter Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('parameter_correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print top correlations with score
    if 'score' in correlation_matrix.columns:
        score_corr = correlation_matrix['score'].abs().sort_values(ascending=False)
        print("\n🎯 Parameters Most Correlated with Score:")
        for param, corr in score_corr.items():
            if param != 'score' and not pd.isna(corr):
                direction = "📈" if correlation_matrix.loc['score', param] > 0 else "📉"
                print(f"   {param}: {corr:.3f} {direction}")
    
    # Key parameter vs outcome scatter plots
    key_params = ['target_growth', 'new_hire_rate', 'new_hire_avg_age', 'base_term_rate']
    key_params = [p for p in key_params if p in params_df.columns and params_df[p].notna().any()]
    
    print(f"\n📈 Available key parameters for plotting: {key_params}")
    
    if len(key_params) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('⚙️ Parameter Sensitivity Analysis', fontsize=16, fontweight='bold')
        
        for i, param in enumerate(key_params[:4]):
            row, col = i // 2, i % 2
            
            # Score vs parameter
            scatter = axes[row, col].scatter(params_df[param], params_df['score'], 
                                           c=params_df['hc_growth']*100, cmap='viridis', alpha=0.7)
            axes[row, col].set_xlabel(param.replace('_', ' ').title())
            axes[row, col].set_ylabel('Score')
            axes[row, col].set_title(f'Score vs {param.replace("_", " ").title()}')
            axes[row, col].grid(True, alpha=0.3)
            
            # Add colorbar for HC growth
            cbar = plt.colorbar(scatter, ax=axes[row, col])
            cbar.set_label('HC Growth (%)')
        
        plt.tight_layout()
        plt.savefig('parameter_sensitivity_plots.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    print(f"\n✅ Parameter sensitivity analysis completed successfully!")
    print(f"📊 Analyzed {len(params_df)} configurations")
    print(f"🔢 Found {len(numeric_cols)} numeric parameters")
    
    return params_df

if __name__ == "__main__":
    test_parameter_sensitivity_analysis()
