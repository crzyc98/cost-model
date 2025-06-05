#!/usr/bin/env python3
"""
Campaign 3 Results Analysis Script

This script performs comprehensive analysis of Campaign 3 auto-tuning results
to understand parameter sensitivities and identify refinements for Campaign 4.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import yaml
from pathlib import Path
import warnings
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Configuration
PROJECT_ROOT = Path('..')
CAMPAIGN_DIR = PROJECT_ROOT / 'campaign_3_results'
CENSUS_FILE = PROJECT_ROOT / 'data' / 'census_template.parquet'

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

def analyze_campaign_performance(campaign_df):
    """Analyze overall campaign performance"""
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
    
    return top_10

def analyze_demographic_performance(campaign_df):
    """Analyze demographic distribution performance"""
    print(f"\n👥 Demographic Analysis:")
    
    # Define targets (these should match your tuning baseline)
    age_targets = {
        '<30': 0.15,  # Target 15% under 30
        '30-39': 0.35,
        '40-49': 0.25,
        '50-59': 0.15,
        '60-65': 0.07,
        '65+': 0.03
    }
    
    tenure_targets = {
        '<1': 0.20,  # Target 20% new hires
        '1-3': 0.25,
        '3-5': 0.20,
        '5-10': 0.20,
        '10-15': 0.10,
        '15+': 0.05
    }
    
    # Calculate errors for top configurations
    top_5 = campaign_df.nsmallest(5, 'score')
    
    print(f"\n🎯 Age Distribution Analysis (Top 5 configs):")
    for age_band, target in age_targets.items():
        col_name = f'age_hist.{age_band}'
        if col_name in top_5.columns:
            actual_values = top_5[col_name]
            errors = actual_values - target
            print(f"   {age_band:>6}: Target={target:.1%}, Actual={actual_values.mean():.1%}, Error={errors.mean():.1%}")
    
    print(f"\n🎯 Tenure Distribution Analysis (Top 5 configs):")
    for tenure_band, target in tenure_targets.items():
        col_name = f'tenure_hist.{tenure_band}'
        if col_name in top_5.columns:
            actual_values = top_5[col_name]
            errors = actual_values - target
            print(f"   {tenure_band:>6}: Target={target:.1%}, Actual={actual_values.mean():.1%}, Error={errors.mean():.1%}")

def load_config_parameters(campaign_df):
    """Load configuration parameters from YAML files"""
    print(f"\n🔧 Loading Configuration Parameters...")
    
    config_params = []
    for _, row in campaign_df.iterrows():
        config_path = Path(row['config_path'])
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Extract key parameters
                params = {
                    'config_name': config_path.stem,
                    'score': row['score'],
                    'hc_growth': row['hc_growth'],
                    'pay_growth': row['pay_growth']
                }
                
                # Extract global parameters
                if 'global_params' in config:
                    gp = config['global_params']
                    params.update({
                        'target_growth': gp.get('target_growth', 0),
                        'new_hire_rate': gp.get('new_hire_rate', 0),
                        'new_hire_average_age': gp.get('new_hire_average_age', 0),
                        'new_hire_age_std_dev': gp.get('new_hire_age_std_dev', 0)
                    })
                    
                    # Extract termination hazard parameters
                    if 'termination_hazard' in gp:
                        th = gp['termination_hazard']
                        params.update({
                            'base_rate_for_new_hire': th.get('base_rate_for_new_hire', 0),
                            'tenure_mult_lt1': th.get('tenure_multipliers', {}).get('<1', 1.0),
                            'tenure_mult_1_3': th.get('tenure_multipliers', {}).get('1-3', 1.0),
                            'age_mult_lt30': th.get('age_multipliers', {}).get('<30', 1.0),
                            'age_mult_60_65': th.get('age_multipliers', {}).get('60-65', 1.0),
                            'age_mult_65plus': th.get('age_multipliers', {}).get('65+', 1.0)
                        })
                
                config_params.append(params)
                
            except Exception as e:
                print(f"   ⚠️  Error loading {config_path}: {e}")
    
    params_df = pd.DataFrame(config_params)
    print(f"✅ Loaded parameters for {len(params_df)} configurations")
    return params_df

def analyze_parameter_sensitivity(params_df):
    """Analyze parameter sensitivity and correlations"""
    print(f"\n🔍 Parameter Sensitivity Analysis:")
    
    # Calculate correlations with key outcomes
    numeric_cols = params_df.select_dtypes(include=[np.number]).columns
    param_cols = [col for col in numeric_cols if col not in ['score', 'hc_growth', 'pay_growth']]
    
    print(f"\n📊 Correlations with Score (lower is better):")
    score_corrs = params_df[param_cols + ['score']].corr()['score'].sort_values()
    for param, corr in score_corrs.items():
        if param != 'score':
            print(f"   {param:>25}: {corr:>7.3f}")
    
    print(f"\n📊 Correlations with HC Growth:")
    hc_corrs = params_df[param_cols + ['hc_growth']].corr()['hc_growth'].sort_values(ascending=False)
    for param, corr in hc_corrs.items():
        if param != 'hc_growth':
            print(f"   {param:>25}: {corr:>7.3f}")
    
    return score_corrs, hc_corrs

def generate_insights_and_recommendations(campaign_df, params_df, score_corrs, hc_corrs):
    """Generate insights and recommendations for Campaign 4"""
    print(f"\n💡 Key Insights and Campaign 4 Recommendations:")
    
    # Analyze headcount growth issue
    avg_hc_growth = campaign_df['hc_growth'].mean()
    target_hc_growth = 0.03  # 3% target
    hc_gap = target_hc_growth - avg_hc_growth
    
    print(f"\n🎯 Headcount Growth Analysis:")
    print(f"   Target: {target_hc_growth:.1%}")
    print(f"   Average Achieved: {avg_hc_growth:.1%}")
    print(f"   Gap: {hc_gap:.1%}")
    
    # Analyze age distribution issues
    top_5 = campaign_df.nsmallest(5, 'score')
    avg_under_30 = top_5['age_hist.<30'].mean()
    target_under_30 = 0.15
    age_gap = target_under_30 - avg_under_30
    
    print(f"\n👶 Age Distribution Analysis:")
    print(f"   Target <30: {target_under_30:.1%}")
    print(f"   Average Achieved <30: {avg_under_30:.1%}")
    print(f"   Gap: {age_gap:.1%}")
    
    # Parameter boundary analysis
    print(f"\n🔧 Parameter Boundary Analysis:")
    if not params_df.empty:
        for col in ['target_growth', 'new_hire_rate', 'new_hire_average_age']:
            if col in params_df.columns:
                min_val = params_df[col].min()
                max_val = params_df[col].max()
                print(f"   {col:>20}: Range [{min_val:.3f}, {max_val:.3f}]")
    
    print(f"\n🚀 Campaign 4 Recommendations:")
    print(f"   1. Increase target_growth range to 0.04-0.06 (currently achieving ~{avg_hc_growth:.1%})")
    print(f"   2. Increase new_hire_rate range to 0.25-0.40 (need more volume)")
    print(f"   3. Lower new_hire_average_age to 22-26 (currently <30 at {avg_under_30:.1%})")
    print(f"   4. Strengthen age multipliers for 60+ (increase retirement pressure)")
    print(f"   5. Adjust score weights: HC_Growth=0.50, Age=0.25, Tenure=0.20, Pay=0.05")

def main():
    """Main analysis function"""
    print("🔍 Campaign 3 Comprehensive Analysis")
    print("=" * 50)
    
    # Load campaign results
    campaign_df = load_campaign_results(CAMPAIGN_DIR)
    if campaign_df is None:
        return
    
    # Perform analyses
    top_10 = analyze_campaign_performance(campaign_df)
    analyze_demographic_performance(campaign_df)
    
    # Load and analyze parameters
    params_df = load_config_parameters(campaign_df)
    if not params_df.empty:
        score_corrs, hc_corrs = analyze_parameter_sensitivity(params_df)
        generate_insights_and_recommendations(campaign_df, params_df, score_corrs, hc_corrs)
    else:
        print("⚠️  No parameter data available for sensitivity analysis")
    
    print(f"\n✅ Analysis Complete!")
    print(f"📊 Campaign 3 achieved best score of {campaign_df['score'].min():.6f}")
    print(f"🎯 Key focus areas for Campaign 4: HC Growth, Age <30, Parameter Ranges")

if __name__ == "__main__":
    main()
