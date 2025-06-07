#!/usr/bin/env python3
"""
Analyze successful configurations from today's campaign to identify stable parameter ranges.
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import os

def extract_key_parameters(config_path):
    """Extract key parameters from a config file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        global_params = config.get('global_parameters', {})
        
        # Extract key parameters
        params = {
            'config_file': os.path.basename(config_path),
            
            # Critical parameters that may cause failures
            'new_hire_rate': global_params.get('new_hires', {}).get('new_hire_rate'),
            'annual_compensation_increase_rate': global_params.get('annual_compensation_increase_rate'),
            'target_growth': global_params.get('target_growth'),
            'termination_base_rate_for_new_hire': global_params.get('termination_hazard', {}).get('base_rate_for_new_hire'),
            
            # Promotion parameters
            'promotion_base_rate': global_params.get('promotion_hazard', {}).get('base_rate'),
            'promotion_level_dampener_factor': global_params.get('promotion_hazard', {}).get('level_dampener_factor'),
            
            # Merit/COLA parameters
            'merit_base': global_params.get('raises_hazard', {}).get('merit_base'),
            'cola_rate': global_params.get('compensation', {}).get('COLA_rate'),
            
            # Check for COLA duplicate keys issue
            'has_cola_duplicate_keys': False,
            'cola_by_year_numeric_keys': {},
            'cola_by_year_string_keys': {},
            
            # Termination age multipliers (high values may be problematic)
            'term_age_mult_60_65': global_params.get('termination_hazard', {}).get('age_multipliers', {}).get('60-65'),
            'term_age_mult_65_plus': global_params.get('termination_hazard', {}).get('age_multipliers', {}).get('65+'),
            
            # Other key parameters
            'max_working_age': global_params.get('max_working_age'),
            'min_working_age': global_params.get('min_working_age'),
            'new_hire_average_age': global_params.get('new_hire_average_age'),
            'new_hire_age_std_dev': global_params.get('new_hire_age_std_dev'),
        }
        
        # Check for COLA duplicate keys
        cola_by_year = global_params.get('cola_hazard', {}).get('by_year', {})
        if cola_by_year:
            numeric_keys = {}
            string_keys = {}
            
            for key, value in cola_by_year.items():
                if isinstance(key, int):
                    numeric_keys[key] = value
                elif isinstance(key, str) and key.isdigit():
                    string_keys[int(key)] = value
            
            # Check if we have both numeric and string versions of the same year
            if set(numeric_keys.keys()) & set(string_keys.keys()):
                params['has_cola_duplicate_keys'] = True
                params['cola_by_year_numeric_keys'] = numeric_keys
                params['cola_by_year_string_keys'] = string_keys
        
        return params
        
    except Exception as e:
        print(f"Error processing {config_path}: {e}")
        return None

def main():
    # List of sampled config files from today
    config_files = [
        "/Users/nicholasamaral/Library/Mobile Documents/com~apple~CloudDocs/Development/cost-model/tuned/config_000_20250606_095828.yaml",
        "/Users/nicholasamaral/Library/Mobile Documents/com~apple~CloudDocs/Development/cost-model/tuned/config_009_20250606_095904.yaml",
        "/Users/nicholasamaral/Library/Mobile Documents/com~apple~CloudDocs/Development/cost-model/tuned/config_019_20250606_095905.yaml",
        "/Users/nicholasamaral/Library/Mobile Documents/com~apple~CloudDocs/Development/cost-model/tuned/config_029_20250606_095905.yaml",
        "/Users/nicholasamaral/Library/Mobile Documents/com~apple~CloudDocs/Development/cost-model/tuned/config_039_20250606_095905.yaml",
        "/Users/nicholasamaral/Library/Mobile Documents/com~apple~CloudDocs/Development/cost-model/tuned/config_049_20250606_095905.yaml",
        "/Users/nicholasamaral/Library/Mobile Documents/com~apple~CloudDocs/Development/cost-model/tuned/config_059_20250606_095905.yaml",
        "/Users/nicholasamaral/Library/Mobile Documents/com~apple~CloudDocs/Development/cost-model/tuned/config_069_20250606_095905.yaml",
        "/Users/nicholasamaral/Library/Mobile Documents/com~apple~CloudDocs/Development/cost-model/tuned/config_079_20250606_095906.yaml",
        "/Users/nicholasamaral/Library/Mobile Documents/com~apple~CloudDocs/Development/cost-model/tuned/config_089_20250606_095906.yaml",
        "/Users/nicholasamaral/Library/Mobile Documents/com~apple~CloudDocs/Development/cost-model/tuned/config_099_20250606_095906.yaml",
        "/Users/nicholasamaral/Library/Mobile Documents/com~apple~CloudDocs/Development/cost-model/tuned/config_109_20250606_095906.yaml",
        "/Users/nicholasamaral/Library/Mobile Documents/com~apple~CloudDocs/Development/cost-model/tuned/config_119_20250606_095906.yaml",
        "/Users/nicholasamaral/Library/Mobile Documents/com~apple~CloudDocs/Development/cost-model/tuned/config_129_20250606_095906.yaml",
        "/Users/nicholasamaral/Library/Mobile Documents/com~apple~CloudDocs/Development/cost-model/tuned/config_139_20250606_095906.yaml",
    ]
    
    # Extract parameters from all configs
    all_params = []
    for config_file in config_files:
        params = extract_key_parameters(config_file)
        if params:
            all_params.append(params)
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_params)
    
    print("=== ANALYSIS OF 15 SUCCESSFUL CONFIGURATIONS FROM 2025-06-06 ===\n")
    
    # Check for COLA duplicate keys issue
    print("1. COLA DUPLICATE KEYS ANALYSIS:")
    cola_duplicates = df[df['has_cola_duplicate_keys'] == True]
    print(f"   Configs with duplicate COLA keys: {len(cola_duplicates)}/{len(df)}")
    if len(cola_duplicates) > 0:
        print("   Files with duplicates:")
        for _, row in cola_duplicates.iterrows():
            print(f"     - {row['config_file']}")
            print(f"       Numeric keys: {row['cola_by_year_numeric_keys']}")
            print(f"       String keys: {row['cola_by_year_string_keys']}")
    print()
    
    # Analyze key parameter ranges
    print("2. KEY PARAMETER RANGES ANALYSIS:")
    
    # Define critical parameters to analyze
    critical_params = [
        'new_hire_rate',
        'annual_compensation_increase_rate', 
        'target_growth',
        'termination_base_rate_for_new_hire',
        'promotion_base_rate',
        'merit_base',
        'term_age_mult_60_65',
        'term_age_mult_65_plus'
    ]
    
    for param in critical_params:
        if param in df.columns:
            values = df[param].dropna()
            if len(values) > 0:
                print(f"   {param}:")
                print(f"     Range: {values.min():.4f} - {values.max():.4f}")
                print(f"     Mean: {values.mean():.4f}")
                print(f"     Std: {values.std():.4f}")
                print(f"     25th percentile: {values.quantile(0.25):.4f}")
                print(f"     75th percentile: {values.quantile(0.75):.4f}")
                print()
    
    # Additional analysis
    print("3. ADDITIONAL PARAMETER ANALYSIS:")
    
    other_params = [
        'promotion_level_dampener_factor',
        'cola_rate',
        'max_working_age',
        'new_hire_average_age',
        'new_hire_age_std_dev'
    ]
    
    for param in other_params:
        if param in df.columns:
            values = df[param].dropna()
            if len(values) > 0:
                print(f"   {param}:")
                print(f"     Range: {values.min():.4f} - {values.max():.4f}")
                print(f"     Mean: {values.mean():.4f}")
                print()
    
    # Check for extreme values that might cause issues
    print("4. POTENTIAL PROBLEMATIC VALUE DETECTION:")
    
    # Check for high new_hire_rate (>= 0.5)
    high_new_hire = df[df['new_hire_rate'] >= 0.5]
    print(f"   Configs with high new_hire_rate (>=0.5): {len(high_new_hire)}")
    if len(high_new_hire) > 0:
        print(f"     Values: {high_new_hire['new_hire_rate'].tolist()}")
    
    # Check for very high target_growth (>= 0.08)
    high_growth = df[df['target_growth'] >= 0.08]
    print(f"   Configs with high target_growth (>=0.08): {len(high_growth)}")
    if len(high_growth) > 0:
        print(f"     Values: {high_growth['target_growth'].tolist()}")
    
    # Check for very low termination base rate (<= 0.02)
    low_term_rate = df[df['termination_base_rate_for_new_hire'] <= 0.02]
    print(f"   Configs with low termination_base_rate_for_new_hire (<=0.02): {len(low_term_rate)}")
    if len(low_term_rate) > 0:
        print(f"     Values: {low_term_rate['termination_base_rate_for_new_hire'].tolist()}")
    
    # Check for very high annual compensation increase (>= 0.04)
    high_comp_increase = df[df['annual_compensation_increase_rate'] >= 0.04]
    print(f"   Configs with high annual_compensation_increase_rate (>=0.04): {len(high_comp_increase)}")
    if len(high_comp_increase) > 0:
        print(f"     Values: {high_comp_increase['annual_compensation_increase_rate'].tolist()}")
    
    print()
    
    # Save detailed data for further analysis
    output_file = "successful_configs_analysis.csv"
    df.to_csv(output_file, index=False)
    print(f"5. DETAILED DATA SAVED TO: {output_file}")
    
    print("\n=== RECOMMENDED STABLE PARAMETER RANGES ===")
    print("Based on successful configurations analysis:")
    print()
    
    # Calculate conservative ranges (10th to 90th percentile)
    for param in critical_params:
        if param in df.columns:
            values = df[param].dropna()
            if len(values) > 0:
                q10 = values.quantile(0.1)
                q90 = values.quantile(0.9)
                print(f"{param}: {q10:.4f} - {q90:.4f} (10th-90th percentile)")

if __name__ == "__main__":
    main()