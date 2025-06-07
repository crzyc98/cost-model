#!/usr/bin/env python3

import json
import os
import glob

def analyze_campaign_results():
    """Analyze all tuning results to find the best configurations"""
    
    # Find all tuning_results.json files
    pattern = "/Users/nicholasamaral/Library/Mobile Documents/com~apple~CloudDocs/Development/cost-model/**/tuning_results.json"
    files = glob.glob(pattern, recursive=True)
    
    all_results = []
    
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list) and len(data) > 0:
                # Get campaign name from directory
                campaign = os.path.basename(os.path.dirname(file_path))
                
                # Find best score in this campaign
                best = min(data, key=lambda x: x.get('score', float('inf')))
                
                result = {
                    'campaign': campaign,
                    'score': best.get('score'),
                    'hc_growth': best.get('summary', {}).get('hc_growth'),
                    'pay_growth': best.get('summary', {}).get('pay_growth'),
                    'config_path': best.get('config_path'),
                    'final_headcount': best.get('summary', {}).get('final_headcount'),
                    'file_path': file_path
                }
                
                all_results.append(result)
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Sort by score (lower is better)
    all_results.sort(key=lambda x: x['score'] if x['score'] is not None else float('inf'))
    
    print("=" * 80)
    print("COMPREHENSIVE CAMPAIGN RESULTS ANALYSIS")
    print("=" * 80)
    print()
    
    # Print all results
    for i, result in enumerate(all_results, 1):
        print(f"{i:2d}. Campaign: {result['campaign']}")
        print(f"    Score: {result['score']:.6f}")
        print(f"    HC Growth: {result['hc_growth']:.4f}" if result['hc_growth'] is not None else "    HC Growth: N/A")
        print(f"    Pay Growth: {result['pay_growth']:.4f}" if result['pay_growth'] is not None else "    Pay Growth: N/A")
        print(f"    Final Headcount: {result['final_headcount']}")
        print(f"    Config Path: {result['config_path']}")
        print()
    
    print("=" * 80)
    print("TOP 5 CANDIDATES FOR PRODUCTION")
    print("=" * 80)
    print()
    
    # Filter for positive headcount growth or at least stable (>= -0.01)
    good_growth = [r for r in all_results if r['hc_growth'] is not None and r['hc_growth'] >= -0.01]
    
    print("Configurations with positive/stable headcount growth:")
    for i, result in enumerate(good_growth[:5], 1):
        print(f"{i}. Campaign: {result['campaign']}")
        print(f"   Score: {result['score']:.6f}")
        print(f"   HC Growth: {result['hc_growth']:.4f}")
        print(f"   Pay Growth: {result['pay_growth']:.4f}")
        print(f"   Config: {result['config_path']}")
        print()
    
    # Overall best score regardless of headcount
    print("Overall best score:")
    if all_results:
        best = all_results[0]
        print(f"   Campaign: {best['campaign']}")
        print(f"   Score: {best['score']:.6f}")
        print(f"   HC Growth: {best['hc_growth']:.4f}")
        print(f"   Pay Growth: {best['pay_growth']:.4f}")
        print(f"   Config: {best['config_path']}")
    
    return all_results

if __name__ == "__main__":
    analyze_campaign_results()