#!/usr/bin/env python3
"""
Comprehensive analysis of the previous tuning campaign with the new understanding
that age distribution preservation (not transformation) is the goal.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Import age band utilities
import sys
sys.path.append('.')
from cost_model.state.age import assign_age_band, AGE_BAND_CATEGORICAL_DTYPE

def load_starting_census_age_distribution():
    """Load and calculate the starting census age distribution (our baseline target)."""
    census_path = Path("data/census_template.parquet")
    df = pd.read_parquet(census_path)
    
    # Calculate ages as of simulation start (2025-01-01)
    simulation_start = pd.Timestamp('2025-01-01')
    df['calculated_age'] = ((simulation_start - pd.to_datetime(df['employee_birth_date'])).dt.days / 365.25).round(1)
    df['calculated_age_band'] = df['calculated_age'].map(assign_age_band).astype(AGE_BAND_CATEGORICAL_DTYPE)
    
    # Calculate age distribution
    age_dist = df['calculated_age_band'].value_counts(normalize=True).sort_index()
    baseline_age_dist = {str(band): float(count) for band, count in age_dist.items()}
    
    return baseline_age_dist, df

def load_tuning_results():
    """Load the tuning results JSON file."""
    with open("tuned/tuning_results.json", 'r') as f:
        results = json.load(f)
    return results

def analyze_best_config_performance():
    """Analyze the performance of config_193 (the 'best' configuration)."""
    
    print("="*80)
    print("ANALYSIS OF PREVIOUS TUNING CAMPAIGN WITH AGE PRESERVATION GOAL")
    print("="*80)
    
    # Load baseline age distribution from starting census
    baseline_age_dist, census_df = load_starting_census_age_distribution()
    
    print("\n1. STARTING CENSUS AGE DISTRIBUTION (NEW BASELINE TARGET)")
    print("-" * 60)
    print("This is what we should preserve over the 5-year simulation:")
    for band, proportion in baseline_age_dist.items():
        count = int(proportion * len(census_df))
        print(f"  {band:>8}: {proportion:6.3f} ({count:3d} employees)")
    
    # Load tuning results
    results = load_tuning_results()
    
    # Find config_193 results
    config_193_result = None
    for result in results:
        if "config_193_20250604_110340" in result["config_path"]:
            config_193_result = result
            break
    
    if not config_193_result:
        print("ERROR: Could not find config_193 results!")
        return
    
    print(f"\n2. WORKFORCE FLOW ANALYSIS (Config 193)")
    print("-" * 60)
    summary = config_193_result["summary"]
    
    # Calculate workforce flows
    initial_headcount = 119  # From census
    final_headcount = summary["final_headcount"]
    years = summary["years"]
    hc_growth_rate = summary["hc_growth"]
    
    # Estimate annual flows (this is approximate)
    annual_hc_growth = hc_growth_rate
    annual_turnover_estimate = 0.15  # From config termination rate
    
    print(f"  Initial headcount: {initial_headcount}")
    print(f"  Final headcount: {final_headcount}")
    print(f"  Simulation years: {years}")
    print(f"  Total HC growth: {hc_growth_rate:.3f} ({hc_growth_rate*100:.1f}%)")
    print(f"  Annual HC growth: {annual_hc_growth:.3f} ({annual_hc_growth*100:.1f}%)")
    print(f"  Estimated annual turnover: {annual_turnover_estimate:.3f} ({annual_turnover_estimate*100:.1f}%)")
    
    # Estimate annual flows
    avg_headcount = (initial_headcount + final_headcount) / 2
    estimated_annual_terminations = avg_headcount * annual_turnover_estimate
    estimated_annual_hires = estimated_annual_terminations + (annual_hc_growth * avg_headcount)
    
    print(f"  Estimated annual terminations: {estimated_annual_terminations:.1f}")
    print(f"  Estimated annual new hires: {estimated_annual_hires:.1f}")
    print(f"  Estimated annual turnover rate: {estimated_annual_terminations/avg_headcount:.3f}")
    
    print(f"\n3. ACHIEVED VS. TARGET COMPARISON")
    print("-" * 60)
    
    # Age distribution comparison
    achieved_age_dist = summary["age_hist"]
    
    print("Age Distribution Comparison:")
    print(f"{'Band':>8} {'Baseline':>10} {'Achieved':>10} {'Difference':>12} {'Error':>8}")
    print("-" * 50)
    
    total_age_error = 0
    for band in baseline_age_dist.keys():
        baseline_prop = baseline_age_dist[band]
        achieved_prop = achieved_age_dist.get(band, 0.0)
        difference = achieved_prop - baseline_prop
        error = abs(difference)
        total_age_error += error
        
        print(f"{band:>8} {baseline_prop:>10.3f} {achieved_prop:>10.3f} {difference:>+12.3f} {error:>8.3f}")
    
    print(f"{'TOTAL':>8} {'':>10} {'':>10} {'':>12} {total_age_error:>8.3f}")
    
    # Other metrics
    print(f"\nOther Metrics:")
    print(f"  Headcount growth: {summary['hc_growth']:.3f} (target: 0.03)")
    print(f"  Pay growth: {summary['pay_growth']:.3f} (target: 0.03)")
    print(f"  Overall score: {config_193_result['score']:.3f} (lower is better)")
    
    print(f"\n4. AGE PRESERVATION ASSESSMENT")
    print("-" * 60)
    
    # Calculate how well age distribution was preserved
    age_preservation_score = 1 - (total_age_error / 2)  # Normalize to 0-1 scale
    
    print(f"Age preservation score: {age_preservation_score:.3f} (1.0 = perfect preservation)")
    print(f"Total age distribution error: {total_age_error:.3f}")
    
    # Analyze which age bands changed most
    age_changes = []
    for band in baseline_age_dist.keys():
        baseline_prop = baseline_age_dist[band]
        achieved_prop = achieved_age_dist.get(band, 0.0)
        change = achieved_prop - baseline_prop
        age_changes.append((band, change, abs(change)))
    
    age_changes.sort(key=lambda x: x[2], reverse=True)  # Sort by absolute change
    
    print(f"\nLargest age distribution changes:")
    for band, change, abs_change in age_changes[:3]:
        direction = "increased" if change > 0 else "decreased"
        print(f"  {band}: {direction} by {abs_change:.3f} ({abs_change*100:.1f} percentage points)")
    
    print(f"\n5. FEASIBILITY ASSESSMENT")
    print("-" * 60)
    
    # Analyze the challenge of preserving age distribution
    print("Challenges for age preservation:")
    
    # New hire age profile from config
    new_hire_avg_age = 30  # From config
    census_avg_age = census_df['calculated_age'].mean()
    
    print(f"  - New hires average age: {new_hire_avg_age}")
    print(f"  - Census average age: {census_avg_age:.1f}")
    print(f"  - Age gap: {census_avg_age - new_hire_avg_age:.1f} years")
    
    if new_hire_avg_age < census_avg_age:
        print(f"  - New hires are younger than average, creating downward age pressure")
        print(f"  - With {estimated_annual_hires:.1f} annual hires, this significantly impacts age mix")
    
    # Age multipliers from config
    print(f"\nAge multipliers in termination hazard (config_193):")
    print(f"  - <30: 0.6 (lower termination)")
    print(f"  - 30-39: 1.0 (baseline)")
    print(f"  - 40-49: 0.9 (slightly lower)")
    print(f"  - 50-59: 1.1 (slightly higher)")
    print(f"  - 60-65: 2.0 (much higher)")
    
    print(f"\nAge multipliers in promotion hazard (config_193):")
    print(f"  - <30: 1.6 (higher promotion)")
    print(f"  - 30-39: 1.3 (higher promotion)")
    print(f"  - 40-49: 0.9 (lower promotion)")
    print(f"  - 50-59: 0.5 (much lower)")
    print(f"  - 60-65: 0.1 (very low)")
    
    print(f"\n6. STRATEGIC RECOMMENDATIONS")
    print("-" * 60)
    
    print("Based on this analysis:")
    print("1. Age preservation is challenging with current new hire age profile")
    print("2. The 'best' config achieved moderate age preservation but missed other targets")
    print("3. Consider adjusting:")
    print("   - New hire age distribution parameters")
    print("   - Age multipliers to better counteract age drift")
    print("   - Score weights to prioritize age preservation if it's primary goal")
    print("4. The current SEARCH_SPACE may need new hire age parameters")

if __name__ == "__main__":
    analyze_best_config_performance()
