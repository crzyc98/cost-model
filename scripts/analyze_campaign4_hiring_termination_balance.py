#!/usr/bin/env python3
"""
User Story Y.2: Analyze Hiring vs. Termination Balance Dynamics

This script analyzes the actual flow of employees (hires and terminations) in the best 
Campaign 4 configuration to understand where simulation reality diverges from hiring 
formula expectations.

Key Analysis:
1. Quantify actual hires and terminations per year
2. Analyze new_hire_termination_rate used vs. actual
3. Calculate net headcount change and compare to target
4. Identify discrepancies in the hiring/termination balance
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from cost_model.state.schema import EMP_ID, EMP_ACTIVE, EMP_HIRE_DATE
from cost_model.state.event_log import EVT_HIRE, EVT_TERM


def load_config_parameters(config_path: Path) -> dict:
    """Load key parameters from the configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    params = {
        'target_growth': config['global_parameters'].get('target_growth', 0.0),
        'new_hire_termination_rate': config['global_parameters']['attrition'].get('new_hire_termination_rate', 0.25),
        'new_hire_rate': config['global_parameters']['new_hires'].get('new_hire_rate', 0.0),
        'projection_years': config['global_parameters'].get('projection_years', 3),
        'start_year': config['global_parameters'].get('start_year', 2025)
    }
    
    print("=== Configuration Parameters ===")
    for key, value in params.items():
        print(f"{key}: {value}")
    print()
    
    return params


def analyze_yearly_events(output_dir: Path, params: dict) -> dict:
    """Analyze hiring and termination events for each simulation year."""
    results = {}
    start_year = params['start_year']
    projection_years = params['projection_years']
    
    print("=== Yearly Event Analysis ===")
    
    for year_offset in range(projection_years):
        year = start_year + year_offset
        year_dir = output_dir / f"year={year}"
        
        if not year_dir.exists():
            print(f"Warning: Year {year} directory not found")
            continue
            
        # Load events and snapshot for this year
        events_file = year_dir / "events.parquet"
        snapshot_file = year_dir / "snapshot.parquet"
        
        if not events_file.exists() or not snapshot_file.exists():
            print(f"Warning: Missing files for year {year}")
            continue
            
        events_df = pd.read_parquet(events_file)
        snapshot_df = pd.read_parquet(snapshot_file)
        
        # Analyze events
        hire_events = events_df[events_df['event_type'] == EVT_HIRE] if not events_df.empty else pd.DataFrame()
        term_events = events_df[events_df['event_type'] == EVT_TERM] if not events_df.empty else pd.DataFrame()
        
        # Count active employees at end of year
        active_count = snapshot_df[EMP_ACTIVE].sum() if EMP_ACTIVE in snapshot_df.columns else len(snapshot_df)
        
        # Analyze new hire terminations (hires that terminated in same year)
        new_hire_terms = 0
        if not hire_events.empty and not term_events.empty:
            hired_ids = set(hire_events[EMP_ID].tolist())
            termed_ids = set(term_events[EMP_ID].tolist())
            new_hire_terms = len(hired_ids.intersection(termed_ids))
        
        year_results = {
            'year': year,
            'total_hires': len(hire_events),
            'total_terminations': len(term_events),
            'new_hire_terminations': new_hire_terms,
            'net_change': len(hire_events) - len(term_events),
            'eoy_active_count': active_count,
            'hire_events': hire_events,
            'term_events': term_events
        }
        
        results[year] = year_results
        
        print(f"Year {year}:")
        print(f"  Total Hires: {year_results['total_hires']}")
        print(f"  Total Terminations: {year_results['total_terminations']}")
        print(f"  New Hire Terminations: {year_results['new_hire_terminations']}")
        print(f"  Net Change: {year_results['net_change']:+d}")
        print(f"  EOY Active Count: {year_results['eoy_active_count']}")
        print()
    
    return results


def calculate_actual_vs_expected(yearly_results: dict, params: dict) -> dict:
    """Calculate actual vs expected hiring/termination rates and growth."""
    analysis = {}
    
    print("=== Actual vs Expected Analysis ===")
    
    # Get initial headcount (from first year's data)
    first_year = min(yearly_results.keys())
    
    # Calculate cumulative changes
    cumulative_hires = 0
    cumulative_terms = 0
    cumulative_nh_terms = 0
    
    for year in sorted(yearly_results.keys()):
        year_data = yearly_results[year]
        
        cumulative_hires += year_data['total_hires']
        cumulative_terms += year_data['total_terminations']
        cumulative_nh_terms += year_data['new_hire_terminations']
        
        # Calculate actual new hire termination rate
        actual_nh_term_rate = (year_data['new_hire_terminations'] / year_data['total_hires'] 
                              if year_data['total_hires'] > 0 else 0.0)
        
        # Calculate expected vs actual for this year
        expected_target_growth = params['target_growth']
        
        year_analysis = {
            'year': year,
            'actual_nh_termination_rate': actual_nh_term_rate,
            'configured_nh_termination_rate': params['new_hire_termination_rate'],
            'nh_termination_rate_diff': actual_nh_term_rate - params['new_hire_termination_rate'],
            'expected_target_growth': expected_target_growth,
            'cumulative_hires': cumulative_hires,
            'cumulative_terms': cumulative_terms,
            'cumulative_nh_terms': cumulative_nh_terms
        }
        
        analysis[year] = year_analysis
        
        print(f"Year {year}:")
        print(f"  Actual NH Termination Rate: {actual_nh_term_rate:.1%}")
        print(f"  Configured NH Termination Rate: {params['new_hire_termination_rate']:.1%}")
        print(f"  Difference: {year_analysis['nh_termination_rate_diff']:+.1%}")
        print()
    
    return analysis


def calculate_growth_analysis(yearly_results: dict, params: dict) -> dict:
    """Calculate overall growth analysis comparing actual vs target."""
    
    print("=== Growth Analysis ===")
    
    years = sorted(yearly_results.keys())
    first_year = years[0]
    last_year = years[-1]
    
    # Estimate initial headcount (before first year changes)
    # Use the first year's EOY count and reverse the net change
    first_year_data = yearly_results[first_year]
    estimated_initial_count = first_year_data['eoy_active_count'] - first_year_data['net_change']
    
    # Calculate final headcount
    final_count = yearly_results[last_year]['eoy_active_count']
    
    # Calculate actual growth
    actual_growth = (final_count - estimated_initial_count) / estimated_initial_count
    
    # Calculate expected growth
    years_simulated = len(years)
    expected_annual_growth = params['target_growth']
    expected_total_growth = (1 + expected_annual_growth) ** years_simulated - 1
    expected_final_count = round(estimated_initial_count * (1 + expected_total_growth))
    
    # Calculate total flows
    total_hires = sum(year_data['total_hires'] for year_data in yearly_results.values())
    total_terms = sum(year_data['total_terminations'] for year_data in yearly_results.values())
    total_nh_terms = sum(year_data['new_hire_terminations'] for year_data in yearly_results.values())
    
    # Calculate overall new hire termination rate
    overall_nh_term_rate = total_nh_terms / total_hires if total_hires > 0 else 0.0
    
    growth_analysis = {
        'estimated_initial_count': estimated_initial_count,
        'final_count': final_count,
        'actual_growth': actual_growth,
        'expected_total_growth': expected_total_growth,
        'expected_final_count': expected_final_count,
        'growth_shortfall': actual_growth - expected_total_growth,
        'headcount_shortfall': final_count - expected_final_count,
        'total_hires': total_hires,
        'total_terminations': total_terms,
        'total_nh_terminations': total_nh_terms,
        'overall_nh_termination_rate': overall_nh_term_rate,
        'configured_nh_termination_rate': params['new_hire_termination_rate'],
        'nh_rate_difference': overall_nh_term_rate - params['new_hire_termination_rate']
    }
    
    print(f"Initial Headcount (estimated): {estimated_initial_count}")
    print(f"Final Headcount: {final_count}")
    print(f"Actual Growth: {actual_growth:.1%}")
    print(f"Expected Growth: {expected_total_growth:.1%}")
    print(f"Growth Shortfall: {growth_analysis['growth_shortfall']:+.1%}")
    print(f"Headcount Shortfall: {growth_analysis['headcount_shortfall']:+d}")
    print()
    print(f"Total Hires: {total_hires}")
    print(f"Total Terminations: {total_terms}")
    print(f"Total NH Terminations: {total_nh_terms}")
    print(f"Overall NH Termination Rate: {overall_nh_term_rate:.1%}")
    print(f"Configured NH Termination Rate: {params['new_hire_termination_rate']:.1%}")
    print(f"NH Rate Difference: {growth_analysis['nh_rate_difference']:+.1%}")
    print()
    
    return growth_analysis


def identify_discrepancies(yearly_results: dict, growth_analysis: dict, params: dict):
    """Identify and report key discrepancies in the hiring/termination balance."""
    
    print("=== Discrepancy Analysis ===")
    
    # Check if new hire termination rate matches expectations
    nh_rate_diff = growth_analysis['nh_rate_difference']
    if abs(nh_rate_diff) > 0.05:  # 5% threshold
        print(f"🚨 MAJOR DISCREPANCY: New hire termination rate")
        print(f"   Configured: {params['new_hire_termination_rate']:.1%}")
        print(f"   Actual: {growth_analysis['overall_nh_termination_rate']:.1%}")
        print(f"   Difference: {nh_rate_diff:+.1%}")
        print()
    
    # Check growth achievement
    growth_shortfall = growth_analysis['growth_shortfall']
    if growth_shortfall < -0.02:  # More than 2% below target
        print(f"🚨 MAJOR DISCREPANCY: Growth target not achieved")
        print(f"   Target: {params['target_growth']:.1%} per year")
        print(f"   Actual: {growth_analysis['actual_growth']:.1%} total")
        print(f"   Shortfall: {growth_shortfall:+.1%}")
        print()
    
    # Check hiring vs termination balance
    net_change = growth_analysis['total_hires'] - growth_analysis['total_terminations']
    expected_net = growth_analysis['expected_final_count'] - growth_analysis['estimated_initial_count']
    
    if abs(net_change - expected_net) > 5:  # More than 5 employees difference
        print(f"🚨 MAJOR DISCREPANCY: Net hiring vs termination balance")
        print(f"   Expected net change: {expected_net}")
        print(f"   Actual net change: {net_change}")
        print(f"   Difference: {net_change - expected_net:+d}")
        print()
    
    # Summary of likely causes
    print("=== Likely Causes ===")
    if abs(nh_rate_diff) > 0.05:
        print("1. New hire termination rate mismatch suggests:")
        print("   - Parameter location issues (Y.1 finding)")
        print("   - Actual termination logic differs from hiring assumptions")
        print()
    
    if growth_shortfall < -0.02:
        print("2. Growth target failure suggests:")
        print("   - Experienced employee termination rates higher than compensated")
        print("   - Hiring algorithm not generating sufficient gross hires")
        print("   - Target growth parameter not properly translated")
        print()


def main():
    """Main analysis function for Campaign 4 best configuration."""
    
    # Configuration paths
    config_path = Path("tuning/tuned/config_004_20250605_103917.yaml")
    output_dir = Path("tuning/tuned/output_config_004_20250605_103917/Baseline")
    
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        return
        
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        return
    
    print("Campaign 4 Best Configuration Analysis")
    print("=" * 50)
    print()
    
    # Load configuration parameters
    params = load_config_parameters(config_path)
    
    # Analyze yearly events
    yearly_results = analyze_yearly_events(output_dir, params)
    
    if not yearly_results:
        print("Error: No yearly results found")
        return
    
    # Calculate actual vs expected rates
    rate_analysis = calculate_actual_vs_expected(yearly_results, params)
    
    # Calculate growth analysis
    growth_analysis = calculate_growth_analysis(yearly_results, params)
    
    # Identify discrepancies
    identify_discrepancies(yearly_results, growth_analysis, params)
    
    # Save results for further analysis
    results = {
        'config_parameters': params,
        'yearly_results': {k: {key: val for key, val in v.items() 
                              if key not in ['hire_events', 'term_events']} 
                          for k, v in yearly_results.items()},
        'rate_analysis': rate_analysis,
        'growth_analysis': growth_analysis
    }
    
    output_file = Path("campaign4_hiring_termination_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
