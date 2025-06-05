#!/usr/bin/env python3
"""
Headcount Management Logic Investigation Script

This script implements the investigation plan from Epic Y to diagnose
the persistent negative headcount growth issue despite aggressive parameter tuning.

Tasks:
1. Quantify experienced employee attrition impact
2. Test maintain_headcount: true mode
3. Validate new hire termination rate assumptions
4. Order-of-operations validation

Usage:
    python scripts/investigate_headcount_logic.py
"""

import sys
import pandas as pd
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cost_model.engines.run_one_year import run_one_year
from cost_model.projections.snapshot import create_initial_snapshot
from cost_model.projections.dynamic_hazard import build_dynamic_hazard_table
from logging_config import get_logger

logger = get_logger(__name__)


def load_campaign_6_best_config() -> Tuple[Dict[str, Any], str]:
    """Load the best configuration from Campaign 6."""
    config_path = project_root / "campaign_6_results" / "best_config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Campaign 6 best config not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded Campaign 6 best config from {config_path}")
    return config, str(config_path)


def extract_termination_analysis(events_df: pd.DataFrame, snapshots: List[pd.DataFrame]) -> Dict[str, Any]:
    """Extract detailed termination analysis from simulation results."""
    
    # Filter termination events
    term_events = events_df[events_df['event_type'] == 'EVT_TERM'].copy()
    
    if term_events.empty:
        logger.warning("No termination events found in simulation results")
        return {}
    
    # Add year column for analysis
    term_events['year'] = pd.to_datetime(term_events['event_time']).dt.year
    
    analysis = {
        'total_terminations': len(term_events),
        'terminations_by_year': term_events['year'].value_counts().sort_index().to_dict(),
        'termination_reasons': {}
    }
    
    # Analyze termination reasons if available in value_json
    if 'value_json' in term_events.columns:
        import json
        reasons = []
        for _, row in term_events.iterrows():
            try:
                if pd.notna(row['value_json']):
                    reason_data = json.loads(row['value_json'])
                    if isinstance(reason_data, dict) and 'reason' in reason_data:
                        reasons.append(reason_data['reason'])
                    else:
                        reasons.append('unknown')
                else:
                    reasons.append('no_reason')
            except (json.JSONDecodeError, TypeError):
                reasons.append('parse_error')
        
        analysis['termination_reasons'] = pd.Series(reasons).value_counts().to_dict()
    
    return analysis


def extract_hiring_analysis(events_df: pd.DataFrame) -> Dict[str, Any]:
    """Extract detailed hiring analysis from simulation results."""
    
    # Filter hiring events
    hire_events = events_df[events_df['event_type'] == 'EVT_HIRE'].copy()
    
    if hire_events.empty:
        logger.warning("No hiring events found in simulation results")
        return {}
    
    # Add year column for analysis
    hire_events['year'] = pd.to_datetime(hire_events['event_time']).dt.year
    
    analysis = {
        'total_hires': len(hire_events),
        'hires_by_year': hire_events['year'].value_counts().sort_index().to_dict(),
    }
    
    return analysis


def analyze_headcount_progression(snapshots: List[pd.DataFrame]) -> Dict[str, Any]:
    """Analyze headcount progression across simulation years."""
    
    progression = {
        'headcount_by_year': {},
        'growth_rates': {},
        'active_employees': {}
    }
    
    for i, snapshot in enumerate(snapshots):
        year = 2025 + i  # Assuming simulation starts in 2025
        
        # Count total employees
        total_count = len(snapshot)
        progression['headcount_by_year'][year] = total_count
        
        # Count active employees if column exists
        if 'employee_active' in snapshot.columns:
            active_count = snapshot['employee_active'].sum()
            progression['active_employees'][year] = int(active_count)
        else:
            progression['active_employees'][year] = total_count
        
        # Calculate growth rate
        if i > 0:
            prev_count = progression['active_employees'][year - 1]
            growth_rate = (progression['active_employees'][year] - prev_count) / prev_count
            progression['growth_rates'][year] = growth_rate
    
    return progression


def run_diagnostic_simulation(config: Dict[str, Any], config_name: str) -> Dict[str, Any]:
    """Run simulation with diagnostic logging and analysis."""

    logger.info(f"Running diagnostic simulation with config: {config_name}")

    # Extract key parameters for logging
    global_params = config.get('global_parameters', {})
    target_growth = global_params.get('target_growth', 'unknown')
    new_hire_rate = global_params.get('new_hire_rate', 'unknown')
    nh_term_rate = global_params.get('new_hire_termination_rate', 'unknown')
    maintain_headcount = global_params.get('maintain_headcount', False)

    logger.info(f"Key parameters - target_growth: {target_growth}, new_hire_rate: {new_hire_rate}, "
                f"new_hire_termination_rate: {nh_term_rate}, maintain_headcount: {maintain_headcount}")

    try:
        # Convert config dict to namespace for compatibility
        from types import SimpleNamespace

        def dict_to_namespace(d):
            if isinstance(d, dict):
                return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
            return d

        config_ns = dict_to_namespace(config)

        # Create initial snapshot from census
        census_path = project_root / "data" / "census_template.parquet"
        if not census_path.exists():
            raise FileNotFoundError(f"Census template not found at {census_path}")

        start_year = global_params.get('start_year', 2025)
        projection_years = global_params.get('projection_years', 3)

        # Initialize simulation components
        snapshot = create_initial_snapshot(start_year, census_path)
        event_log = pd.DataFrame()  # Start with empty event log

        # Build hazard table
        hazard_table = build_dynamic_hazard_table(config_ns.global_parameters)

        # Set up RNG
        rng = np.random.default_rng(global_params.get('random_seed', 42))

        # Run multi-year simulation
        all_events = []
        snapshots = []

        for year_offset in range(projection_years):
            year = start_year + year_offset
            logger.info(f"Simulating year {year}")

            # Run one year
            year_events, final_snapshot = run_one_year(
                event_log=event_log,
                prev_snapshot=snapshot,
                year=year,
                global_params=config_ns.global_parameters,
                plan_rules=config_ns.plan_rules.__dict__ if hasattr(config_ns, 'plan_rules') else {},
                hazard_table=hazard_table,
                rng=rng,
                deterministic_term=True
            )

            # Store results
            all_events.append(year_events)
            snapshots.append(final_snapshot)

            # Update for next year
            event_log = year_events
            snapshot = final_snapshot

        # Combine all events
        events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()

        # Extract analyses
        termination_analysis = extract_termination_analysis(events_df, snapshots)
        hiring_analysis = extract_hiring_analysis(events_df)
        headcount_analysis = analyze_headcount_progression(snapshots)

        # Combine results
        results = {
            'config_name': config_name,
            'parameters': {
                'target_growth': target_growth,
                'new_hire_rate': new_hire_rate,
                'new_hire_termination_rate': nh_term_rate,
                'maintain_headcount': maintain_headcount
            },
            'termination_analysis': termination_analysis,
            'hiring_analysis': hiring_analysis,
            'headcount_analysis': headcount_analysis,
            'events_count': len(events_df),
            'final_headcount': len(snapshots[-1]) if snapshots else 0
        }

        logger.info(f"Simulation completed successfully. Final headcount: {results['final_headcount']}")
        return results

    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise


def test_maintain_headcount_mode(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Test the maintain_headcount: true mode."""
    
    logger.info("Testing maintain_headcount: true mode")
    
    # Create modified config
    modified_config = base_config.copy()
    if 'global_parameters' not in modified_config:
        modified_config['global_parameters'] = {}
    
    modified_config['global_parameters']['maintain_headcount'] = True
    
    # Run simulation with maintain_headcount enabled
    results = run_diagnostic_simulation(modified_config, "maintain_headcount_true")
    
    return results


def main():
    """Main investigation function."""
    
    logger.info("Starting headcount management logic investigation")
    
    try:
        # Task 1: Load Campaign 6 best config and run diagnostic
        logger.info("=== TASK 1: Campaign 6 Best Config Analysis ===")
        base_config, config_path = load_campaign_6_best_config()
        
        baseline_results = run_diagnostic_simulation(base_config, "campaign_6_best")
        
        # Task 2: Test maintain_headcount mode
        logger.info("=== TASK 2: Testing maintain_headcount: true Mode ===")
        maintain_headcount_results = test_maintain_headcount_mode(base_config)
        
        # Task 3 & 4: Analysis and comparison
        logger.info("=== TASK 3 & 4: Analysis and Comparison ===")
        
        # Compare results
        baseline_final = baseline_results['final_headcount']
        maintain_final = maintain_headcount_results['final_headcount']
        
        baseline_growth = baseline_results['headcount_analysis']['growth_rates']
        maintain_growth = maintain_headcount_results['headcount_analysis']['growth_rates']
        
        logger.info(f"Baseline final headcount: {baseline_final}")
        logger.info(f"Maintain headcount final: {maintain_final}")
        logger.info(f"Baseline growth rates: {baseline_growth}")
        logger.info(f"Maintain headcount growth rates: {maintain_growth}")
        
        # Save detailed results
        output_dir = project_root / "investigation_results"
        output_dir.mkdir(exist_ok=True)
        
        import json
        with open(output_dir / "baseline_results.json", 'w') as f:
            json.dump(baseline_results, f, indent=2, default=str)
        
        with open(output_dir / "maintain_headcount_results.json", 'w') as f:
            json.dump(maintain_headcount_results, f, indent=2, default=str)
        
        logger.info(f"Detailed results saved to {output_dir}")
        
        # Summary report
        print("\n" + "="*80)
        print("HEADCOUNT MANAGEMENT INVESTIGATION SUMMARY")
        print("="*80)
        print(f"Campaign 6 Best Config: {config_path}")
        print(f"Target Growth: {baseline_results['parameters']['target_growth']}")
        print(f"New Hire Termination Rate: {baseline_results['parameters']['new_hire_termination_rate']}")
        print()
        print("BASELINE RESULTS:")
        print(f"  Final Headcount: {baseline_final}")
        print(f"  Growth Rates: {baseline_growth}")
        print(f"  Total Hires: {baseline_results['hiring_analysis'].get('total_hires', 'N/A')}")
        print(f"  Total Terminations: {baseline_results['termination_analysis'].get('total_terminations', 'N/A')}")
        print()
        print("MAINTAIN_HEADCOUNT=TRUE RESULTS:")
        print(f"  Final Headcount: {maintain_final}")
        print(f"  Growth Rates: {maintain_growth}")
        print(f"  Total Hires: {maintain_headcount_results['hiring_analysis'].get('total_hires', 'N/A')}")
        print(f"  Total Terminations: {maintain_headcount_results['termination_analysis'].get('total_terminations', 'N/A')}")
        print()
        print(f"Detailed results saved to: {output_dir}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Investigation failed: {e}")
        raise


if __name__ == "__main__":
    main()
