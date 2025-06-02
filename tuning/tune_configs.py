#!/usr/bin/env python3
"""
Auto-tuning script for cost model configurations.

This script automatically searches for optimal configuration parameters by:
1. Defining a search space of configuration parameters
2. Iteratively generating random configurations within that space
3. Running simulations with each configuration
4. Scoring results against baseline distributions
5. Finding the best-performing configuration

Usage:
    python tuning/tune_configs.py [--iterations 25] [--output-dir tuned/]
"""

import argparse
import json
import subprocess
import yaml
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import random
import shutil

# Try to import pandas for parquet file reading
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available. Summary parsing will be limited.")

# Configuration paths - adjust these for your project structure
TEMPLATE = Path("config/dev_tiny.yaml")  # Base configuration template
HAZARD = Path("config/hazard_defaults.yaml")  # Hazard defaults
RUNNER = Path("scripts/run_simulation.py")  # Main simulation script
OUTPUT_BASE = Path("output")  # Base output directory

# Default arguments for run_simulation.py
DEFAULT_SCENARIO = "baseline"
DEFAULT_CENSUS_PATH = "data/census_template.parquet"

# Search space for configuration parameters
# Each key maps to a list of possible values to sample from
SEARCH_SPACE = {
    # === EXISTING GLOBAL PARAMETERS ===
    # Growth and hiring
    "global_parameters.target_growth": [0.01, 0.02, 0.03, 0.04, 0.05],
    "global_parameters.annual_compensation_increase_rate": [0.025, 0.028, 0.030, 0.032, 0.035],

    # === DETAILED HAZARD PARAMETERS ===
    # Termination hazard parameters
    "global_parameters.termination_hazard.base_rate_for_new_hire": [0.20, 0.25, 0.30, 0.35, 0.40],
    "global_parameters.termination_hazard.level_discount_factor": [0.05, 0.08, 0.10, 0.12, 0.15],
    "global_parameters.termination_hazard.min_level_discount_multiplier": [0.3, 0.4, 0.5, 0.6],

    # Termination tenure multipliers
    "global_parameters.termination_hazard.tenure_multipliers.<1": [0.8, 1.0, 1.2],
    "global_parameters.termination_hazard.tenure_multipliers.1-3": [0.4, 0.6, 0.8],
    "global_parameters.termination_hazard.tenure_multipliers.3-5": [0.3, 0.4, 0.5],
    "global_parameters.termination_hazard.tenure_multipliers.5-10": [0.2, 0.28, 0.35],
    "global_parameters.termination_hazard.tenure_multipliers.10-15": [0.15, 0.20, 0.25],
    "global_parameters.termination_hazard.tenure_multipliers.15+": [0.2, 0.24, 0.3],

    # Termination age multipliers
    "global_parameters.termination_hazard.age_multipliers.<30": [0.6, 0.8, 1.0],
    "global_parameters.termination_hazard.age_multipliers.30-39": [0.8, 1.0, 1.2],
    "global_parameters.termination_hazard.age_multipliers.40-49": [0.9, 1.1, 1.3],
    "global_parameters.termination_hazard.age_multipliers.50-59": [1.1, 1.3, 1.5],
    "global_parameters.termination_hazard.age_multipliers.60-65": [1.5, 2.0, 2.5],

    # Promotion hazard parameters
    "global_parameters.promotion_hazard.base_rate": [0.08, 0.10, 0.12, 0.15],
    "global_parameters.promotion_hazard.level_dampener_factor": [0.10, 0.15, 0.20],

    # Promotion tenure multipliers
    "global_parameters.promotion_hazard.tenure_multipliers.<1": [0.3, 0.5, 0.7],
    "global_parameters.promotion_hazard.tenure_multipliers.1-3": [1.2, 1.5, 1.8],
    "global_parameters.promotion_hazard.tenure_multipliers.3-5": [1.5, 2.0, 2.5],
    "global_parameters.promotion_hazard.tenure_multipliers.5-10": [0.8, 1.0, 1.2],
    "global_parameters.promotion_hazard.tenure_multipliers.10-15": [0.2, 0.3, 0.4],
    "global_parameters.promotion_hazard.tenure_multipliers.15+": [0.05, 0.1, 0.15],

    # Promotion age multipliers
    "global_parameters.promotion_hazard.age_multipliers.<30": [1.2, 1.4, 1.6],
    "global_parameters.promotion_hazard.age_multipliers.30-39": [1.0, 1.1, 1.3],
    "global_parameters.promotion_hazard.age_multipliers.40-49": [0.7, 0.9, 1.1],
    "global_parameters.promotion_hazard.age_multipliers.50-59": [0.3, 0.4, 0.5],
    "global_parameters.promotion_hazard.age_multipliers.60-65": [0.05, 0.1, 0.15],

    # Compensation raises parameters
    "global_parameters.raises_hazard.merit_base": [0.025, 0.03, 0.035, 0.04],
    "global_parameters.raises_hazard.merit_tenure_bump_value": [0.003, 0.005, 0.007],
    "global_parameters.raises_hazard.merit_low_level_bump_value": [0.003, 0.005, 0.007],
    "global_parameters.raises_hazard.promotion_raise": [0.10, 0.12, 0.15],

    # COLA parameters by year
    "global_parameters.cola_hazard.by_year.2025": [0.015, 0.018, 0.020, 0.022, 0.025],
    "global_parameters.cola_hazard.by_year.2026": [0.014, 0.016, 0.018, 0.020, 0.022],
    "global_parameters.cola_hazard.by_year.2027": [0.012, 0.014, 0.016, 0.018, 0.020],
    "global_parameters.cola_hazard.by_year.2028": [0.010, 0.012, 0.015, 0.017, 0.019],
    "global_parameters.cola_hazard.by_year.2029": [0.008, 0.010, 0.014, 0.016, 0.018],
}


def load_baseline_distributions() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load baseline distributions and target values for comparison.

    This function loads target distributions from a configuration file or
    uses default values based on typical workforce demographics.

    Returns:
        Tuple of (age_distribution, tenure_distribution) dictionaries
    """
    # Try to load from a baseline configuration file
    baseline_config_path = Path("config/tuning_baseline.yaml")

    if baseline_config_path.exists():
        try:
            with open(baseline_config_path, 'r') as f:
                baseline_config = yaml.safe_load(f)

            age_dist = baseline_config.get("target_age_distribution", {})
            tenure_dist = baseline_config.get("target_tenure_distribution", {})

            print(f"Loaded baseline distributions from: {baseline_config_path}")
            return age_dist, tenure_dist

        except Exception as e:
            print(f"Error loading baseline config from {baseline_config_path}: {e}")
            print("Falling back to default distributions")

    # Default baseline distributions based on typical workforce demographics
    # These should be calibrated against actual company data
    print("Using default baseline distributions")

    age_dist = {
        "<30": 0.20,      # 20% under 30 (early career)
        "30-39": 0.30,    # 30% in 30s (core workforce)
        "40-49": 0.30,    # 30% in 40s (experienced)
        "50-59": 0.15,    # 15% in 50s (senior)
        "60-65": 0.04,    # 4% near retirement
        "65+": 0.01       # 1% past retirement age
    }

    tenure_dist = {
        "<1": 0.25,       # 25% new hires (< 1 year)
        "1-3": 0.30,      # 30% early tenure (1-3 years)
        "3-5": 0.20,      # 20% established (3-5 years)
        "5-10": 0.15,     # 15% experienced (5-10 years)
        "10-15": 0.07,    # 7% senior (10-15 years)
        "15+": 0.03       # 3% very senior (15+ years)
    }

    return age_dist, tenure_dist


def kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """
    Calculate Kullback-Leibler divergence between two probability distributions.

    KL(P||Q) = sum(P(i) * log(P(i) / Q(i))) for all i

    Args:
        p: Reference distribution (baseline)
        q: Comparison distribution (simulation result)

    Returns:
        KL divergence value (lower is better, 0 is perfect match)
    """
    # Ensure both distributions have the same keys
    all_keys = set(p.keys()) | set(q.keys())

    kl_div = 0.0
    for key in all_keys:
        p_val = p.get(key, 1e-10)  # Small epsilon to avoid log(0)
        q_val = q.get(key, 1e-10)

        if p_val > 0:
            if q_val > 0:
                # Standard KL divergence formula: p * log(p/q)
                kl_div += p_val * math.log(p_val / q_val)
            else:
                # If q_val is 0 but p_val > 0, KL divergence is infinite
                return float('inf')

    return kl_div


def score(sim_summary: Dict[str, Any]) -> float:
    """
    Score a simulation result against baseline distributions and target metrics.

    Args:
        sim_summary: Dictionary containing simulation summary statistics

    Returns:
        Score (lower is better, 0 is perfect match)
    """
    baseline_age, baseline_tenure = load_baseline_distributions()

    # Define target values for growth metrics
    TARGET_HC_GROWTH = 0.03  # 3% headcount growth target
    TARGET_PAY_GROWTH = 0.03  # 3% compensation growth target

    # Extract distributions from simulation summary
    sim_age = sim_summary.get("age_hist", {})
    sim_tenure = sim_summary.get("tenure_hist", {})
    sim_hc_growth = sim_summary.get("hc_growth", 0.0)
    sim_pay_growth = sim_summary.get("pay_growth", 0.0)

    # Calculate KL divergences for distributions
    # Use a large penalty value instead of infinity for missing distributions
    MISSING_DISTRIBUTION_PENALTY = 10.0

    if sim_age:
        age_err = kl_divergence(baseline_age, sim_age)
    else:
        age_err = MISSING_DISTRIBUTION_PENALTY
        print("Warning: Age distribution missing, using penalty score")

    if sim_tenure:
        tenure_err = kl_divergence(baseline_tenure, sim_tenure)
    else:
        tenure_err = MISSING_DISTRIBUTION_PENALTY
        print("Warning: Tenure distribution missing, using penalty score")

    # Calculate growth errors (absolute difference from targets)
    hc_growth_err = abs(sim_hc_growth - TARGET_HC_GROWTH)
    pay_growth_err = abs(sim_pay_growth - TARGET_PAY_GROWTH)

    # Define weights for different error components
    # These can be adjusted based on relative importance
    WEIGHT_AGE = 0.25
    WEIGHT_TENURE = 0.25
    WEIGHT_HC_GROWTH = 0.25
    WEIGHT_PAY_GROWTH = 0.25

    # Calculate weighted total score
    total_score = (
        WEIGHT_AGE * age_err +
        WEIGHT_TENURE * tenure_err +
        WEIGHT_HC_GROWTH * hc_growth_err +
        WEIGHT_PAY_GROWTH * pay_growth_err
    )

    # Log the component scores for debugging
    print(f"Score components: age_err={age_err:.4f}, tenure_err={tenure_err:.4f}, "
          f"hc_growth_err={hc_growth_err:.4f}, pay_growth_err={pay_growth_err:.4f}")

    return total_score


def set_nested(config: Dict[str, Any], key: str, value: Any) -> None:
    """
    Set a nested configuration value using dot notation.

    Args:
        config: Configuration dictionary to modify
        key: Dot-separated key path (e.g., "global_parameters.target_growth")
        value: Value to set
    """
    keys = key.split('.')
    current = config

    # Navigate to the parent of the target key
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]

    # Set the final value
    current[keys[-1]] = value


def iterate_configs(n: int = 25) -> List[Path]:
    """
    Generate n random configurations by sampling from the search space.

    Args:
        n: Number of configurations to generate

    Yields:
        Path to each generated configuration file
    """
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Template configuration not found: {TEMPLATE}")

    # Create output directory
    output_dir = Path("tuned")
    output_dir.mkdir(exist_ok=True)

    config_paths = []

    for i in range(n):
        # Load base template
        with open(TEMPLATE, 'r') as f:
            config = yaml.safe_load(f)

        # Sample random values from search space
        for param_key, possible_values in SEARCH_SPACE.items():
            value = random.choice(possible_values)
            set_nested(config, param_key, value)

        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_path = output_dir / f"config_{i:03d}_{timestamp}.yaml"

        # Write configuration
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        config_paths.append(config_path)
        print(f"Generated config {i+1}/{n}: {config_path}")

    return config_paths


def parse_summary(output_file_path: Path) -> Dict[str, Any]:
    """
    Parse simulation output to extract summary statistics.

    Args:
        output_file_path: Path to simulation output file (metrics CSV/parquet)

    Returns:
        Dictionary containing parsed summary statistics including age_hist and tenure_hist
    """
    if not HAS_PANDAS:
        print("Warning: pandas not available, returning mock data")
        return {
            "age_hist": {"<30": 0.25, "30-39": 0.35, "40-49": 0.25, "50-59": 0.12, "60-65": 0.03},
            "tenure_hist": {"<1": 0.20, "1-3": 0.30, "3-5": 0.25, "5-10": 0.15, "10-15": 0.07, "15+": 0.03},
            "hc_growth": 0.03,
            "pay_growth": 0.045
        }

    try:
        # Read the metrics summary file (CSV or parquet)
        if str(output_file_path).endswith('.csv'):
            df = pd.read_csv(output_file_path)
        else:
            df = pd.read_parquet(output_file_path)

        # Extract basic statistics from metrics summary
        if 'active_headcount' in df.columns and len(df) > 1:
            # Calculate headcount growth from first to last year
            initial_hc = df['active_headcount'].iloc[0]
            final_hc = df['active_headcount'].iloc[-1]
            hc_growth = (final_hc - initial_hc) / initial_hc if initial_hc > 0 else 0.0

            # Calculate compensation growth
            if 'avg_compensation' in df.columns:
                initial_comp = df['avg_compensation'].iloc[0]
                final_comp = df['avg_compensation'].iloc[-1]
                pay_growth = (final_comp - initial_comp) / initial_comp if initial_comp > 0 else 0.0
            else:
                pay_growth = 0.0
        else:
            hc_growth = 0.0
            pay_growth = 0.0

        # Initialize summary with basic metrics
        summary = {
            "hc_growth": float(hc_growth),
            "pay_growth": float(pay_growth),
            "age_hist": {},
            "tenure_hist": {},
            "total_employees": int(df['active_headcount'].iloc[-1]) if 'active_headcount' in df.columns else len(df),
            "file_path": str(output_file_path),
            "years": int(len(df)),
            "final_headcount": int(df['active_headcount'].iloc[-1]) if 'active_headcount' in df.columns else 0
        }

        # Try to extract age and tenure distributions from detailed snapshots
        output_dir = output_file_path.parent
        consolidated_snapshots_path = output_dir / "consolidated_snapshots.parquet"

        snapshots_df = None
        data_source = None

        # First try consolidated snapshots
        if consolidated_snapshots_path.exists():
            print(f"Loading detailed snapshot data from: {consolidated_snapshots_path}")
            try:
                snapshots_df = pd.read_parquet(consolidated_snapshots_path)
                data_source = "consolidated"
            except Exception as e:
                print(f"Error loading consolidated snapshots: {e}")

        # If consolidated snapshots don't exist or don't have age data, try yearly snapshots
        if snapshots_df is None or 'employee_age_band' not in snapshots_df.columns:
            yearly_snapshots_dir = output_dir / "yearly_snapshots"
            if yearly_snapshots_dir.exists():
                print(f"Looking for yearly snapshots in: {yearly_snapshots_dir}")
                # Find the most recent snapshot file
                snapshot_files = list(yearly_snapshots_dir.glob("*.parquet"))
                if snapshot_files:
                    # Sort by filename to get the latest year
                    latest_snapshot = sorted(snapshot_files)[-1]
                    print(f"Loading latest yearly snapshot: {latest_snapshot}")
                    try:
                        snapshots_df = pd.read_parquet(latest_snapshot)
                        data_source = "yearly"
                    except Exception as e:
                        print(f"Error loading yearly snapshot: {e}")

        if snapshots_df is not None:
            # Get the final year's data for distribution analysis
            if 'simulation_year' in snapshots_df.columns:
                final_year = snapshots_df['simulation_year'].max()
                final_year_data = snapshots_df[snapshots_df['simulation_year'] == final_year]
            elif 'year' in snapshots_df.columns:
                final_year = snapshots_df['year'].max()
                final_year_data = snapshots_df[snapshots_df['year'] == final_year]
            else:
                # Use all data if no year column found
                final_year_data = snapshots_df

            print(f"Analyzing {len(final_year_data)} employees from {data_source} snapshots")

            # Extract tenure distribution
            tenure_col_candidates = ['employee_tenure_band', 'tenure_band']
            tenure_col = None
            for col in tenure_col_candidates:
                if col in final_year_data.columns:
                    tenure_col = col
                    break

            if tenure_col is not None:
                tenure_counts = final_year_data[tenure_col].value_counts(normalize=True)
                # Map tenure bands to standard format
                tenure_mapping = {
                    '<1': '<1', '0-1': '<1',
                    '1-3': '1-3',
                    '3-5': '3-5',
                    '5-10': '5-10', '5+': '5-10',  # Handle legacy 5+ format
                    '10-15': '10-15',
                    '15+': '15+'
                }

                tenure_hist = {}
                for band, count in tenure_counts.items():
                    mapped_band = tenure_mapping.get(str(band), str(band))
                    tenure_hist[mapped_band] = float(count)

                summary["tenure_hist"] = tenure_hist
                print(f"Extracted tenure distribution: {tenure_hist}")
            else:
                print("Warning: No tenure band column found in snapshots")

            # Extract age distribution (if available)
            age_col_candidates = ['employee_age_band', 'age_band']
            age_col = None
            for col in age_col_candidates:
                if col in final_year_data.columns:
                    age_col = col
                    break

            if age_col is not None:
                age_counts = final_year_data[age_col].value_counts(normalize=True)
                age_hist = {str(band): float(count) for band, count in age_counts.items()}
                summary["age_hist"] = age_hist
                print(f"Extracted age distribution: {age_hist}")
            else:
                print("Warning: No age band column found in snapshots")
        else:
            print(f"No snapshot data found in {output_dir}")

        # Fallback: try to extract distributions from the metrics summary itself
        if not summary["age_hist"] and 'age_band' in df.columns:
            age_counts = df['age_band'].value_counts(normalize=True).to_dict()
            summary["age_hist"] = age_counts

        if not summary["tenure_hist"] and 'tenure_band' in df.columns:
            tenure_counts = df['tenure_band'].value_counts(normalize=True).to_dict()
            summary["tenure_hist"] = tenure_counts

        return summary

    except Exception as e:
        print(f"Error parsing summary file {output_file_path}: {e}")
        # Return mock data as fallback
        return {
            "age_hist": {"<30": 0.25, "30-39": 0.35, "40-49": 0.25, "50-59": 0.12, "60-65": 0.03},
            "tenure_hist": {"<1": 0.20, "1-3": 0.30, "3-5": 0.25, "5-10": 0.15, "10-15": 0.07, "15+": 0.03},
            "hc_growth": 0.03,
            "pay_growth": 0.045,
            "error": str(e)
        }


def run_and_summarize(cfg_path: Path) -> Dict[str, Any]:
    """
    Run simulation with given configuration and return summary.

    Args:
        cfg_path: Path to configuration file

    Returns:
        Dictionary containing simulation summary
    """
    # Construct output directory path
    output_dir = Path("tuned") / f"output_{cfg_path.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run simulation with correct arguments for run_simulation.py
    cmd = [
        "python", str(RUNNER),
        "--config", str(cfg_path),
        "--scenario", DEFAULT_SCENARIO,
        "--census", DEFAULT_CENSUS_PATH,
        "--output", str(output_dir),
        "--debug"  # Add debug flag for more verbose output
    ]

    print(f"Running simulation: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for longer simulations
            check=False   # Don't raise exception on non-zero exit
        )

        if result.returncode != 0:
            print(f"Simulation failed for {cfg_path.name} with exit code {result.returncode}")
            print(f"Stdout:\n{result.stdout}")
            print(f"Stderr:\n{result.stderr}")
            return {}

        print(f"Simulation completed successfully for {cfg_path.name}")

    except subprocess.TimeoutExpired:
        print(f"Simulation timed out for {cfg_path}")
        return {}
    except Exception as e:
        print(f"Error running simulation for {cfg_path}: {e}")
        return {}

    # Look for the expected output file from run_simulation.py
    # The simulation generates metrics CSV files in scenario subdirectories
    scenario_dir = output_dir / "Baseline"  # Default scenario name
    summary_file = scenario_dir / "Baseline_metrics.csv"

    if summary_file.exists():
        print(f"Found summary file: {summary_file}")
        return parse_summary(summary_file)
    else:
        print(f"Summary file not found: {summary_file}")
        print(f"Available files in {output_dir}:")
        if output_dir.exists():
            for file in output_dir.rglob("*"):  # Recursive search
                if file.is_file():
                    print(f"  {file}")

        # Return placeholder data for now
        return {
            "status": "success_no_summary",
            "path": str(output_dir),
            "hc_growth": 0.03,
            "pay_growth": 0.03,
            "age_hist": {},
            "tenure_hist": {}
        }


def main():
    """Main tuning loop."""
    parser = argparse.ArgumentParser(description="Auto-tune cost model configurations")
    parser.add_argument("--iterations", "-n", type=int, default=25,
                       help="Number of configurations to test")
    parser.add_argument("--output-dir", type=str, default="tuned",
                       help="Directory to store generated configurations")

    args = parser.parse_args()

    print(f"Starting auto-tuning with {args.iterations} iterations...")
    print(f"Template: {TEMPLATE}")
    print(f"Runner: {RUNNER}")
    print(f"Search space has {len(SEARCH_SPACE)} parameters")

    # Generate configurations
    config_paths = iterate_configs(args.iterations)

    best_score = float('inf')
    best_config = None
    results = []

    # Test each configuration
    for i, cfg_path in enumerate(config_paths):
        print(f"\nTesting configuration {i+1}/{len(config_paths)}: {cfg_path}")

        # Run simulation
        summary = run_and_summarize(cfg_path)
        if not summary:
            print(f"Skipping {cfg_path} due to simulation failure")
            continue

        # Score results
        score_val = score(summary)
        results.append({
            "config_path": str(cfg_path),
            "score": score_val,
            "summary": summary
        })

        print(f"Score: {score_val:.4f}")

        # Track best configuration
        if score_val < best_score:
            best_score = score_val
            best_config = cfg_path
            print(f"New best score: {score_val:.4f}")

    # Save results
    results_dir = Path(args.output_dir)
    results_dir.mkdir(exist_ok=True)  # Ensure output directory exists
    results_file = results_dir / "tuning_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Copy best configuration
    if best_config:
        best_config_dest = Path(args.output_dir) / "best_config.yaml"
        shutil.copy2(best_config, best_config_dest)

        print(f"\nTuning complete!")
        print(f"Best configuration: {best_config}")
        print(f"Best score: {best_score:.4f}")
        print(f"Best config copied to: {best_config_dest}")
        print(f"Full results saved to: {results_file}")
    else:
        print("\nNo successful configurations found!")


if __name__ == "__main__":
    main()
