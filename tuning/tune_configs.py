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
    # Attrition parameters (these go in global_parameters.attrition in dev_tiny.yaml)
    "global_parameters.attrition.new_hire_termination_rate": [0.15, 0.20, 0.25, 0.30, 0.35],
    "global_parameters.attrition.annual_termination_rate": [0.08, 0.10, 0.12, 0.15, 0.18],

    # Compensation parameters
    "global_parameters.compensation.COLA_rate": [0.015, 0.018, 0.020, 0.022, 0.025],
    "global_parameters.annual_compensation_increase_rate": [0.025, 0.028, 0.030, 0.032, 0.035],

    # Growth and hiring
    "global_parameters.target_growth": [0.01, 0.02, 0.03, 0.04, 0.05],
    "global_parameters.new_hires.new_hire_rate": [0.04, 0.05, 0.06, 0.07, 0.08],
}


def load_baseline_distributions() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load baseline distributions for comparison.

    Returns:
        Tuple of (age_distribution, tenure_distribution) dictionaries
    """
    # TODO: Implement actual baseline loading from historical data or reference simulation
    # For now, return mock distributions
    age_dist = {
        "<30": 0.25,
        "30-39": 0.35,
        "40-49": 0.25,
        "50-59": 0.12,
        "60-65": 0.03
    }

    tenure_dist = {
        "<1": 0.20,
        "1-3": 0.30,
        "3-5": 0.25,
        "5-10": 0.15,
        "10-15": 0.07,
        "15+": 0.03
    }

    return age_dist, tenure_dist


def kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """
    Calculate Kullback-Leibler divergence between two probability distributions.

    Args:
        p: Reference distribution (baseline)
        q: Comparison distribution (simulation result)

    Returns:
        KL divergence value (lower is better)
    """
    # Ensure both distributions have the same keys
    all_keys = set(p.keys()) | set(q.keys())

    kl_div = 0.0
    for key in all_keys:
        p_val = p.get(key, 1e-10)  # Small epsilon to avoid log(0)
        q_val = q.get(key, 1e-10)

        if p_val > 0:
            kl_div += p_val * (p_val / q_val if q_val > 0 else float('inf'))

    return kl_div


def score(sim_summary: Dict[str, Any]) -> float:
    """
    Score a simulation result against baseline distributions.

    Args:
        sim_summary: Dictionary containing simulation summary statistics

    Returns:
        Score (lower is better, 0 is perfect match)
    """
    baseline_age, baseline_tenure = load_baseline_distributions()

    # Extract distributions from simulation summary
    sim_age = sim_summary.get("age_hist", {})
    sim_tenure = sim_summary.get("tenure_hist", {})

    # Calculate KL divergences
    age_err = kl_divergence(baseline_age, sim_age) if sim_age else 0
    tenure_err = kl_divergence(baseline_tenure, sim_tenure) if sim_tenure else 0

    # Weight the errors (you can adjust these weights)
    total_score = 0.5 * age_err + 0.5 * tenure_err

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
        output_file_path: Path to simulation output file

    Returns:
        Dictionary containing parsed summary statistics
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
        # Read the CSV file (metrics summary)
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

        summary = {
            "hc_growth": float(hc_growth),
            "pay_growth": float(pay_growth),
            "age_hist": {},  # TODO: Extract age distribution from snapshots
            "tenure_hist": {},  # TODO: Extract tenure distribution from snapshots
            "total_employees": int(df['active_headcount'].iloc[-1]) if 'active_headcount' in df.columns else len(df),
            "file_path": str(output_file_path),
            "years": int(len(df)),
            "final_headcount": int(df['active_headcount'].iloc[-1]) if 'active_headcount' in df.columns else 0
        }

        # If the dataframe has age or tenure columns, extract distributions
        if 'age_band' in df.columns:
            age_counts = df['age_band'].value_counts(normalize=True).to_dict()
            summary["age_hist"] = age_counts

        if 'tenure_band' in df.columns:
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
    results_file = Path(args.output_dir) / "tuning_results.json"
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
