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
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import random
import shutil

# Add project root to Python path to enable imports when running from tuning/ directory
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Try to import pandas for parquet file reading
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available. Summary parsing will be limited.")

# Import age calculation utilities
try:
    from cost_model.utils.date_utils import calculate_age, age_to_band
    from cost_model.state.schema import EMP_BIRTH_DATE
    HAS_AGE_UTILS = True
    print("SUCCESS: Age calculation utilities imported successfully!")
except ImportError as e:
    HAS_AGE_UTILS = False
    print(f"Warning: Age calculation utilities not available: {e}")
    print("Using fallback age distribution.")

# Configuration paths - adjust these for your project structure
# Use project_root to make paths relative to project root, not tuning directory
TEMPLATE = project_root / "config/dev_tiny.yaml"  # Base configuration template
HAZARD = project_root / "config/hazard_defaults.yaml"  # Hazard defaults
RUNNER = project_root / "scripts/run_simulation.py"  # Main simulation script
OUTPUT_BASE = project_root / "output"  # Base output directory

# Default arguments for run_simulation.py
DEFAULT_SCENARIO = "baseline"
DEFAULT_CENSUS_PATH = project_root / "data/census_template.parquet"

# Search space for configuration parameters
# Each key maps to a list of possible values to sample from
#
# UPDATED based on Epic 3 User Story 3.3 Analysis (200-iteration campaign results):
# - Added critical missing parameters: new_hire_rate, new_hire_average_age, max_working_age
# - Expanded ranges for better young workforce retention and demographic targeting
# - Lowered new hire termination rates and strengthened young employee protection
# - See docs/auto_calibration/epic_3.md for detailed analysis and rationale
SEARCH_SPACE = {
    # === EXISTING GLOBAL PARAMETERS ===
    # Growth and hiring - REFINED RANGES based on Campaign 1 analysis (Epic 3 User Story 3.3)
    # Issue: Best config achieved 0% growth vs 3% target - need higher minimum targets
    "global_parameters.target_growth": [0.025, 0.030, 0.035, 0.040, 0.045, 0.050],  # Focused on 3% target
    # Issue: Pay growth overshot (5.3% vs 3% target) - tighten compensation ranges
    "global_parameters.annual_compensation_increase_rate": [0.020, 0.025, 0.028, 0.030, 0.032],  # Lower max

    # === CRITICAL MISSING PARAMETERS - REFINED based on Campaign 1 analysis ===
    # Issue: Need higher hiring rates to achieve 3% growth target
    "global_parameters.new_hires.new_hire_rate": [0.12, 0.15, 0.18, 0.20, 0.22],  # Higher minimum
    # Issue: Age distribution skewed away from <30 (3.5% vs 10.9% target) - need younger hires
    "global_parameters.new_hire_average_age": [25, 27, 28, 30, 32],  # Focus on younger range
    "global_parameters.new_hire_age_std_dev": [3, 5, 7, 9],  # Control age distribution spread
    "global_parameters.max_working_age": [62, 63, 64, 65],  # Was fixed at 65

    # NEW: Additional new hire age parameters for fine-grained control
    "global_parameters.compensation.new_hire.age_mean": [25, 30, 35, 40, 45],  # Alternative age control
    "global_parameters.compensation.new_hire.age_std": [5, 8, 10, 12],  # Alternative std dev control

    # === DETAILED HAZARD PARAMETERS ===
    # Issue: High <1 year tenure (45% vs 20% target) suggests excessive new hire termination
    # Termination hazard parameters - LOWER RANGES to reduce new hire churn
    "global_parameters.termination_hazard.base_rate_for_new_hire": [0.08, 0.10, 0.12, 0.15, 0.18],  # Lower max
    "global_parameters.termination_hazard.level_discount_factor": [0.05, 0.08, 0.10, 0.12, 0.15],
    "global_parameters.termination_hazard.min_level_discount_multiplier": [0.3, 0.4, 0.5, 0.6],

    # Issue: 45% <1 year tenure vs 20% target - need stronger protection for new hires
    # Termination tenure multipliers - LOWER <1 year rates to reduce new hire churn
    "global_parameters.termination_hazard.tenure_multipliers.<1": [0.5, 0.6, 0.8],  # Much lower max
    "global_parameters.termination_hazard.tenure_multipliers.1-3": [0.4, 0.6, 0.8],
    "global_parameters.termination_hazard.tenure_multipliers.3-5": [0.3, 0.4, 0.5],
    "global_parameters.termination_hazard.tenure_multipliers.5-10": [0.2, 0.28, 0.35],
    "global_parameters.termination_hazard.tenure_multipliers.10-15": [0.15, 0.20, 0.25],
    "global_parameters.termination_hazard.tenure_multipliers.15+": [0.2, 0.24, 0.3],

    # Issue: <30 age band too low (3.5% vs 10.9% target) - need stronger young employee protection
    # Termination age multipliers - STRONGER protection for young employees
    "global_parameters.termination_hazard.age_multipliers.<30": [0.2, 0.3, 0.4, 0.5],  # Even stronger protection
    "global_parameters.termination_hazard.age_multipliers.30-39": [0.8, 1.0, 1.2],
    "global_parameters.termination_hazard.age_multipliers.40-49": [0.9, 1.1, 1.3],
    "global_parameters.termination_hazard.age_multipliers.50-59": [1.1, 1.3, 1.5],
    "global_parameters.termination_hazard.age_multipliers.60-65": [2.0, 2.5, 3.0],  # Stronger retirement pressure

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

    # Issue: Pay growth overshot (5.3% vs 3% target) - need to constrain compensation increases
    # Compensation raises parameters - LOWER RANGES to control pay growth
    "global_parameters.raises_hazard.merit_base": [0.020, 0.025, 0.030, 0.032],  # Lower max
    "global_parameters.raises_hazard.merit_tenure_bump_value": [0.002, 0.003, 0.005],  # Lower max
    "global_parameters.raises_hazard.merit_low_level_bump_value": [0.002, 0.003, 0.005],  # Lower max
    "global_parameters.raises_hazard.promotion_raise": [0.08, 0.10, 0.12],  # Lower max

    # Issue: Pay growth overshot (5.3% vs 3% target) - need to constrain COLA increases
    # COLA parameters by year - LOWER RANGES to control overall compensation growth
    "global_parameters.cola_hazard.by_year.2025": [0.010, 0.012, 0.015, 0.018, 0.020],  # Lower max
    "global_parameters.cola_hazard.by_year.2026": [0.008, 0.010, 0.012, 0.015, 0.018],  # Lower max
    "global_parameters.cola_hazard.by_year.2027": [0.006, 0.008, 0.010, 0.012, 0.015],  # Lower max
    "global_parameters.cola_hazard.by_year.2028": [0.005, 0.007, 0.010, 0.012, 0.014],  # Lower max
    "global_parameters.cola_hazard.by_year.2029": [0.004, 0.006, 0.008, 0.010, 0.012],  # Lower max
}


def load_baseline_distributions() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load baseline distributions and target values for comparison.

    This function calculates the baseline age distribution from the actual starting
    census file to preserve the initial workforce demographics. The tenure distribution
    is loaded from configuration or uses defaults.

    Returns:
        Tuple of (age_distribution, tenure_distribution) dictionaries
    """
    print("Attempting to load baseline distributions...")
    age_dist_from_census = {}

    # --- Logic for age_dist from DEFAULT_CENSUS_PATH ---
    if HAS_PANDAS and HAS_AGE_UTILS and Path(DEFAULT_CENSUS_PATH).exists():
        try:
            print(f"Loading starting census for age baseline: {DEFAULT_CENSUS_PATH}")
            census_df = pd.read_parquet(DEFAULT_CENSUS_PATH)

            # Get simulation start year from template config
            sim_start_year = 2025  # Default fallback
            if TEMPLATE.exists():
                try:
                    with open(TEMPLATE, 'r') as f_template:
                        template_config = yaml.safe_load(f_template)
                        sim_start_year = template_config.get('global_parameters', {}).get('start_year', 2025)
                        print(f"Using simulation start year from template: {sim_start_year}")
                except Exception as e:
                    print(f"Warning: Could not read start_year from template: {e}")
                    print(f"Using default start year: {sim_start_year}")

            as_of_date_census = pd.Timestamp(f"{sim_start_year}-01-01")

            # Check for birth date column (try multiple possible names)
            birth_col_candidates = [EMP_BIRTH_DATE, 'employee_birth_date', 'birth_date']
            birth_col = None
            for col in birth_col_candidates:
                if col in census_df.columns:
                    birth_col = col
                    break

            if birth_col is not None:
                print(f"Found birth date column: {birth_col}")
                # Calculate ages using the imported utility function
                census_df['age_at_start'] = calculate_age(census_df[birth_col], as_of_date_census)

                # Convert ages to age bands using the imported utility function
                census_df['age_band_at_start'] = census_df['age_at_start'].apply(
                    lambda x: age_to_band(int(x)) if pd.notna(x) and x >= 0 else None
                )

                # Calculate normalized age distribution
                age_dist_from_census = census_df['age_band_at_start'].value_counts(normalize=True).to_dict()

                # Convert any ">65" bands to "65+" for consistency with scoring function
                if ">65" in age_dist_from_census:
                    age_dist_from_census["65+"] = age_dist_from_census.pop(">65")

                print(f"Successfully calculated age distribution from census: {age_dist_from_census}")
            else:
                print(f"Warning: Birth date column not found in census. Tried: {birth_col_candidates}")
                print("Cannot calculate age baseline from census.")
        except Exception as e:
            print(f"Error loading or processing census for age baseline: {e}")
            print("Falling back to default age distribution for safety.")
    else:
        if not HAS_PANDAS:
            print("Warning: pandas not available for census processing.")
        if not HAS_AGE_UTILS:
            print("Warning: Age calculation utilities not available.")
        if not Path(DEFAULT_CENSUS_PATH).exists():
            print(f"Warning: Default census file not found at {DEFAULT_CENSUS_PATH}.")
        print("Using default age distribution due to missing dependencies or census file.")

    # Fallback to default if census processing failed or census not found
    if not age_dist_from_census:
        age_dist_from_census = {
            "<30": 0.20,      # 20% under 30 (early career)
            "30-39": 0.30,    # 30% in 30s (core workforce)
            "40-49": 0.30,    # 30% in 40s (experienced)
            "50-59": 0.15,    # 15% in 50s (senior)
            "60-65": 0.04,    # 4% near retirement
            "65+": 0.01       # 1% past retirement age
        }
        print(f"Using fallback/default age distribution: {age_dist_from_census}")

    # --- Logic for tenure_dist (can remain as is for now) ---
    tenure_dist = {}
    baseline_config_path = project_root / "config/tuning_baseline.yaml"
    if baseline_config_path.exists():
        try:
            with open(baseline_config_path, 'r') as f:
                baseline_config = yaml.safe_load(f)
            tenure_dist = baseline_config.get("target_tenure_distribution", {})
            print(f"Loaded target tenure distribution from: {baseline_config_path}")
            if not tenure_dist:  # If key exists but is empty
                print("Warning: 'target_tenure_distribution' empty in baseline YAML. Using default tenure distribution.")
        except Exception as e:
            print(f"Error loading tenure baseline from {baseline_config_path}: {e}")

    if not tenure_dist:  # Fallback if file not found or key missing/empty
        print("Using default tenure distribution.")
        tenure_dist = {
            "<1": 0.25,       # 25% new hires (< 1 year)
            "1-3": 0.30,      # 30% early tenure (1-3 years)
            "3-5": 0.20,      # 20% established (3-5 years)
            "5-10": 0.15,     # 15% experienced (5-10 years)
            "10-15": 0.07,    # 7% senior (10-15 years)
            "15+": 0.03       # 3% very senior (15+ years)
        }

    return age_dist_from_census, tenure_dist


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
    # UPDATED for Campaign 2 based on Campaign 1 analysis (Epic 3 User Story 3.3):
    # - Increased HC_GROWTH weight due to significant miss (0% vs 3% target)
    # - Increased TENURE weight due to major imbalance (45% <1yr vs 20% target)
    # - Slightly reduced AGE weight to allow more flexibility for business targets
    # - Increased PAY_GROWTH weight due to overshoot (5.3% vs 3% target)
    WEIGHT_AGE = 0.30          # Important: age preservation with some flexibility for business needs
    WEIGHT_TENURE = 0.30       # Critical: tenure imbalance is major workforce stability issue
    WEIGHT_HC_GROWTH = 0.30    # Critical: business growth targets must be achieved
    WEIGHT_PAY_GROWTH = 0.10   # Moderate: compensation growth affects budget but less critical

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
            # Look for yearly snapshots in year=YYYY subdirectories
            year_dirs = list(output_dir.glob("year=*"))
            if year_dirs:
                # Sort by year to get the latest year
                latest_year_dir = sorted(year_dirs)[-1]
                snapshot_file = latest_year_dir / "snapshot.parquet"
                if snapshot_file.exists():
                    print(f"Loading latest yearly snapshot: {snapshot_file}")
                    try:
                        snapshots_df = pd.read_parquet(snapshot_file)
                        data_source = "yearly"
                    except Exception as e:
                        print(f"Error loading yearly snapshot: {e}")
                else:
                    print(f"No snapshot.parquet found in {latest_year_dir}")
            else:
                # Fallback: try the old yearly_snapshots directory structure
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
                    else:
                        print(f"No snapshot files found in {yearly_snapshots_dir}")
                else:
                    print(f"No yearly snapshots directory found: {yearly_snapshots_dir}")

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
        "--census", str(DEFAULT_CENSUS_PATH),
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
