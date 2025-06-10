#!/usr/bin/env python3
"""
Precision tuning V2 - Refined for exact 10bp targeting.
Addressing: Need higher HC growth (+50bp) and much less negative pay growth (+300bp)
"""

import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Project root for relative paths
project_root = Path(__file__).parent.parent

# Configuration paths
TEMPLATE = project_root / "config/dev_tiny.yaml"
RUNNER = project_root / "scripts/run_simulation.py"
DEFAULT_CENSUS_PATH = project_root / "data/census_template.parquet"

# PRECISION TARGETS (within 10 basis points)
TARGET_HC_GROWTH = 0.030  # 3.00%
TARGET_PAY_GROWTH = 0.000  # 0.00%
TOLERANCE = 0.001  # 10 basis points (0.10%)

# REFINED SEARCH SPACE V2 - Addressing campaign 1 results
# Need: +50bp HC growth, +300bp pay growth (less negative)
PRECISION_SEARCH_SPACE_V2 = {
    # ADJUSTED FOR HIGHER HC GROWTH (+50bp needed)
    "global_parameters.target_growth": [0.031, 0.032, 0.033, 0.034],  # Higher targets
    "global_parameters.new_hires.new_hire_rate": [0.52, 0.54, 0.56],  # Higher hiring
    # ADJUSTED FOR LESS NEGATIVE PAY GROWTH (+300bp needed)
    "global_parameters.annual_compensation_increase_rate": [
        0.032,
        0.034,
        0.036,
    ],  # Higher increases
    "global_parameters.raises_hazard.merit_base": [0.028, 0.030, 0.032, 0.034],  # Higher merit
    "global_parameters.raises_hazard.promotion_raise": [0.11, 0.12, 0.13],  # Higher promo raises
    # FINE-TUNED RETENTION (balance HC growth)
    "global_parameters.termination_hazard.base_rate_for_new_hire": [
        0.035,
        0.038,
        0.040,
    ],  # Lower terms
    # CONSERVATIVE OTHER PARAMETERS
    "global_parameters.promotion_hazard.base_rate": [0.100, 0.105],
    "global_parameters.promotion_hazard.level_dampener_factor": [0.15],
    # HIGHER COLA FOR BETTER PAY GROWTH
    "_cola_2025": [0.020, 0.022, 0.024],  # Higher COLA
    "_cola_2026": [0.018, 0.020, 0.022],
    "_cola_2027": [0.016, 0.018, 0.020],
    "_cola_2028": [0.014, 0.016, 0.018],
    "_cola_2029": [0.012, 0.014, 0.016],
    # STABLE DEMOGRAPHICS
    "global_parameters.new_hire_average_age": [26, 27],
    "global_parameters.new_hire_age_std_dev": [3],
    "global_parameters.max_working_age": [64, 65],
    # MINIMAL PARAMETER SET FOR FOCUSED TUNING
    "global_parameters.termination_hazard.level_discount_factor": [0.10],
    "global_parameters.termination_hazard.min_level_discount_multiplier": [0.5],
    # OPTIMIZED TENURE MULTIPLIERS
    "global_parameters.termination_hazard.tenure_multipliers.<1": [0.25],
    "global_parameters.termination_hazard.tenure_multipliers.1-3": [0.7],
    "global_parameters.termination_hazard.tenure_multipliers.3-5": [0.4],
    "global_parameters.termination_hazard.tenure_multipliers.5-10": [0.20],
    "global_parameters.termination_hazard.tenure_multipliers.10-15": [0.12],
    "global_parameters.termination_hazard.tenure_multipliers.15+": [0.24],
    # STANDARD AGE MULTIPLIERS
    "global_parameters.termination_hazard.age_multipliers.<30": [0.3],
    "global_parameters.termination_hazard.age_multipliers.30-39": [0.8],
    "global_parameters.termination_hazard.age_multipliers.40-49": [1.0],
    "global_parameters.termination_hazard.age_multipliers.50-59": [1.5],
    "global_parameters.termination_hazard.age_multipliers.60-65": [8.0],
    "global_parameters.termination_hazard.age_multipliers.65+": [12.0],
    # STANDARD PROMOTION MULTIPLIERS
    "global_parameters.promotion_hazard.tenure_multipliers.<1": [0.5],
    "global_parameters.promotion_hazard.tenure_multipliers.1-3": [1.5],
    "global_parameters.promotion_hazard.tenure_multipliers.3-5": [2.0],
    "global_parameters.promotion_hazard.tenure_multipliers.5-10": [1.0],
    "global_parameters.promotion_hazard.tenure_multipliers.10-15": [0.3],
    "global_parameters.promotion_hazard.tenure_multipliers.15+": [0.1],
    "global_parameters.promotion_hazard.age_multipliers.<30": [1.4],
    "global_parameters.promotion_hazard.age_multipliers.30-39": [1.1],
    "global_parameters.promotion_hazard.age_multipliers.40-49": [0.9],
    "global_parameters.promotion_hazard.age_multipliers.50-59": [0.4],
    "global_parameters.promotion_hazard.age_multipliers.60-65": [0.1],
    # MERIT COMPONENTS
    "global_parameters.raises_hazard.merit_tenure_bump_value": [0.003],
    "global_parameters.raises_hazard.merit_low_level_bump_value": [0.003],
}


def set_nested(config: Dict[str, Any], key: str, value: Any) -> None:
    """Set a nested configuration value using dot notation."""
    keys = key.split(".")
    current = config
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value


def set_cola_by_year_clean(config: Dict[str, Any], cola_values: Dict[str, float]) -> None:
    """Completely replace the cola_hazard.by_year section to eliminate duplicate keys."""
    if "global_parameters" not in config:
        config["global_parameters"] = {}
    if "cola_hazard" not in config["global_parameters"]:
        config["global_parameters"]["cola_hazard"] = {}

    # Build clean by_year section with only numeric keys
    by_year = {}
    for special_key, rate in cola_values.items():
        if special_key.startswith("_cola_"):
            year = int(special_key[6:])  # Extract year from '_cola_2025' -> 2025
            by_year[year] = rate

    # Completely replace the by_year section
    config["global_parameters"]["cola_hazard"]["by_year"] = by_year


def generate_precision_configs_v2(n: int = 30) -> List[Path]:
    """Generate refined precision configurations."""
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Template configuration not found: {TEMPLATE}")

    # Create output directory
    output_dir = Path("precision_tuned_v2")
    output_dir.mkdir(exist_ok=True)

    config_paths = []

    for i in range(n):
        # Load base template
        with open(TEMPLATE, "r") as f:
            config = yaml.safe_load(f)

        # Sample random values from refined search space
        cola_values = {}
        for param_key, possible_values in PRECISION_SEARCH_SPACE_V2.items():
            value = random.choice(possible_values)

            # Handle special COLA parameters separately
            if param_key.startswith("_cola_"):
                cola_values[param_key] = value
            else:
                set_nested(config, param_key, value)

        # Apply COLA values with special handling
        if cola_values:
            set_cola_by_year_clean(config, cola_values)

        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_path = output_dir / f"precision_v2_config_{i:03d}_{timestamp}.yaml"

        # Write configuration
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        config_paths.append(config_path)

    return config_paths


def run_simulation_quick(config_path: Path) -> Dict[str, Any]:
    """Run simulation and extract precise metrics."""
    import subprocess

    output_dir = Path("precision_tuned_v2") / f"output_{config_path.stem}"

    cmd = [
        "python3",
        str(RUNNER),
        "--config",
        str(config_path),
        "--scenario",
        "baseline",
        "--census",
        str(DEFAULT_CENSUS_PATH),
        "--output",
        str(output_dir),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return None

        # Parse results quickly
        metrics_file = output_dir / "Baseline/Baseline_metrics.csv"
        if metrics_file.exists():
            import pandas as pd

            df = pd.read_csv(metrics_file)
            if len(df) > 1:
                initial_hc = df["active_headcount"].iloc[0]
                final_hc = df["active_headcount"].iloc[-1]
                hc_growth = (final_hc - initial_hc) / initial_hc

                initial_comp = df["avg_compensation"].iloc[0]
                final_comp = df["avg_compensation"].iloc[-1]
                pay_growth = (final_comp - initial_comp) / initial_comp

                return {
                    "hc_growth": hc_growth,
                    "pay_growth": pay_growth,
                    "hc_diff_bp": abs(hc_growth - TARGET_HC_GROWTH) * 10000,
                    "pay_diff_bp": abs(pay_growth - TARGET_PAY_GROWTH) * 10000,
                    "within_10bp": (
                        abs(hc_growth - TARGET_HC_GROWTH) <= 0.001
                        and abs(pay_growth - TARGET_PAY_GROWTH) <= 0.001
                    ),
                }
        return None

    except Exception as e:
        return None


def main():
    """Run precision targeting campaign V2."""
    print("=== PRECISION TARGETING CAMPAIGN V2 ===")
    print("Addressing V1 results: Need +50bp HC growth, +300bp pay growth")
    print(f"Target: {TARGET_HC_GROWTH:.1%} ± 10bp HC growth")
    print(f"Target: {TARGET_PAY_GROWTH:.1%} ± 10bp pay growth")
    print()

    # Generate refined configurations
    config_paths = generate_precision_configs_v2(30)
    print(f"Generated {len(config_paths)} V2 precision configurations")
    print()

    # Run simulations with quick analysis
    results = []
    perfect_configs = []

    for i, config_path in enumerate(config_paths):
        print(f"Testing V2 config {i+1}/{len(config_paths)}: {config_path.name}")

        summary = run_simulation_quick(config_path)
        if summary:
            print(
                f"  HC Growth: {summary['hc_growth']:.3%} ({summary['hc_diff_bp']:.0f}bp from target)"
            )
            print(
                f"  Pay Growth: {summary['pay_growth']:.3%} ({summary['pay_diff_bp']:.0f}bp from target)"
            )

            if summary["within_10bp"]:
                print(f"  🎯 WITHIN 10BP TOLERANCE! 🎯")
                perfect_configs.append((config_path, summary))

            results.append(
                {
                    "config": str(config_path),
                    "hc_growth": summary["hc_growth"],
                    "pay_growth": summary["pay_growth"],
                    "hc_diff_bp": summary["hc_diff_bp"],
                    "pay_diff_bp": summary["pay_diff_bp"],
                    "total_diff_bp": summary["hc_diff_bp"] + summary["pay_diff_bp"],
                    "within_10bp": summary["within_10bp"],
                }
            )
            print()

    # Analysis
    print("=== PRECISION V2 RESULTS ===")
    print(f"Perfect configs (within 10bp): {len(perfect_configs)}")

    if perfect_configs:
        print("\\n🎯 PERFECT CONFIGURATIONS:")
        for config_path, summary in perfect_configs:
            print(f"  {config_path.name}")
            print(f"    HC: {summary['hc_growth']:.3%}, Pay: {summary['pay_growth']:.3%}")

            # Copy the first perfect config
            best_dest = Path("precision_tuned_v2/PERFECT_10BP_CONFIG.yaml")
            shutil.copy2(config_path, best_dest)
            print(f"    ✅ COPIED TO: {best_dest}")
            break

    if results:
        # Find closest even if not perfect
        best = min(results, key=lambda x: x["total_diff_bp"])
        print(f"\\n🏆 CLOSEST RESULT:")
        print(f"  Config: {best['config']}")
        print(f"  HC Growth: {best['hc_growth']:.3%} ({best['hc_diff_bp']:.0f}bp off)")
        print(f"  Pay Growth: {best['pay_growth']:.3%} ({best['pay_diff_bp']:.0f}bp off)")
        print(f"  Total Error: {best['total_diff_bp']:.0f}bp")


if __name__ == "__main__":
    main()
