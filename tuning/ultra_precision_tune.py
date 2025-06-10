#!/usr/bin/env python3
"""
Ultra-precision tuning for exact 10bp targeting.
Using micro-adjustments around known good values.
"""

import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

# Project root
project_root = Path(__file__).parent.parent

# Configuration paths
TEMPLATE = project_root / "config/dev_tiny.yaml"
RUNNER = project_root / "scripts/run_simulation.py"
DEFAULT_CENSUS_PATH = project_root / "data/census_template.parquet"

# PRECISION TARGETS
TARGET_HC_GROWTH = 0.030  # 3.00%
TARGET_PAY_GROWTH = 0.000  # 0.00%
TOLERANCE = 0.001  # 10 basis points


def set_nested(config: Dict[str, Any], key: str, value: Any) -> None:
    """Set a nested configuration value using dot notation."""
    keys = key.split(".")
    current = config
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value


def create_ultra_precision_config(
    target_growth: float,
    new_hire_rate: float,
    annual_comp_rate: float,
    term_base_rate: float,
    merit_base: float,
    cola_rates: list,
) -> Path:
    """Create a specific ultra-precision configuration."""

    # Load base template
    with open(TEMPLATE, "r") as f:
        config = yaml.safe_load(f)

    # Set precise parameters
    set_nested(config, "global_parameters.target_growth", target_growth)
    set_nested(config, "global_parameters.new_hires.new_hire_rate", new_hire_rate)
    set_nested(config, "global_parameters.annual_compensation_increase_rate", annual_comp_rate)
    set_nested(
        config, "global_parameters.termination_hazard.base_rate_for_new_hire", term_base_rate
    )
    set_nested(config, "global_parameters.raises_hazard.merit_base", merit_base)

    # Set conservative other parameters
    set_nested(config, "global_parameters.promotion_hazard.base_rate", 0.100)
    set_nested(config, "global_parameters.promotion_hazard.level_dampener_factor", 0.15)
    set_nested(config, "global_parameters.new_hire_average_age", 26)
    set_nested(config, "global_parameters.new_hire_age_std_dev", 3)
    set_nested(config, "global_parameters.max_working_age", 64)

    # Clean COLA setup
    if "global_parameters" not in config:
        config["global_parameters"] = {}
    if "cola_hazard" not in config["global_parameters"]:
        config["global_parameters"]["cola_hazard"] = {}

    config["global_parameters"]["cola_hazard"]["by_year"] = {
        2025: cola_rates[0],
        2026: cola_rates[1],
        2027: cola_rates[2],
        2028: cola_rates[3],
        2029: cola_rates[4],
    }

    # Standard multipliers
    term_mult = {"<1": 0.25, "1-3": 0.7, "3-5": 0.4, "5-10": 0.20, "10-15": 0.12, "15+": 0.24}
    age_mult = {"<30": 0.3, "30-39": 0.8, "40-49": 1.0, "50-59": 1.5, "60-65": 8.0, "65+": 12.0}
    promo_mult = {"<1": 0.5, "1-3": 1.5, "3-5": 2.0, "5-10": 1.0, "10-15": 0.3, "15+": 0.1}
    promo_age_mult = {"<30": 1.4, "30-39": 1.1, "40-49": 0.9, "50-59": 0.4, "60-65": 0.1}

    for key, val in term_mult.items():
        set_nested(config, f"global_parameters.termination_hazard.tenure_multipliers.{key}", val)
    for key, val in age_mult.items():
        set_nested(config, f"global_parameters.termination_hazard.age_multipliers.{key}", val)
    for key, val in promo_mult.items():
        set_nested(config, f"global_parameters.promotion_hazard.tenure_multipliers.{key}", val)
    for key, val in promo_age_mult.items():
        set_nested(config, f"global_parameters.promotion_hazard.age_multipliers.{key}", val)

    # Other standard parameters
    set_nested(config, "global_parameters.termination_hazard.level_discount_factor", 0.10)
    set_nested(config, "global_parameters.termination_hazard.min_level_discount_multiplier", 0.5)
    set_nested(config, "global_parameters.raises_hazard.merit_tenure_bump_value", 0.003)
    set_nested(config, "global_parameters.raises_hazard.merit_low_level_bump_value", 0.003)
    set_nested(config, "global_parameters.raises_hazard.promotion_raise", 0.11)

    # Create output directory
    output_dir = Path("ultra_precision")
    output_dir.mkdir(exist_ok=True)

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params_str = f"tg{target_growth:.3f}_nh{new_hire_rate:.2f}_ac{annual_comp_rate:.3f}_tb{term_base_rate:.3f}_mb{merit_base:.3f}"
    config_path = output_dir / f"ultra_{params_str}_{timestamp}.yaml"

    # Write configuration
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    return config_path


def run_and_analyze(config_path: Path) -> Dict[str, Any]:
    """Run simulation and return precise results."""
    output_dir = Path("ultra_precision") / f"output_{config_path.stem}"

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

        # Parse results
        metrics_file = output_dir / "Baseline/Baseline_metrics.csv"
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            if len(df) > 1:
                initial_hc = df["active_headcount"].iloc[0]
                final_hc = df["active_headcount"].iloc[-1]
                hc_growth = (final_hc - initial_hc) / initial_hc

                initial_comp = df["avg_compensation"].iloc[0]
                final_comp = df["avg_compensation"].iloc[-1]
                pay_growth = (final_comp - initial_comp) / initial_comp

                hc_error_bp = abs(hc_growth - TARGET_HC_GROWTH) * 10000
                pay_error_bp = abs(pay_growth - TARGET_PAY_GROWTH) * 10000

                return {
                    "hc_growth": hc_growth,
                    "pay_growth": pay_growth,
                    "hc_error_bp": hc_error_bp,
                    "pay_error_bp": pay_error_bp,
                    "total_error_bp": hc_error_bp + pay_error_bp,
                    "within_10bp": hc_error_bp <= 10 and pay_error_bp <= 10,
                    "config_path": config_path,
                }
        return None

    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    """Run ultra-precision targeting."""
    print("=== ULTRA-PRECISION 10BP TARGETING ===")
    print("Systematic micro-adjustment approach")
    print(f"Target: {TARGET_HC_GROWTH:.1%} ± 10bp HC, {TARGET_PAY_GROWTH:.1%} ± 10bp Pay")
    print()

    # Ultra-precise parameter combinations
    # Based on observed patterns: need ~2.9-3.1% target_growth for 3% actual
    # Need higher comp/merit for less negative pay growth

    test_configs = [
        # [target_growth, new_hire_rate, annual_comp_rate, term_base_rate, merit_base, [cola_rates]]
        # Attempt 1: Moderate values
        [0.0295, 0.51, 0.035, 0.039, 0.032, [0.022, 0.020, 0.018, 0.016, 0.014]],
        [0.0298, 0.51, 0.035, 0.039, 0.032, [0.022, 0.020, 0.018, 0.016, 0.014]],
        [0.0302, 0.51, 0.035, 0.039, 0.032, [0.022, 0.020, 0.018, 0.016, 0.014]],
        # Attempt 2: Higher compensation growth
        [0.0300, 0.51, 0.038, 0.039, 0.035, [0.025, 0.023, 0.021, 0.019, 0.017]],
        [0.0295, 0.51, 0.038, 0.039, 0.035, [0.025, 0.023, 0.021, 0.019, 0.017]],
        # Attempt 3: Fine-tuned hiring
        [0.0300, 0.515, 0.036, 0.0385, 0.033, [0.023, 0.021, 0.019, 0.017, 0.015]],
        [0.0300, 0.505, 0.036, 0.0385, 0.033, [0.023, 0.021, 0.019, 0.017, 0.015]],
        # Attempt 4: Balanced approach
        [0.0299, 0.51, 0.037, 0.039, 0.034, [0.024, 0.022, 0.020, 0.018, 0.016]],
        [0.0301, 0.51, 0.037, 0.039, 0.034, [0.024, 0.022, 0.020, 0.018, 0.016]],
        # Attempt 5: Conservative termination
        [0.0300, 0.52, 0.036, 0.037, 0.033, [0.023, 0.021, 0.019, 0.017, 0.015]],
    ]

    results = []
    perfect_configs = []

    for i, params in enumerate(test_configs):
        target_growth, new_hire_rate, annual_comp_rate, term_base_rate, merit_base, cola_rates = (
            params
        )

        print(f"Testing ultra-precision config {i+1}/{len(test_configs)}:")
        print(f"  target_growth={target_growth:.4f}, new_hire_rate={new_hire_rate:.3f}")
        print(f"  annual_comp_rate={annual_comp_rate:.3f}, merit_base={merit_base:.3f}")

        # Create configuration
        config_path = create_ultra_precision_config(
            target_growth, new_hire_rate, annual_comp_rate, term_base_rate, merit_base, cola_rates
        )

        # Run simulation
        result = run_and_analyze(config_path)
        if result:
            print(
                f"  ✅ HC Growth: {result['hc_growth']:.4%} ({result['hc_error_bp']:.1f}bp from target)"
            )
            print(
                f"  ✅ Pay Growth: {result['pay_growth']:.4%} ({result['pay_error_bp']:.1f}bp from target)"
            )
            print(f"  ✅ Total Error: {result['total_error_bp']:.1f}bp")

            if result["within_10bp"]:
                print(f"  🎯 PERFECT! WITHIN 10BP TOLERANCE! 🎯")
                perfect_configs.append(result)

            results.append(result)
        else:
            print(f"  ❌ Simulation failed")
        print()

    # Final analysis
    print("=== ULTRA-PRECISION RESULTS ===")
    print(f"Perfect configs (within 10bp): {len(perfect_configs)}")

    if perfect_configs:
        print("\\n🎯 PERFECT 10BP CONFIGURATIONS:")
        for i, result in enumerate(perfect_configs):
            print(f"  {i+1}. {result['config_path'].name}")
            print(f"     HC: {result['hc_growth']:.4%} ({result['hc_error_bp']:.1f}bp)")
            print(f"     Pay: {result['pay_growth']:.4%} ({result['pay_error_bp']:.1f}bp)")

            # Copy the first perfect config
            if i == 0:
                perfect_dest = Path("ultra_precision/PERFECT_10BP_ULTRA.yaml")
                shutil.copy2(result["config_path"], perfect_dest)
                print(f"     ✅ DELIVERED: {perfect_dest}")

    if results:
        # Find closest even if not perfect
        best = min(results, key=lambda x: x["total_error_bp"])
        print(f"\\n🏆 CLOSEST ULTRA-PRECISION RESULT:")
        print(f"  Config: {best['config_path'].name}")
        print(f"  HC Growth: {best['hc_growth']:.4%} ({best['hc_error_bp']:.1f}bp from target)")
        print(f"  Pay Growth: {best['pay_growth']:.4%} ({best['pay_error_bp']:.1f}bp from target)")
        print(f"  Total Error: {best['total_error_bp']:.1f}bp")

        # Also save the closest
        closest_dest = Path("ultra_precision/CLOSEST_TO_10BP.yaml")
        shutil.copy2(best["config_path"], closest_dest)
        print(f"  ✅ SAVED: {closest_dest}")


if __name__ == "__main__":
    main()
