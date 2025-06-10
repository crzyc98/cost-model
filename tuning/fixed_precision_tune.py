#!/usr/bin/env python3
"""
FIXED Precision Tuning - Addressing the pay growth calculation issue.

The problem: Workforce composition changes (new hires vs departures) dominate
compensation parameters by a factor of 20:1, making precision targeting impossible.

The solution: Balance workforce dynamics with compensation growth to achieve
true precision targeting within 10 basis points.
"""

import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

# Project paths
project_root = Path(__file__).parent.parent
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


def create_fixed_precision_config(
    target_growth: float,
    new_hire_rate: float,
    # COMPENSATION PARAMETERS - Much higher to counteract workforce effects
    annual_comp_rate: float,
    merit_base: float,
    cola_rates: List[float],
    promotion_raise: float,
    # WORKFORCE BALANCE PARAMETERS
    term_base_rate: float,
    new_hire_age: float,
    new_hire_comp_boost: float = 1.2,  # New parameter to balance workforce composition
) -> Path:
    """
    Create a FIXED precision configuration that properly balances:
    1. Workforce composition effects (-15% structural impact)
    2. Compensation policy effects (+15% to counteract)
    """

    # Load base template
    with open(TEMPLATE, "r") as f:
        config = yaml.safe_load(f)

    # === CORE PARAMETERS ===
    set_nested(config, "global_parameters.target_growth", target_growth)
    set_nested(config, "global_parameters.new_hires.new_hire_rate", new_hire_rate)
    set_nested(
        config, "global_parameters.termination_hazard.base_rate_for_new_hire", term_base_rate
    )

    # === BOOSTED COMPENSATION PARAMETERS ===
    # These need to be much higher to counteract the -15% workforce composition effect
    set_nested(config, "global_parameters.annual_compensation_increase_rate", annual_comp_rate)
    set_nested(config, "global_parameters.raises_hazard.merit_base", merit_base)
    set_nested(config, "global_parameters.raises_hazard.promotion_raise", promotion_raise)

    # Boost other merit components
    set_nested(
        config, "global_parameters.raises_hazard.merit_tenure_bump_value", 0.010
    )  # Higher bump
    set_nested(
        config, "global_parameters.raises_hazard.merit_low_level_bump_value", 0.010
    )  # Higher bump

    # === BOOSTED COLA RATES ===
    # Clean COLA setup with higher rates to counteract workforce effects
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

    # === WORKFORCE COMPOSITION FIXES ===
    # Hire older, more experienced workers to reduce the composition effect
    set_nested(config, "global_parameters.new_hire_average_age", new_hire_age)  # Older hires
    set_nested(config, "global_parameters.new_hire_age_std_dev", 8)  # More age variety
    set_nested(config, "global_parameters.max_working_age", 67)  # Keep senior employees longer

    # === JOB LEVEL COMPENSATION BOOSTS ===
    # Increase the job level merit increases to help counteract workforce composition
    for level in config["job_levels"]:
        current_merit = level.get("avg_annual_merit_increase", 0.03)
        boosted_merit = current_merit * new_hire_comp_boost  # Boost by 20%
        level["avg_annual_merit_increase"] = boosted_merit

        # Also boost base salaries for new hires
        if "comp_base_salary" in level:
            level["comp_base_salary"] = int(level["comp_base_salary"] * new_hire_comp_boost)

    # === RETENTION IMPROVEMENTS ===
    # Reduce termination of senior employees to preserve higher-paid workforce
    senior_protection_multipliers = {
        "<30": 0.8,  # Standard for young workers
        "30-39": 0.6,  # Protect prime workforce
        "40-49": 0.4,  # Strong protection for experienced
        "50-59": 0.3,  # Very strong protection for senior
        "60-65": 1.5,  # Allow some retirement but reduce pressure
        "65+": 3.0,  # Reduced from extreme retirement pressure
    }

    for age_band, multiplier in senior_protection_multipliers.items():
        set_nested(
            config, f"global_parameters.termination_hazard.age_multipliers.{age_band}", multiplier
        )

    # Standard parameters for consistency
    set_nested(config, "global_parameters.promotion_hazard.base_rate", 0.10)
    set_nested(config, "global_parameters.promotion_hazard.level_dampener_factor", 0.15)
    set_nested(config, "global_parameters.termination_hazard.level_discount_factor", 0.10)
    set_nested(config, "global_parameters.termination_hazard.min_level_discount_multiplier", 0.5)

    # Standard multipliers
    tenure_mult = {"<1": 0.25, "1-3": 0.7, "3-5": 0.4, "5-10": 0.20, "10-15": 0.12, "15+": 0.24}
    promo_mult = {"<1": 0.5, "1-3": 1.5, "3-5": 2.0, "5-10": 1.0, "10-15": 0.3, "15+": 0.1}
    promo_age_mult = {"<30": 1.4, "30-39": 1.1, "40-49": 0.9, "50-59": 0.4, "60-65": 0.1}

    for key, val in tenure_mult.items():
        set_nested(config, f"global_parameters.termination_hazard.tenure_multipliers.{key}", val)
    for key, val in promo_mult.items():
        set_nested(config, f"global_parameters.promotion_hazard.tenure_multipliers.{key}", val)
    for key, val in promo_age_mult.items():
        set_nested(config, f"global_parameters.promotion_hazard.age_multipliers.{key}", val)

    # Create output directory
    output_dir = Path("fixed_precision")
    output_dir.mkdir(exist_ok=True)

    # Generate descriptive filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params_str = f"FIXED_tg{target_growth:.3f}_ac{annual_comp_rate:.3f}_mb{merit_base:.3f}_boost{new_hire_comp_boost:.1f}"
    config_path = output_dir / f"{params_str}_{timestamp}.yaml"

    # Write configuration
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    return config_path


def run_and_analyze_fixed(config_path: Path) -> Dict[str, Any]:
    """Run simulation and return precise results with detailed analysis."""
    output_dir = Path("fixed_precision") / f"output_{config_path.stem}"

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
            print(f"❌ Simulation failed: {result.stderr}")
            return None

        # Parse results with detailed analysis
        metrics_file = output_dir / "Baseline/Baseline_metrics.csv"
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            if len(df) > 1:
                initial_hc = int(df["active_headcount"].iloc[0])
                final_hc = int(df["active_headcount"].iloc[-1])
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
                    "initial_hc": initial_hc,
                    "final_hc": final_hc,
                    "initial_comp": initial_comp,
                    "final_comp": final_comp,
                }
        return None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def run_fixed_precision_campaign():
    """Run the FIXED precision campaign with corrected compensation dynamics."""

    print("=== FIXED PRECISION TARGETING CAMPAIGN ===")
    print("Addressing the workforce composition vs compensation parameter imbalance")
    print(f"Target: {TARGET_HC_GROWTH:.1%} ± 10bp HC, {TARGET_PAY_GROWTH:.1%} ± 10bp Pay")
    print()
    print("🔧 FIXES IMPLEMENTED:")
    print("  • Boosted compensation parameters to counteract -15% workforce effect")
    print("  • Improved workforce composition (older hires, senior retention)")
    print("  • Enhanced job level merit increases and base salaries")
    print("  • Balanced termination patterns to preserve high-paid employees")
    print()

    # FIXED configuration attempts
    # Strategy: Much higher compensation parameters to overcome structural workforce deflation

    fixed_configs = [
        # [target_growth, new_hire_rate, annual_comp_rate, merit_base, [cola_rates], promotion_raise, term_base_rate, new_hire_age, comp_boost]
        # Attempt 1: Moderate boost
        [0.030, 0.51, 0.18, 0.15, [0.10, 0.09, 0.08, 0.07, 0.06], 0.25, 0.039, 35, 1.5],
        # Attempt 2: High boost
        [0.030, 0.51, 0.20, 0.18, [0.12, 0.11, 0.10, 0.09, 0.08], 0.30, 0.039, 40, 1.8],
        # Attempt 3: Very high boost
        [0.030, 0.51, 0.25, 0.20, [0.15, 0.14, 0.13, 0.12, 0.11], 0.35, 0.039, 45, 2.0],
        # Attempt 4: Extreme boost
        [0.030, 0.51, 0.30, 0.25, [0.18, 0.17, 0.16, 0.15, 0.14], 0.40, 0.039, 50, 2.5],
        # Attempt 5: Ultra boost (to overcome -15% structural effect)
        [0.030, 0.51, 0.35, 0.30, [0.20, 0.19, 0.18, 0.17, 0.16], 0.45, 0.039, 55, 3.0],
    ]

    results = []
    perfect_configs = []

    for i, params in enumerate(fixed_configs):
        (
            target_growth,
            new_hire_rate,
            annual_comp_rate,
            merit_base,
            cola_rates,
            promotion_raise,
            term_base_rate,
            new_hire_age,
            comp_boost,
        ) = params

        print(f"🧪 TESTING FIXED CONFIG {i+1}/{len(fixed_configs)}:")
        print(f"  annual_comp_rate={annual_comp_rate:.1%}, merit_base={merit_base:.1%}")
        print(f"  COLA_2025={cola_rates[0]:.1%}, promotion_raise={promotion_raise:.1%}")
        print(f"  new_hire_age={new_hire_age}, comp_boost={comp_boost:.1f}x")

        # Create configuration
        config_path = create_fixed_precision_config(
            target_growth,
            new_hire_rate,
            annual_comp_rate,
            merit_base,
            cola_rates,
            promotion_raise,
            term_base_rate,
            new_hire_age,
            comp_boost,
        )

        # Run simulation
        result = run_and_analyze_fixed(config_path)
        if result:
            print(
                f"  ✅ HC Growth: {result['hc_growth']:.4%} ({result['hc_error_bp']:.1f}bp from target)"
            )
            print(
                f"  ✅ Pay Growth: {result['pay_growth']:.4%} ({result['pay_error_bp']:.1f}bp from target)"
            )
            print(f"  ✅ Total Error: {result['total_error_bp']:.1f}bp")

            if result["within_10bp"]:
                print(f"  🎯 PERFECT! FIXED AND WITHIN 10BP TOLERANCE! 🎯")
                perfect_configs.append(result)
            elif result["total_error_bp"] < 100:
                print(f"  🟡 CLOSE! Within 100bp tolerance")

            results.append(result)
        else:
            print(f"  ❌ Simulation failed")
        print()

    # Final analysis
    print("=== FIXED PRECISION RESULTS ===")
    print(f"Perfect configs (within 10bp): {len(perfect_configs)}")

    if perfect_configs:
        print("\\n🎯 PERFECT FIXED CONFIGURATIONS:")
        for i, result in enumerate(perfect_configs):
            print(f"  {i+1}. {result['config_path'].name}")
            print(f"     HC: {result['hc_growth']:.4%} ({result['hc_error_bp']:.1f}bp)")
            print(f"     Pay: {result['pay_growth']:.4%} ({result['pay_error_bp']:.1f}bp)")

            # Copy the first perfect config
            if i == 0:
                perfect_dest = Path("fixed_precision/PERFECT_FIXED_10BP.yaml")
                shutil.copy2(result["config_path"], perfect_dest)
                print(f"     ✅ FIXED AND DELIVERED: {perfect_dest}")

    if results:
        # Find closest even if not perfect
        best = min(results, key=lambda x: x["total_error_bp"])
        print(f"\\n🏆 BEST FIXED RESULT:")
        print(f"  Config: {best['config_path'].name}")
        print(f"  HC Growth: {best['hc_growth']:.4%} ({best['hc_error_bp']:.1f}bp from target)")
        print(f"  Pay Growth: {best['pay_growth']:.4%} ({best['pay_error_bp']:.1f}bp from target)")
        print(f"  Total Error: {best['total_error_bp']:.1f}bp")

        # Save the best fixed version
        best_dest = Path("fixed_precision/BEST_FIXED_CONFIG.yaml")
        shutil.copy2(best["config_path"], best_dest)
        print(f"  ✅ FIXED VERSION SAVED: {best_dest}")

        # Show improvement
        original_error = 410  # From previous ultra-precision campaign
        improvement = original_error - best["total_error_bp"]
        print(f"\\n📈 IMPROVEMENT:")
        print(f"  Original error: {original_error:.0f}bp")
        print(f"  Fixed error: {best['total_error_bp']:.0f}bp")
        print(f"  Improvement: {improvement:.0f}bp ({improvement/original_error:.1%} better)")


if __name__ == "__main__":
    run_fixed_precision_campaign()
