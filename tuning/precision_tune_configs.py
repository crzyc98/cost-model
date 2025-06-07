#!/usr/bin/env python3
"""
Precision tuning for exact headcount and compensation growth targeting.
Targeting: 3.00% ± 0.10% headcount growth, 0.00% ± 0.10% compensation growth
"""

import yaml
import random
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

# Project root for relative paths
project_root = Path(__file__).parent.parent

# Configuration paths
TEMPLATE = project_root / "config/dev_tiny.yaml"
RUNNER = project_root / "scripts/run_simulation.py"
DEFAULT_CENSUS_PATH = project_root / "data/census_template.parquet"
DEFAULT_SCENARIO = "baseline"

# PRECISION TARGETS (within 10 basis points)
TARGET_HC_GROWTH = 0.030      # 3.00%
TARGET_PAY_GROWTH = 0.000     # 0.00%
TOLERANCE = 0.001             # 10 basis points (0.10%)

HC_GROWTH_MIN = TARGET_HC_GROWTH - TOLERANCE    # 2.90%
HC_GROWTH_MAX = TARGET_HC_GROWTH + TOLERANCE    # 3.10%
PAY_GROWTH_MIN = TARGET_PAY_GROWTH - TOLERANCE  # -0.10%
PAY_GROWTH_MAX = TARGET_PAY_GROWTH + TOLERANCE  # +0.10%

print(f"PRECISION TARGETING:")
print(f"Headcount Growth: {HC_GROWTH_MIN:.1%} - {HC_GROWTH_MAX:.1%}")
print(f"Pay Growth: {PAY_GROWTH_MIN:.1%} - {PAY_GROWTH_MAX:.1%}")
print()

# PRECISION SEARCH SPACE - Very narrow ranges around known good values
# Based on config_101 and refined analysis, tightly constrained for exact targeting
PRECISION_SEARCH_SPACE = {
    # CORE PARAMETERS - Narrow ranges for exact calibration
    "global_parameters.target_growth": [0.028, 0.029, 0.030, 0.031, 0.032],  # Tight around 3%
    "global_parameters.new_hires.new_hire_rate": [0.48, 0.50, 0.52],  # Narrow range around optimal
    "global_parameters.annual_compensation_increase_rate": [0.028, 0.030, 0.032],  # Precise comp control
    
    # TERMINATION CONTROL - Fine-tuned for exact balance
    "global_parameters.termination_hazard.base_rate_for_new_hire": [0.038, 0.040, 0.042],
    
    # PROMOTION RATES - Conservative adjustments
    "global_parameters.promotion_hazard.base_rate": [0.095, 0.100, 0.105],
    "global_parameters.promotion_hazard.level_dampener_factor": [0.14, 0.15, 0.16],
    
    # MERIT PARAMETERS - Precise pay growth control
    "global_parameters.raises_hazard.merit_base": [0.024, 0.026, 0.028],
    "global_parameters.raises_hazard.merit_tenure_bump_value": [0.002, 0.003],
    "global_parameters.raises_hazard.merit_low_level_bump_value": [0.002, 0.003],
    "global_parameters.raises_hazard.promotion_raise": [0.09, 0.10, 0.11],
    
    # AGE PARAMETERS - Maintain optimal workforce
    "global_parameters.new_hire_average_age": [25, 27],
    "global_parameters.new_hire_age_std_dev": [3, 4],
    "global_parameters.max_working_age": [63, 64, 65],
    
    # TERMINATION MULTIPLIERS - Fine adjustments
    "global_parameters.termination_hazard.level_discount_factor": [0.08, 0.10, 0.12],
    "global_parameters.termination_hazard.min_level_discount_multiplier": [0.4, 0.5],
    
    # TENURE MULTIPLIERS - Precise retention control
    "global_parameters.termination_hazard.tenure_multipliers.<1": [0.2, 0.25, 0.3],
    "global_parameters.termination_hazard.tenure_multipliers.1-3": [0.6, 0.7, 0.8],
    "global_parameters.termination_hazard.tenure_multipliers.3-5": [0.35, 0.4, 0.45],
    "global_parameters.termination_hazard.tenure_multipliers.5-10": [0.18, 0.20, 0.22],
    "global_parameters.termination_hazard.tenure_multipliers.10-15": [0.11, 0.12, 0.13],
    "global_parameters.termination_hazard.tenure_multipliers.15+": [0.22, 0.24, 0.26],
    
    # AGE MULTIPLIERS - Maintain retirement patterns
    "global_parameters.termination_hazard.age_multipliers.<30": [0.25, 0.3, 0.35],
    "global_parameters.termination_hazard.age_multipliers.30-39": [0.7, 0.8, 0.9],
    "global_parameters.termination_hazard.age_multipliers.40-49": [0.9, 1.0, 1.1],
    "global_parameters.termination_hazard.age_multipliers.50-59": [1.2, 1.5, 1.8],
    "global_parameters.termination_hazard.age_multipliers.60-65": [6.0, 8.0, 10.0],
    "global_parameters.termination_hazard.age_multipliers.65+": [10.0, 12.0, 15.0],
    
    # PROMOTION MULTIPLIERS - Maintain career progression
    "global_parameters.promotion_hazard.tenure_multipliers.<1": [0.4, 0.5, 0.6],
    "global_parameters.promotion_hazard.tenure_multipliers.1-3": [1.3, 1.5, 1.7],
    "global_parameters.promotion_hazard.tenure_multipliers.3-5": [1.8, 2.0, 2.2],
    "global_parameters.promotion_hazard.tenure_multipliers.5-10": [0.9, 1.0, 1.1],
    "global_parameters.promotion_hazard.tenure_multipliers.10-15": [0.25, 0.3, 0.35],
    "global_parameters.promotion_hazard.tenure_multipliers.15+": [0.08, 0.1, 0.12],
    
    "global_parameters.promotion_hazard.age_multipliers.<30": [1.3, 1.4, 1.5],
    "global_parameters.promotion_hazard.age_multipliers.30-39": [1.0, 1.1, 1.2],
    "global_parameters.promotion_hazard.age_multipliers.40-49": [0.8, 0.9, 1.0],
    "global_parameters.promotion_hazard.age_multipliers.50-59": [0.35, 0.4, 0.45],
    "global_parameters.promotion_hazard.age_multipliers.60-65": [0.08, 0.1, 0.12],
    
    # PRECISION COLA PARAMETERS - Clean numeric keys only
    "_cola_2025": [0.016, 0.018, 0.020],
    "_cola_2026": [0.014, 0.015, 0.016],
    "_cola_2027": [0.010, 0.012, 0.014],
    "_cola_2028": [0.008, 0.010, 0.012],
    "_cola_2029": [0.006, 0.008, 0.010]
}

def set_nested(config: Dict[str, Any], key: str, value: Any) -> None:
    """Set a nested configuration value using dot notation."""
    keys = key.split('.')
    current = config
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value

def set_cola_by_year_clean(config: Dict[str, Any], cola_values: Dict[str, float]) -> None:
    """Completely replace the cola_hazard.by_year section to eliminate duplicate keys."""
    if 'global_parameters' not in config:
        config['global_parameters'] = {}
    if 'cola_hazard' not in config['global_parameters']:
        config['global_parameters']['cola_hazard'] = {}
    
    # Build clean by_year section with only numeric keys
    by_year = {}
    for special_key, rate in cola_values.items():
        if special_key.startswith('_cola_'):
            year = int(special_key[6:])  # Extract year from '_cola_2025' -> 2025
            by_year[year] = rate
    
    # Completely replace the by_year section
    config['global_parameters']['cola_hazard']['by_year'] = by_year

def generate_precision_configs(n: int = 50) -> List[Path]:
    """Generate precision configurations for exact targeting."""
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Template configuration not found: {TEMPLATE}")

    # Create output directory
    output_dir = Path("precision_tuned")
    output_dir.mkdir(exist_ok=True)

    config_paths = []

    for i in range(n):
        # Load base template
        with open(TEMPLATE, 'r') as f:
            config = yaml.safe_load(f)

        # Sample random values from precision search space
        cola_values = {}
        for param_key, possible_values in PRECISION_SEARCH_SPACE.items():
            value = random.choice(possible_values)
            
            # Handle special COLA parameters separately
            if param_key.startswith('_cola_'):
                cola_values[param_key] = value
            else:
                set_nested(config, param_key, value)
        
        # Apply COLA values with special handling to prevent duplicate keys
        if cola_values:
            set_cola_by_year_clean(config, cola_values)

        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_path = output_dir / f"precision_config_{i:03d}_{timestamp}.yaml"

        # Write configuration
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        config_paths.append(config_path)
        if i % 10 == 0 or i == n-1:
            print(f"Generated precision config {i+1}/{n}: {config_path}")

    return config_paths

def precision_score(hc_growth: float, pay_growth: float) -> float:
    """
    Calculate precision score - lower is better.
    Perfect score is 0.0 when both targets are hit exactly.
    """
    hc_error = abs(hc_growth - TARGET_HC_GROWTH)
    pay_error = abs(pay_growth - TARGET_PAY_GROWTH)
    
    # Heavy penalty if outside tolerance
    if hc_growth < HC_GROWTH_MIN or hc_growth > HC_GROWTH_MAX:
        hc_error += 0.01  # 1% penalty
    if pay_growth < PAY_GROWTH_MIN or pay_growth > PAY_GROWTH_MAX:
        pay_error += 0.01  # 1% penalty
    
    # Weighted score - equal importance to both targets
    return 0.5 * hc_error + 0.5 * pay_error

def run_precision_simulation(config_path: Path) -> Dict[str, Any]:
    """Run simulation and extract precise metrics."""
    import subprocess
    
    output_dir = Path("precision_tuned") / f"output_{config_path.stem}"
    
    cmd = [
        "python3", str(RUNNER),
        "--config", str(config_path),
        "--scenario", DEFAULT_SCENARIO,
        "--census", str(DEFAULT_CENSUS_PATH),
        "--output", str(output_dir)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"Simulation failed for {config_path}: {result.stderr}")
            return None
            
        # Parse results
        metrics_file = output_dir / "Baseline/Baseline_metrics.csv"
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            if len(df) > 1:
                initial_hc = df['active_headcount'].iloc[0]
                final_hc = df['active_headcount'].iloc[-1]
                hc_growth = (final_hc - initial_hc) / initial_hc
                
                initial_comp = df['avg_compensation'].iloc[0]
                final_comp = df['avg_compensation'].iloc[-1]
                pay_growth = (final_comp - initial_comp) / initial_comp
                
                return {
                    "hc_growth": hc_growth,
                    "pay_growth": pay_growth,
                    "initial_hc": initial_hc,
                    "final_hc": final_hc,
                    "initial_comp": initial_comp,
                    "final_comp": final_comp
                }
        
        return None
        
    except Exception as e:
        print(f"Error running simulation for {config_path}: {e}")
        return None

def main():
    """Run precision targeting campaign."""
    print("=== PRECISION TARGETING CAMPAIGN ===")
    print(f"Target: {TARGET_HC_GROWTH:.1%} ± {TOLERANCE:.1%} HC growth")
    print(f"Target: {TARGET_PAY_GROWTH:.1%} ± {TOLERANCE:.1%} pay growth")
    print()
    
    # Generate precision configurations
    print("Generating precision configurations...")
    config_paths = generate_precision_configs(50)
    print(f"Generated {len(config_paths)} precision configurations")
    print()
    
    # Run simulations
    results = []
    best_score = float('inf')
    best_config = None
    
    print("Running precision simulations...")
    for i, config_path in enumerate(config_paths):
        print(f"Testing precision config {i+1}/{len(config_paths)}: {config_path.name}")
        
        summary = run_precision_simulation(config_path)
        if summary:
            score = precision_score(summary['hc_growth'], summary['pay_growth'])
            
            # Check if within tolerance
            hc_in_range = HC_GROWTH_MIN <= summary['hc_growth'] <= HC_GROWTH_MAX
            pay_in_range = PAY_GROWTH_MIN <= summary['pay_growth'] <= PAY_GROWTH_MAX
            within_tolerance = hc_in_range and pay_in_range
            
            result = {
                "config_path": str(config_path),
                "precision_score": score,
                "hc_growth": summary['hc_growth'],
                "pay_growth": summary['pay_growth'],
                "within_tolerance": within_tolerance,
                "hc_in_range": hc_in_range,
                "pay_in_range": pay_in_range,
                "summary": summary
            }
            
            results.append(result)
            
            print(f"  HC Growth: {summary['hc_growth']:.3%} {'✅' if hc_in_range else '❌'}")
            print(f"  Pay Growth: {summary['pay_growth']:.3%} {'✅' if pay_in_range else '❌'}")
            print(f"  Precision Score: {score:.6f}")
            print(f"  Within Tolerance: {'✅' if within_tolerance else '❌'}")
            
            if score < best_score:
                best_score = score
                best_config = config_path
                print(f"  🎯 NEW BEST PRECISION SCORE!")
            print()
    
    # Save results
    results_file = Path("precision_tuned/precision_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Analysis
    print("=== PRECISION CAMPAIGN RESULTS ===")
    print(f"Total configurations tested: {len(results)}")
    
    if results:
        # Find configurations within tolerance
        within_tolerance = [r for r in results if r['within_tolerance']]
        print(f"Configurations within tolerance: {len(within_tolerance)}")
        
        if within_tolerance:
            print("\\n🎯 PERFECT CONFIGURATIONS (within 10bp):")
            for result in sorted(within_tolerance, key=lambda x: x['precision_score'])[:5]:
                print(f"  {result['config_path']}")
                print(f"    HC Growth: {result['hc_growth']:.3%}")
                print(f"    Pay Growth: {result['pay_growth']:.3%}")
                print(f"    Score: {result['precision_score']:.6f}")
        
        # Best overall
        best_result = min(results, key=lambda x: x['precision_score'])
        print(f"\\n🏆 BEST PRECISION RESULT:")
        print(f"  Config: {best_result['config_path']}")
        print(f"  HC Growth: {best_result['hc_growth']:.3%} (target: {TARGET_HC_GROWTH:.1%})")
        print(f"  Pay Growth: {best_result['pay_growth']:.3%} (target: {TARGET_PAY_GROWTH:.1%})")
        print(f"  Precision Score: {best_result['precision_score']:.6f}")
        
        # Copy best config
        if best_config:
            best_dest = Path("precision_tuned/best_precision_config.yaml")
            shutil.copy2(best_config, best_dest)
            print(f"\\n✅ Best precision config saved to: {best_dest}")

if __name__ == "__main__":
    main()