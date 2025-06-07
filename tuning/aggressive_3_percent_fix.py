#!/usr/bin/env python3
"""
AGGRESSIVE 3% PAY GROWTH TARGETING

Since the corrected 3% target campaign still shows -0.28% pay growth with 23% annual_comp_rate,
we need much more aggressive compensation parameters to overcome the structural workforce deflation.

Strategy: Progressively boost compensation parameters until we hit 3% pay growth.
"""

import yaml
import subprocess
import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Project paths
project_root = Path(__file__).parent.parent
TEMPLATE = project_root / "config/dev_tiny.yaml"
RUNNER = project_root / "scripts/run_simulation.py"
DEFAULT_CENSUS_PATH = project_root / "data/census_template.parquet"

# PRECISION TARGETS
TARGET_HC_GROWTH = 0.030      # 3.00%
TARGET_PAY_GROWTH = 0.030     # 3.00%
TOLERANCE = 0.001             # 10 basis points

def set_nested(config: Dict[str, Any], key: str, value: Any) -> None:
    """Set a nested configuration value using dot notation."""
    keys = key.split('.')
    current = config
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value

def create_aggressive_3_percent_config(
    config_name: str,
    # AGGRESSIVE COMPENSATION PARAMETERS FOR 3% TARGET
    annual_comp_rate: float,
    merit_base: float,
    cola_2025: float,
    promotion_raise: float,
    comp_boost: float,
    # HC TARGETING PARAMETERS  
    target_growth: float = 0.035,
    new_hire_rate: float = 0.60,
    term_base_rate: float = 0.020,
    new_hire_age: float = 40
) -> Path:
    """Create aggressive configuration targeting 3% pay growth."""
    
    # Load base template
    with open(TEMPLATE, 'r') as f:
        config = yaml.safe_load(f)
    
    # === AGGRESSIVE COMPENSATION PARAMETERS ===
    set_nested(config, "global_parameters.annual_compensation_increase_rate", annual_comp_rate)
    set_nested(config, "global_parameters.raises_hazard.merit_base", merit_base)
    set_nested(config, "global_parameters.raises_hazard.promotion_raise", promotion_raise)
    set_nested(config, "global_parameters.raises_hazard.merit_tenure_bump_value", 0.015)  # Higher
    set_nested(config, "global_parameters.raises_hazard.merit_low_level_bump_value", 0.015)  # Higher
    
    # === AGGRESSIVE COLA RATES ===
    config['global_parameters']['cola_hazard'] = {
        'by_year': {
            2025: cola_2025,
            2026: cola_2025 - 0.01, 
            2027: cola_2025 - 0.02,
            2028: cola_2025 - 0.03,
            2029: cola_2025 - 0.04
        }
    }
    
    # === HC TARGETING ===
    set_nested(config, "global_parameters.target_growth", target_growth)
    set_nested(config, "global_parameters.new_hires.new_hire_rate", new_hire_rate)
    set_nested(config, "global_parameters.termination_hazard.base_rate_for_new_hire", term_base_rate)
    
    # === WORKFORCE COMPOSITION ===
    set_nested(config, "global_parameters.new_hire_average_age", new_hire_age)
    set_nested(config, "global_parameters.new_hire_age_std_dev", 8)
    set_nested(config, "global_parameters.max_working_age", 67)
    
    # === AGGRESSIVE JOB LEVEL BOOSTS ===
    for level in config['job_levels']:
        current_merit = level.get('avg_annual_merit_increase', 0.03)
        level['avg_annual_merit_increase'] = current_merit * comp_boost
        if 'comp_base_salary' in level:
            level['comp_base_salary'] = int(level['comp_base_salary'] * comp_boost)
    
    # === RETENTION PATTERNS ===
    age_multipliers = {
        "<30": 0.8, "30-39": 0.6, "40-49": 0.4, "50-59": 0.3, "60-65": 1.5, "65+": 3.0
    }
    for age_band, multiplier in age_multipliers.items():
        set_nested(config, f"global_parameters.termination_hazard.age_multipliers.{age_band}", multiplier)
    
    # Standard parameters
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
    output_dir = Path("aggressive_3_percent")
    output_dir.mkdir(exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = output_dir / f"{config_name}_{timestamp}.yaml"
    
    # Write configuration
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    return config_path

def run_and_analyze_aggressive(config_path: Path) -> Dict[str, Any]:
    """Run simulation and return precise results."""
    output_dir = Path("aggressive_3_percent") / f"output_{config_path.stem}"
    
    cmd = [
        "python3", str(RUNNER),
        "--config", str(config_path),
        "--scenario", "baseline",
        "--census", str(DEFAULT_CENSUS_PATH),
        "--output", str(output_dir)
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
                initial_hc = int(df['active_headcount'].iloc[0])
                final_hc = int(df['active_headcount'].iloc[-1])
                hc_growth = (final_hc - initial_hc) / initial_hc
                
                initial_comp = df['avg_compensation'].iloc[0]
                final_comp = df['avg_compensation'].iloc[-1]
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
                    "config_path": config_path
                }
        return None
        
    except Exception as e:
        return None

def run_aggressive_3_percent_campaign():
    """Run aggressive campaign targeting 3% pay growth."""
    
    print("=== AGGRESSIVE 3% PAY GROWTH TARGETING ===")
    print("Progressively boosting compensation parameters to hit 3% pay growth target")
    print(f"Target: {TARGET_HC_GROWTH:.1%} ± 10bp HC, {TARGET_PAY_GROWTH:.1%} ± 10bp Pay")
    print()
    
    # Progressive compensation boosts
    # [config_name, annual_comp_rate, merit_base, cola_2025, promotion_raise, comp_boost]
    aggressive_configs = [
        ["MODERATE", 0.30, 0.25, 0.20, 0.40, 2.5],   # Moderate boost
        ["HIGH", 0.35, 0.30, 0.25, 0.45, 3.0],       # High boost  
        ["VERY_HIGH", 0.40, 0.35, 0.30, 0.50, 3.5],  # Very high boost
        ["EXTREME", 0.45, 0.40, 0.35, 0.55, 4.0],    # Extreme boost
        ["ULTRA", 0.50, 0.45, 0.40, 0.60, 4.5],      # Ultra boost
    ]
    
    results = []
    perfect_configs = []
    
    for config_name, annual_comp_rate, merit_base, cola_2025, promotion_raise, comp_boost in aggressive_configs:
        print(f"🚀 TESTING {config_name} BOOST:")
        print(f"  annual_comp_rate={annual_comp_rate:.0%}, merit_base={merit_base:.0%}")
        print(f"  COLA_2025={cola_2025:.0%}, promotion_raise={promotion_raise:.0%}")
        print(f"  comp_boost={comp_boost:.1f}x")
        
        # Create configuration
        config_path = create_aggressive_3_percent_config(
            config_name, annual_comp_rate, merit_base, cola_2025, promotion_raise, comp_boost
        )
        
        # Run simulation
        result = run_and_analyze_aggressive(config_path)
        if result:
            print(f"  ✅ HC Growth: {result['hc_growth']:.4%} ({result['hc_error_bp']:.1f}bp from target)")
            print(f"  ✅ Pay Growth: {result['pay_growth']:.4%} ({result['pay_error_bp']:.1f}bp from target)")
            print(f"  ✅ Total Error: {result['total_error_bp']:.1f}bp")
            
            if result['within_10bp']:
                print(f"  🎯 PERFECT! ACHIEVED 3% TARGET WITHIN 10BP! 🎯")
                perfect_configs.append(result)
            elif result['pay_error_bp'] < 50:
                print(f"  🟡 VERY CLOSE! Pay growth within 50bp of 3% target")
            elif result['pay_error_bp'] < 100:
                print(f"  🟡 CLOSE! Pay growth within 100bp of 3% target")
            
            results.append(result)
        else:
            print(f"  ❌ Simulation failed")
        print()
    
    # Final analysis
    print("=== AGGRESSIVE 3% RESULTS ===")
    print(f"Perfect configs (within 10bp): {len(perfect_configs)}")
    
    if perfect_configs:
        print("\\n🎯 PERFECT 3% CONFIGURATIONS:")
        for i, result in enumerate(perfect_configs):
            print(f"  {i+1}. {result['config_path'].name}")
            print(f"     HC: {result['hc_growth']:.4%} ({result['hc_error_bp']:.1f}bp)")
            print(f"     Pay: {result['pay_growth']:.4%} ({result['pay_error_bp']:.1f}bp)")
            
            # Copy the first perfect config
            if i == 0:
                perfect_dest = Path("aggressive_3_percent/PERFECT_3_PERCENT.yaml")
                shutil.copy2(result['config_path'], perfect_dest)
                print(f"     ✅ 3% TARGET ACHIEVED: {perfect_dest}")
                return result
    
    if results:
        # Find closest to 3% pay growth
        best = min(results, key=lambda x: x['pay_error_bp'])
        print(f"\\n🏆 CLOSEST TO 3% PAY GROWTH:")
        print(f"  Config: {best['config_path'].name}")
        print(f"  HC Growth: {best['hc_growth']:.4%} ({best['hc_error_bp']:.1f}bp from target)")
        print(f"  Pay Growth: {best['pay_growth']:.4%} ({best['pay_error_bp']:.1f}bp from target)")
        print(f"  Total Error: {best['total_error_bp']:.1f}bp")
        
        # Save the best version
        best_dest = Path("aggressive_3_percent/CLOSEST_TO_3_PERCENT.yaml")
        shutil.copy2(best['config_path'], best_dest)
        print(f"  ✅ CLOSEST TO 3% SAVED: {best_dest}")
        
        return best
    
    return None

if __name__ == "__main__":
    result = run_aggressive_3_percent_campaign()