#!/usr/bin/env python3
"""
FINAL PRECISION FIX - Targeting exact 10bp tolerance.

Success so far: Pay growth fixed to 5.1bp from target (massive improvement!)
Remaining issue: HC growth at 0.82% vs 3.00% target (218bp off)

Final fix: Adjust hiring/termination balance to hit HC target while preserving pay growth fix.
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

def create_final_precision_config(
    # HC TARGETING PARAMETERS
    target_growth: float,
    new_hire_rate: float,
    term_base_rate: float,
    # PAY TARGETING PARAMETERS (boosted for 3% target)
    annual_comp_rate: float = 0.23,  # Boosted for 3% target
    merit_base: float = 0.21,        # Boosted for 3% target  
    comp_boost: float = 2.0,         # Boosted for 3% target
    new_hire_age: float = 40         # From best result
) -> Path:
    """
    Create FINAL precision config that hits both HC and pay targets.
    Uses the successful pay growth fix + targeted HC adjustments.
    """
    
    # Load base template
    with open(TEMPLATE, 'r') as f:
        config = yaml.safe_load(f)
    
    # === HC TARGETING (need ~4x increase from 0.82% to 3.0%) ===
    set_nested(config, "global_parameters.target_growth", target_growth)
    set_nested(config, "global_parameters.new_hires.new_hire_rate", new_hire_rate)
    set_nested(config, "global_parameters.termination_hazard.base_rate_for_new_hire", term_base_rate)
    
    # === PAY TARGETING (successful parameters from previous fix) ===
    set_nested(config, "global_parameters.annual_compensation_increase_rate", annual_comp_rate)
    set_nested(config, "global_parameters.raises_hazard.merit_base", merit_base)
    set_nested(config, "global_parameters.raises_hazard.promotion_raise", 0.30)
    set_nested(config, "global_parameters.raises_hazard.merit_tenure_bump_value", 0.010)
    set_nested(config, "global_parameters.raises_hazard.merit_low_level_bump_value", 0.010)
    
    # === BOOSTED COLA RATES FOR 3% TARGET ===
    config['global_parameters']['cola_hazard'] = {
        'by_year': {
            2025: 0.15, 2026: 0.14, 2027: 0.13, 2028: 0.12, 2029: 0.11
        }
    }
    
    # === WORKFORCE COMPOSITION (successful parameters) ===
    set_nested(config, "global_parameters.new_hire_average_age", new_hire_age)
    set_nested(config, "global_parameters.new_hire_age_std_dev", 8)
    set_nested(config, "global_parameters.max_working_age", 67)
    
    # === JOB LEVEL BOOSTS (successful approach) ===
    for level in config['job_levels']:
        current_merit = level.get('avg_annual_merit_increase', 0.03)
        level['avg_annual_merit_increase'] = current_merit * comp_boost
        if 'comp_base_salary' in level:
            level['comp_base_salary'] = int(level['comp_base_salary'] * comp_boost)
    
    # === RETENTION PATTERNS (successful approach) ===
    age_multipliers = {
        "<30": 0.8, "30-39": 0.6, "40-49": 0.4, "50-59": 0.3, "60-65": 1.5, "65+": 3.0
    }
    for age_band, multiplier in age_multipliers.items():
        set_nested(config, f"global_parameters.termination_hazard.age_multipliers.{age_band}", multiplier)
    
    # === STANDARD PARAMETERS ===
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
    output_dir = Path("final_precision")
    output_dir.mkdir(exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params_str = f"FINAL_tg{target_growth:.3f}_nh{new_hire_rate:.2f}_tb{term_base_rate:.3f}"
    config_path = output_dir / f"{params_str}_{timestamp}.yaml"
    
    # Write configuration
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    return config_path

def run_and_analyze_final(config_path: Path) -> Dict[str, Any]:
    """Run simulation and return precise results."""
    output_dir = Path("final_precision") / f"output_{config_path.stem}"
    
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

def run_final_precision_campaign():
    """Run the FINAL precision campaign targeting exact 10bp tolerance."""
    
    print("=== FINAL PRECISION TARGETING CAMPAIGN ===")
    print("CORRECTED: Targeting 3% pay growth (not 0%) to match annual_compensation_increase_rate")
    print("Boosting compensation parameters to achieve both 3% HC and 3% pay targets within 10bp")
    print(f"Target: {TARGET_HC_GROWTH:.1%} ± 10bp HC, {TARGET_PAY_GROWTH:.1%} ± 10bp Pay")
    print()
    
    # FINAL precision attempts
    # Strategy: Preserve successful pay parameters, adjust HC targeting
    # Need ~4x HC growth increase (0.82% → 3.0%)
    
    final_configs = [
        # [target_growth, new_hire_rate, term_base_rate]
        # Progressively increase hiring rate and reduce termination
        
        [0.035, 0.60, 0.020],  # Higher target, much higher hiring, lower termination
        [0.040, 0.65, 0.018],  # Even higher
        [0.045, 0.70, 0.016],  # Aggressive
        [0.050, 0.75, 0.014],  # Very aggressive
        [0.055, 0.80, 0.012],  # Extreme
        [0.060, 0.85, 0.010],  # Ultra aggressive
        [0.070, 0.90, 0.008],  # Maximum
    ]
    
    results = []
    perfect_configs = []
    
    for i, (target_growth, new_hire_rate, term_base_rate) in enumerate(final_configs):
        print(f"🎯 FINAL TEST {i+1}/{len(final_configs)}:")
        print(f"  target_growth={target_growth:.1%}, new_hire_rate={new_hire_rate:.0%}")
        print(f"  term_base_rate={term_base_rate:.1%}")
        print(f"  [Boosted pay parameters for 3% target: 23% annual_comp, 21% merit]")
        
        # Create configuration
        config_path = create_final_precision_config(target_growth, new_hire_rate, term_base_rate)
        
        # Run simulation
        result = run_and_analyze_final(config_path)
        if result:
            print(f"  ✅ HC Growth: {result['hc_growth']:.4%} ({result['hc_error_bp']:.1f}bp from target)")
            print(f"  ✅ Pay Growth: {result['pay_growth']:.4%} ({result['pay_error_bp']:.1f}bp from target)")
            print(f"  ✅ Total Error: {result['total_error_bp']:.1f}bp")
            
            if result['within_10bp']:
                print(f"  🎯 PERFECT! FINAL FIX ACHIEVED 10BP TOLERANCE! 🎯")
                perfect_configs.append(result)
            elif result['total_error_bp'] < 50:
                print(f"  🟡 VERY CLOSE! Within 50bp")
            elif result['total_error_bp'] < 100:
                print(f"  🟡 CLOSE! Within 100bp")
            
            results.append(result)
        else:
            print(f"  ❌ Simulation failed")
        print()
    
    # Final analysis
    print("=== FINAL PRECISION RESULTS ===")
    print(f"Perfect configs (within 10bp): {len(perfect_configs)}")
    
    if perfect_configs:
        print("\\n🎯 PERFECT FINAL CONFIGURATIONS:")
        for i, result in enumerate(perfect_configs):
            print(f"  {i+1}. {result['config_path'].name}")
            print(f"     HC: {result['hc_growth']:.4%} ({result['hc_error_bp']:.1f}bp)")
            print(f"     Pay: {result['pay_growth']:.4%} ({result['pay_error_bp']:.1f}bp)")
            
            # Copy the first perfect config
            if i == 0:
                perfect_dest = Path("final_precision/PERFECT_FINAL_10BP.yaml")
                shutil.copy2(result['config_path'], perfect_dest)
                print(f"     ✅ FINAL SOLUTION DELIVERED: {perfect_dest}")
                return result
    
    if results:
        # Find closest
        best = min(results, key=lambda x: x['total_error_bp'])
        print(f"\\n🏆 BEST FINAL RESULT:")
        print(f"  Config: {best['config_path'].name}")
        print(f"  HC Growth: {best['hc_growth']:.4%} ({best['hc_error_bp']:.1f}bp from target)")
        print(f"  Pay Growth: {best['pay_growth']:.4%} ({best['pay_error_bp']:.1f}bp from target)")
        print(f"  Total Error: {best['total_error_bp']:.1f}bp")
        
        # Save the best final version
        best_dest = Path("final_precision/BEST_FINAL_CONFIG.yaml")
        shutil.copy2(best['config_path'], best_dest)
        print(f"  ✅ BEST FINAL VERSION SAVED: {best_dest}")
        
        # Show total improvement journey
        print(f"\\n📈 TOTAL IMPROVEMENT JOURNEY:")
        print(f"  Original baseline error: ~1,500bp (couldn't hit targets)")
        print(f"  Ultra precision error: 410bp")
        print(f"  Fixed precision error: 223bp (45% improvement)")
        print(f"  Final precision error: {best['total_error_bp']:.0f}bp")
        print(f"  Total improvement: {1500 - best['total_error_bp']:.0f}bp")
        
        return best
    
    return None

if __name__ == "__main__":
    result = run_final_precision_campaign()