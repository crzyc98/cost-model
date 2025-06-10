#!/usr/bin/env python3
"""
Demonstration script for exact headcount targeting functionality.

This script shows how to use the manage_headcount_to_exact_target function
with configuration values from YAML files to achieve precise workforce growth.
"""

import os
import sys
from pathlib import Path

import yaml

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cost_model.engines.run_one_year.utils import manage_headcount_to_exact_target


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def demo_exact_headcount_targeting():
    """
    Demonstrate exact headcount targeting using configuration from dev_tiny.yaml.
    """
    print("=== Exact Headcount Targeting Demonstration ===\n")

    # Load configuration
    config_path = project_root / "config" / "dev_tiny.yaml"
    config = load_config(config_path)

    # Extract relevant parameters
    global_params = config["global_parameters"]
    target_growth = global_params["target_growth"]
    new_hire_term_rate = global_params["attrition"]["new_hire_termination_rate"]

    print(f"Configuration loaded from: {config_path}")
    print(f"Target growth rate: {target_growth:.1%}")
    print(f"New hire termination rate: {new_hire_term_rate:.1%}")
    print()

    # Scenario 1: Normal growth with typical attrition
    print("--- Scenario 1: Normal Growth with Typical Attrition ---")
    soy_actives = 1000  # Start with 1000 employees
    markov_exits = 150  # 15% natural attrition

    gross_hires, forced_terms = manage_headcount_to_exact_target(
        soy_actives=soy_actives,
        target_growth_rate=target_growth,
        num_markov_exits_existing=markov_exits,
        new_hire_termination_rate=new_hire_term_rate,
    )

    target_eoy = round(soy_actives * (1 + target_growth))
    survivors = soy_actives - markov_exits

    print(f"Start of Year Actives: {soy_actives:,}")
    print(f"Natural Attrition (Markov exits): {markov_exits:,}")
    print(f"Survivors after attrition: {survivors:,}")
    print(f"Target End of Year: {target_eoy:,}")
    print(f"Growth needed: {target_eoy - survivors:,}")
    print(f"→ Gross hires needed: {gross_hires:,}")
    print(f"→ Forced terminations: {forced_terms:,}")

    # Calculate expected final headcount
    expected_new_hire_survivors = gross_hires * (1 - new_hire_term_rate)
    final_headcount = survivors + expected_new_hire_survivors - forced_terms
    print(f"Expected final headcount: {final_headcount:.0f} (target: {target_eoy})")
    print()

    # Scenario 2: Low attrition year (survivors exceed target)
    print("--- Scenario 2: Low Attrition Year (Downsizing Needed) ---")
    soy_actives = 1000
    markov_exits = 10  # Only 1% attrition this year
    # Use negative growth to create a scenario where survivors exceed target
    downsizing_growth = -0.05  # 5% reduction target (target = 950, survivors = 990)

    gross_hires, forced_terms = manage_headcount_to_exact_target(
        soy_actives=soy_actives,
        target_growth_rate=downsizing_growth,
        num_markov_exits_existing=markov_exits,
        new_hire_termination_rate=new_hire_term_rate,
    )

    target_eoy = round(soy_actives * (1 + downsizing_growth))
    survivors = soy_actives - markov_exits

    print(f"Start of Year Actives: {soy_actives:,}")
    print(f"Natural Attrition (Markov exits): {markov_exits:,}")
    print(f"Survivors after attrition: {survivors:,}")
    print(f"Target End of Year: {target_eoy:,}")
    print(f"Excess survivors: {survivors - target_eoy:,}")
    print(f"→ Gross hires needed: {gross_hires:,}")
    print(f"→ Forced terminations: {forced_terms:,}")

    final_headcount = survivors - forced_terms
    print(f"Expected final headcount: {final_headcount:.0f} (target: {target_eoy})")
    print()

    # Scenario 3: High growth target
    print("--- Scenario 3: High Growth Target (10%) ---")
    high_growth = 0.10  # 10% growth
    soy_actives = 500
    markov_exits = 75  # 15% attrition

    gross_hires, forced_terms = manage_headcount_to_exact_target(
        soy_actives=soy_actives,
        target_growth_rate=high_growth,
        num_markov_exits_existing=markov_exits,
        new_hire_termination_rate=new_hire_term_rate,
    )

    target_eoy = round(soy_actives * (1 + high_growth))
    survivors = soy_actives - markov_exits

    print(f"Start of Year Actives: {soy_actives:,}")
    print(f"Target growth rate: {high_growth:.1%}")
    print(f"Natural Attrition (Markov exits): {markov_exits:,}")
    print(f"Survivors after attrition: {survivors:,}")
    print(f"Target End of Year: {target_eoy:,}")
    print(f"Growth needed: {target_eoy - survivors:,}")
    print(f"→ Gross hires needed: {gross_hires:,}")
    print(f"→ Forced terminations: {forced_terms:,}")

    expected_new_hire_survivors = gross_hires * (1 - new_hire_term_rate)
    final_headcount = survivors + expected_new_hire_survivors - forced_terms
    print(f"Expected final headcount: {final_headcount:.0f} (target: {target_eoy})")
    print()

    print("=== Summary ===")
    print("The manage_headcount_to_exact_target function provides precise control")
    print("over workforce size by calculating either:")
    print("1. Gross hires needed (accounting for new hire attrition)")
    print("2. Forced terminations needed (when survivors exceed target)")
    print("\nThis ensures exact adherence to growth targets specified in configuration.")


if __name__ == "__main__":
    demo_exact_headcount_targeting()
