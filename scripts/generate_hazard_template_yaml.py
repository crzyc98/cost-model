# scripts/generate_hazard_template_yaml.py
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: Path) -> dict:
    """Loads the YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"Successfully loaded configuration from {config_path}")
    return config


def generate_hazard_table_from_config(config: dict, output_dir: Path, filename: str):
    """
    Generates a hazard table CSV based on parameters from a loaded configuration dictionary.
    Uses vectorized operations for efficiency.
    """
    cfg_global = config["global"]
    cfg_cola = config["cola"]
    cfg_term = config["termination"]
    cfg_promo = config["promotion"]
    cfg_raises = config["raises"]

    simulation_years = cfg_global["simulation_years"]
    job_levels = cfg_global["job_levels"]  # Numeric e.g. [1, 2, 3, 4, 5]
    tenure_bands = cfg_global["tenure_bands"]

    # Create all combinations for the DataFrame
    all_combinations = list(itertools.product(simulation_years, job_levels, tenure_bands))
    df = pd.DataFrame(
        all_combinations, columns=["simulation_year", "employee_level", "tenure_band"]
    )

    # --- Apply COLA ---
    df["cola_pct"] = (
        df["simulation_year"].map(cfg_cola["by_year"]).fillna(0.02)
    )  # Default if year not in map

    # --- Calculate Termination Rate (Vectorized) ---
    # 1. Apply tenure multiplier to base new hire rate
    tenure_term_multiplier_map = cfg_term["tenure_multipliers"]
    df["term_rate"] = (
        df["tenure_band"].map(tenure_term_multiplier_map) * cfg_term["base_rate_for_new_hire"]
    )

    # 2. Apply level discount
    # level_num is 1-based. (level_num - 1) makes it 0-based for multiplier.
    level_discount_effect = np.maximum(
        cfg_term["min_level_discount_multiplier"],
        1 - cfg_term["level_discount_factor"] * (df["employee_level"] - 1),
    )
    df["term_rate"] = df["term_rate"] * level_discount_effect

    # --- Calculate Promotion Rate (Vectorized) ---
    df["promotion_rate"] = cfg_promo["base_rate"]

    # 1. Apply tenure multiplier
    tenure_promo_multiplier_map = cfg_promo["tenure_multipliers"]
    df["promotion_rate"] = df["promotion_rate"] * df["tenure_band"].map(tenure_promo_multiplier_map)

    # 2. Apply level dampener
    # level_num is 1-based. (level_num - 1) makes it 0-based for multiplier.
    level_dampener_effect = np.maximum(
        0.0,  # Promotion rate can be dampened to 0
        1 - cfg_promo["level_dampener_factor"] * (df["employee_level"] - 1),
    )
    df["promotion_rate"] = df["promotion_rate"] * level_dampener_effect

    # 3. Set promo rate to 0 for the highest level
    df.loc[df["employee_level"] == max(job_levels), "promotion_rate"] = 0.0

    # --- Calculate Merit Raise Percentage (Vectorized) ---
    df["merit_raise_pct"] = cfg_raises["merit_base"]

    # 1. Add tenure bump
    tenure_bump_condition = df["tenure_band"].isin(cfg_raises["merit_tenure_bump_bands"])
    df.loc[tenure_bump_condition, "merit_raise_pct"] += cfg_raises["merit_tenure_bump_value"]

    # 2. Add low-level bump
    level_bump_condition = df["employee_level"] <= cfg_raises["merit_low_level_cutoff"]
    df.loc[level_bump_condition, "merit_raise_pct"] += cfg_raises["merit_low_level_bump_value"]

    # --- Calculate Promotion Raise Percentage (Vectorized) ---
    df["promotion_raise_pct"] = cfg_raises["promotion_raise"]
    # Zero out promotion raise if promotion rate is (near) zero
    df.loc[df["promotion_rate"] < 0.001, "promotion_raise_pct"] = 0.0

    # --- Add CFG column ---
    df["cfg"] = cfg_global["cfg_scenario_name"]

    # Ensure rates are within sensible bounds and round
    for rate_col in ["term_rate", "promotion_rate"]:
        df[rate_col] = df[rate_col].clip(0, 1).round(4)
    for pct_col in ["cola_pct", "merit_raise_pct", "promotion_raise_pct"]:
        df[pct_col] = df[pct_col].clip(0).round(4)  # Min 0, no upper bound for raises generally

    # --- Reorder columns to a logical sequence ---
    final_columns = [
        "simulation_year",
        "employee_level",
        "tenure_band",
        "cfg",
        "term_rate",
        "promotion_rate",
        "cola_pct",
        "merit_raise_pct",
        "promotion_raise_pct",
    ]
    df = df[final_columns]

    # --- Save to CSV ---
    output_path = output_dir / filename
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated YAML-driven hazard table template at: {output_path}")
    print(f"Generated {len(df)} rows.")


if __name__ == "__main__":
    # Define paths
    # Assuming the script is in a 'scripts' folder, and config is in 'config' folder at project root
    project_root = Path(__file__).parent.parent
    config_file_path = project_root / "config" / "hazard_defaults.yaml"

    # Output directory for the generated CSV
    output_csv_dir = project_root / "data"
    output_csv_filename = "generated_hazard_table_yaml_template.csv"

    try:
        config_data = load_config(config_file_path)
        generate_hazard_table_from_config(config_data, output_csv_dir, output_csv_filename)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure 'config/hazard_defaults.yaml' exists relative to the project root.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
