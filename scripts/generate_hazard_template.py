# scripts/generate_hazard_template.py
import pandas as pd
import itertools
from pathlib import Path

def generate_hazard_table_template(output_dir: Path = Path("."), filename: str = "hazard_table.csv"):
    """
    Generates a template CSV file for a hazard table with reasonable starting points
    for termination rates, promotion rates, and compensation raise components,
    sensitive to job level and tenure band.

    The table includes the new tenure bands: <1, 1-3, 3-5, 5-10, 10-15, 15+.
    Job levels are numeric: 1, 2, 3, 4, 5.
    """

    # --- Configuration ---
    simulation_years = [2024, 2025, 2026]
    job_levels = [1, 2, 3, 4, 5]  # Updated to numeric job levels
    tenure_bands = ["<1", "1-3", "3-5", "5-10", "10-15", "15+"]
    
    # Base rates (these are illustrative and should be calibrated)
    base_rates = {
        "term_rate": 0.08,
        "promotion_rate": 0.10,
        "merit_raise_pct": 0.03,
        "cola_pct": 0.02,  # Can be made year-dependent if needed
        "promotion_raise_pct": 0.12, # Additional raise upon promotion
        "cfg": "baseline"
    }

    # --- Create all combinations ---
    all_combinations = list(itertools.product(simulation_years, job_levels, tenure_bands))
    df = pd.DataFrame(all_combinations, columns=["simulation_year", "employee_level", "tenure_band"])

    # --- Initialize columns with base rates ---
    for col, base_value in base_rates.items():
        df[col] = base_value

    # --- Apply adjustments based on level and tenure band ---
    # These adjustments are examples; refine them based on your specific knowledge.

    # Termination Rate Adjustments
    # Higher for new hires, decreases, then might slightly increase for very long tenure.
    # Generally lower for higher levels once established.
    for index, row in df.iterrows():
        level_num = row["employee_level"] # Updated: employee_level is now numeric

        # Tenure-based term_rate adjustments
        if row["tenure_band"] == "<1":
            df.loc[index, "term_rate"] = 0.25  # High new hire attrition
        elif row["tenure_band"] == "1-3":
            df.loc[index, "term_rate"] = 0.15
        elif row["tenure_band"] == "3-5":
            df.loc[index, "term_rate"] = 0.10
        elif row["tenure_band"] == "5-10":
            df.loc[index, "term_rate"] = 0.07
        elif row["tenure_band"] == "10-15":
            df.loc[index, "term_rate"] = 0.05
        elif row["tenure_band"] == "15+":
            df.loc[index, "term_rate"] = 0.06 # Slight uptick for late career/retirement

        # Level-based term_rate adjustments (example: higher levels are slightly more stable)
        # level_num is 1-based. (level_num - 1) makes it 0-based for multiplier.
        df.loc[index, "term_rate"] = max(0.01, df.loc[index, "term_rate"] * (1 - (level_num - 1) * 0.1))


    # Promotion Rate Adjustments
    # Higher for early/core tenure at lower/mid-levels.
    # Zero for the highest level or very late tenure.
    for index, row in df.iterrows():
        level_num = row["employee_level"] # Updated: employee_level is now numeric

        if level_num == max(job_levels): # Highest level cannot be promoted
            df.loc[index, "promotion_rate"] = 0.00
            df.loc[index, "promotion_raise_pct"] = 0.00 # No promotion raise if no promotion
            continue

        promo_rate_multiplier = 1.0
        if row["tenure_band"] == "<1":
            promo_rate_multiplier = 0.5 # Lower promo chance in first year
        elif row["tenure_band"] == "1-3":
            promo_rate_multiplier = 1.5
        elif row["tenure_band"] == "3-5":
            promo_rate_multiplier = 2.0 # Peak promo window
        elif row["tenure_band"] == "5-10":
            promo_rate_multiplier = 1.0
        elif row["tenure_band"] == "10-15":
            promo_rate_multiplier = 0.3
        elif row["tenure_band"] == "15+":
            promo_rate_multiplier = 0.1 # Very low promo chance for veterans

        # Higher levels might have lower base promotion rates (harder to climb higher)
        # level_num is 1-based. (level_num - 1) makes it 0-based for multiplier.
        level_promo_dampener = (1 - (level_num - 1) * 0.15)
        
        df.loc[index, "promotion_rate"] = max(0.0, min(1.0, base_rates["promotion_rate"] * promo_rate_multiplier * level_promo_dampener))
        
        # If promotion rate is effectively zero, promotion raise should also be zero
        if df.loc[index, "promotion_rate"] < 0.001:
             df.loc[index, "promotion_raise_pct"] = 0.00


    # Merit Raise % Adjustments
    # Example: slightly higher for earlier tenure bands or lower levels
    for index, row in df.iterrows():
        level_num = row["employee_level"] # Updated: employee_level is now numeric
        merit_adj = 0.00

        if row["tenure_band"] in ["<1", "1-3", "3-5"]:
            merit_adj += 0.005
        if level_num <= 2: # For levels 1 and 2
            merit_adj += 0.005
        
        df.loc[index, "merit_raise_pct"] = base_rates["merit_raise_pct"] + merit_adj
        df.loc[index, "merit_raise_pct"] = max(0.0, df.loc[index, "merit_raise_pct"])


    # COLA % Adjustments (Example: making it vary by year)
    # for index, row in df.iterrows():
    #     if row["simulation_year"] == 2024:
    #         df.loc[index, "cola_pct"] = 0.025
    #     elif row["simulation_year"] == 2025:
    #         df.loc[index, "cola_pct"] = 0.020
    #     elif row["simulation_year"] == 2026:
    #         df.loc[index, "cola_pct"] = 0.015
    # For now, keeping COLA constant as per base_rates.

    # Ensure rates are within sensible bounds (0 to 1 for probabilities)
    for rate_col in ["term_rate", "promotion_rate"]:
        df[rate_col] = df[rate_col].clip(0, 1)
    for pct_col in ["merit_raise_pct", "cola_pct", "promotion_raise_pct"]:
         df[pct_col] = df[pct_col].clip(0) # Percentages can be > 1 if something is very wrong, but not < 0

    # Round numeric columns for display
    numeric_cols = ["term_rate", "promotion_rate", "merit_raise_pct", "cola_pct", "promotion_raise_pct"]
    df[numeric_cols] = df[numeric_cols].round(4)

    # --- Save to CSV ---
    output_path = output_dir / filename
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated hazard table template at: {output_path}")
    print(f"Generated {len(df)} rows.")

if __name__ == "__main__":
    # Example: Save to a 'data' subdirectory relative to the script
    # script_dir = Path(__file__).parent
    # output_directory = script_dir.parent / "data" 
    
    # For simplicity, save in the current working directory or a 'data' subfolder
    output_directory = Path("cost_model/state") 
    generate_hazard_table_template(output_dir=output_directory)
