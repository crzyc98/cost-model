# /scripts/convert_hazard_table.py
import pandas as pd
from pathlib import Path

def main():
    input_csv = Path("data/generated_hazard_table_yaml_template.csv")
    output_parquet = Path("data/hazard_table.parquet")

    output_parquet.parent.mkdir(exist_ok=True)

    print(f"Reading hazard table from {input_csv}")
    if not input_csv.exists():
        print(f"ERROR: Input CSV file not found at {input_csv}")
        return
    df = pd.read_csv(input_csv)
    print(f"Successfully read {len(df)} rows from {input_csv}")

    # Define column mapping based on the new CSV structure
    # The new CSV has: simulation_year, employee_level, tenure_band, cfg, term_rate,
    # promotion_rate, cola_pct, merit_raise_pct, promotion_raise_pct
    column_mapping = {
        'simulation_year': 'simulation_year',
        'employee_level': 'employee_level',
        'tenure_band': 'tenure_band',
        'cfg': 'cfg',
        'term_rate': 'term_rate',
        'promotion_rate': 'promotion_rate',
        'cola_pct': 'cola_pct',
        'merit_raise_pct': 'merit_raise_pct',
        'promotion_raise_pct': 'promotion_raise_pct'
    }
    # Only rename columns that exist in the CSV
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # Add other missing globally applicable columns with default values
    if 'new_hire_termination_rate' not in df.columns:
        df['new_hire_termination_rate'] = 0.25
        print("Added 'new_hire_termination_rate' column with default 0.25")

    # Note: 'cfg' should already be present in the new CSV template

    # Define the exact columns expected in the output Parquet file
    # This Parquet is the INPUT to cost_model.projections.hazard.py
    # Updated to include granular raise components instead of comp_raise_pct
    expected_parquet_columns = [
        'simulation_year',
        'employee_level',
        'tenure_band',
        'term_rate',
        'promotion_rate',
        'merit_raise_pct',
        'promotion_raise_pct',
        'cola_pct',
        'new_hire_termination_rate',
        'cfg'
    ]
    # If you decided to keep 'role' in your CSV and Parquet, add it here too:
    # if 'role' in df.columns:
    #     if 'role' not in expected_parquet_columns: # To avoid duplicates if already there
    #        expected_parquet_columns.insert(1, 'role') # Insert after simulation_year for example

    # Ensure all expected columns are present
    for col in expected_parquet_columns:
        if col not in df.columns:
            # If 'employee_level' is missing here, it means it wasn't in your CSV or mapping
            raise ValueError(f"FATAL: Column '{col}' is missing from DataFrame after CSV load and rename. Columns present: {df.columns.tolist()}. Please check your CSV and column_mapping.")

    # Select and reorder columns
    df = df[expected_parquet_columns]

    print(f"Writing to {output_parquet}. Columns: {df.columns.tolist()}")
    df.to_parquet(output_parquet, index=False)

    print("Conversion complete!")

if __name__ == "__main__":
    main()