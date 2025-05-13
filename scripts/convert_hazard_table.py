"""
Convert hazard table from CSV to Parquet format

This script reads the hazard table CSV from cost_model/state/ and converts it to Parquet format
in the data directory for use in simulations. It also adds required columns that might be missing.
"""
import pandas as pd
from pathlib import Path

def main():
    # Define paths
    input_csv = Path("cost_model/state/hazard_table.csv")
    output_parquet = Path("data/hazard_table.parquet")
    
    # Create data directory if it doesn't exist
    output_parquet.parent.mkdir(exist_ok=True)
    
    # Read CSV
    print(f"Reading hazard table from {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Rename columns to match expected names
    column_mapping = {
        'year': 'simulation_year',
        'role': 'role',
        'tenure_band': 'tenure_band',
        'term_rate': 'term_rate',
        'comp_raise_pct': 'comp_raise_pct',
        'cola_pct': 'cola_pct'
    }
    df = df.rename(columns=column_mapping)
    
    # Add missing columns with default values
    df['new_hire_termination_rate'] = 0.25  # Default value from config
    df['cfg'] = None  # Will be filled by the simulation engine
    
    # Ensure all required columns are present
    required_columns = {
        'simulation_year',
        'role',
        'tenure_band',
        'term_rate',
        'comp_raise_pct',
        'new_hire_termination_rate',
        'cola_pct',
        'cfg'
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns after processing: {missing_columns}")
    
    # Write to Parquet
    print(f"Writing to {output_parquet}")
    df.to_parquet(output_parquet, index=False)
    
    print("Conversion complete!")
    print("Final columns:", df.columns.tolist())

if __name__ == "__main__":
    main()
