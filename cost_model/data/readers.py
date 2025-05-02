# cost_model/data/readers.py
"""
Functions for reading input data files (census, reference data).
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional, List

# Attempt to import column constants, provide fallbacks
try:
    from ..utils.columns import (
        EMP_HIRE_DATE, EMP_TERM_DATE, EMP_BIRTH_DATE, EMP_SSN
    )
except ImportError:
    print("Warning (readers.py): Could not import column constants from utils. Using string literals.")
    EMP_HIRE_DATE, EMP_TERM_DATE, EMP_BIRTH_DATE = 'employee_hire_date', 'employee_termination_date', 'employee_birth_date'
    EMP_SSN = 'employee_ssn'


logger = logging.getLogger(__name__)

# Define a custom exception for data reading errors
class DataReadError(Exception):
    """Custom exception for errors during data reading."""
    pass

def read_census_data(file_path: Path) -> Optional[pd.DataFrame]:
    """
    Reads census data from a CSV or Parquet file.

    Handles basic date parsing and standardizes the employee identifier column
    to 'employee_id'.

    Args:
        file_path: Path object pointing to the census file.

    Returns:
        A pandas DataFrame containing the loaded census data, or None if loading fails.

    Raises:
        DataReadError: If the file cannot be found, read, or processed.
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path) # Ensure it's a Path object

    logger.info(f"Attempting to read census data from: {file_path}")

    if not file_path.exists():
        logger.error(f"Census file not found: {file_path}")
        raise DataReadError(f"Census file not found: {file_path}")

    try:
        # Determine file type and read accordingly
        file_suffix = file_path.suffix.lower()
        df: Optional[pd.DataFrame] = None

        # Columns to attempt parsing as dates
        date_cols_to_parse = [EMP_HIRE_DATE, EMP_TERM_DATE, EMP_BIRTH_DATE]

        if file_suffix == '.parquet':
            df = pd.read_parquet(file_path)
            logger.info(f"Loaded {len(df)} records from Parquet census: {file_path}")
            # Ensure date columns are datetime (Parquet can preserve types, but good to check)
            for col in date_cols_to_parse:
                if col in df.columns:
                    if not pd.api.types.is_datetime64_any_dtype(df[col]):
                         logger.debug(f"Converting column '{col}' to datetime (read as {df[col].dtype}).")
                         df[col] = pd.to_datetime(df[col], errors='coerce')
                         if df[col].isnull().any():
                             logger.warning(f"Column '{col}' contained values that could not be parsed as dates during Parquet load.")

        elif file_suffix == '.csv':
            # Check which date cols actually exist before trying to parse
            try:
                # Read just the header first to check columns
                cols_in_csv = pd.read_csv(file_path, nrows=0).columns.tolist()
                parse_dates_present = [c for c in date_cols_to_parse if c in cols_in_csv]
                logger.debug(f"Attempting to parse date columns in CSV: {parse_dates_present}")
                df = pd.read_csv(file_path, parse_dates=parse_dates_present)
                logger.info(f"Loaded {len(df)} records from CSV census: {file_path}")
                # Check for parse errors after loading
                for col in parse_dates_present:
                     if pd.api.types.is_object_dtype(df[col]):
                         logger.warning(f"Column '{col}' in CSV might not have been fully parsed as dates (still object type). Check format or consider errors='coerce'.")
                         # Optionally force coercion if needed:
                         # df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception as read_err:
                 logger.error(f"Error reading CSV file {file_path}: {read_err}")
                 raise DataReadError(f"Error reading CSV file {file_path}") from read_err
        else:
            logger.error(f"Unsupported census file format: {file_path}. Please use .csv or .parquet.")
            raise DataReadError(f"Unsupported census file format: {file_path.suffix}")

        if df is None or df.empty:
             logger.warning(f"No data loaded from census file: {file_path}")
             return None # Return None for empty data

        logger.debug(f"Columns loaded: {df.columns.tolist()}")

        # --- Standardize Identifier Column ---
        # Check for common identifier names and rename to EMP_ID ('employee_id')
        possible_id_cols = ['employee_id', EMP_SSN, 'Employee ID', 'ID', 'Emp ID'] # Add other common names
        id_col_found = None
        for col_name in possible_id_cols:
            if col_name in df.columns:
                id_col_found = col_name
                break

        if id_col_found:
            if id_col_found != 'employee_id':
                if 'employee_id' in df.columns:
                    logger.warning(f"Both '{id_col_found}' and 'employee_id' columns exist. Using 'employee_id'.")
                    # Optionally drop the other column: df.drop(columns=[id_col_found], inplace=True)
                else:
                    logger.info(f"Renaming identifier column '{id_col_found}' to 'employee_id'.")
                    df.rename(columns={id_col_found: 'employee_id'}, inplace=True)
            else:
                 logger.debug(f"Identifier column 'employee_id' already present.")
        else:
             # If no recognizable ID column is found
             logger.error(f"Could not find a recognizable employee identifier column ({possible_id_cols}) in {file_path}.")
             raise DataReadError(f"Missing required employee identifier column in {file_path}")

        # --- Final Checks ---
        # Ensure EMP_ID is suitable as an identifier (e.g., check for uniqueness)
        if df['employee_id'].duplicated().any():
            logger.warning(f"Duplicate values found in the 'employee_id' column. Ensure IDs are unique.")

        logger.info(f"Census data read and prepared successfully.")
        return df

    except DataReadError: # Re-raise specific data read errors
        raise
    except Exception as e:
        logger.exception(f"An unexpected error occurred while reading census data from {file_path}")
        raise DataReadError(f"Unexpected error reading data from {file_path}") from e

# Example Usage (for testing this module directly)
if __name__ == '__main__':
    # This block is for demonstration/testing purposes only
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s")

    # Create dummy data for testing
    dummy_csv_path = Path("./_temp_dummy_census.csv")
    dummy_parquet_path = Path("./_temp_dummy_census.parquet")
    dummy_data = pd.DataFrame({
        'employee_ssn': [f'DUMMY_100{i}' for i in range(5)],
        'employee_role': ['Staff'] * 5,
        'employee_birth_date': pd.to_datetime(['1990-01-15', '1985-05-20', '1992-11-30', '1988-07-01', '1995-03-10']),
        'employee_hire_date': pd.to_datetime(['2015-06-01', '2010-08-15', '2018-01-20', '2012-09-01', '2020-04-15']),
        'employee_termination_date': [pd.NaT, pd.NaT, '2023-10-31', pd.NaT, '2024-12-31 10:00:00'], # Mix formats
        'employee_gross_compensation': [50000, 65000, 55000, 75000, 48000]
    })
    dummy_data.to_csv(dummy_csv_path, index=False, date_format='%Y-%m-%d')
    dummy_data.to_parquet(dummy_parquet_path, index=False)

    print("\n--- Testing CSV Load ---")
    try:
        df_csv = read_census_data(dummy_csv_path)
        if df_csv is not None:
            print("CSV Loaded Successfully:")
            print(df_csv.info())
            print(df_csv.head())
        else:
            print("CSV Loading returned None.")
    except DataReadError as e:
        print(f"ERROR reading CSV: {e}")

    print("\n--- Testing Parquet Load ---")
    try:
        df_parquet = read_census_data(dummy_parquet_path)
        if df_parquet is not None:
            print("Parquet Loaded Successfully:")
            print(df_parquet.info())
            print(df_parquet.head())
        else:
            print("Parquet Loading returned None.")
    except DataReadError as e:
        print(f"ERROR reading Parquet: {e}")

    # Clean up dummy files
    dummy_csv_path.unlink(missing_ok=True)
    dummy_parquet_path.unlink(missing_ok=True)
