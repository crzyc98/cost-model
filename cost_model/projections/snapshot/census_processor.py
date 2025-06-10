# cost_model/projections/snapshot/census_processor.py
"""
Census data processing module for snapshot creation.
Handles file loading, format detection, and basic data validation.
"""
import logging
from pathlib import Path
from typing import Dict, Union

import pandas as pd

from cost_model.state.schema import (
    EMP_BIRTH_DATE,
    EMP_DEFERRAL_RATE,
    EMP_GROSS_COMP,
    EMP_HIRE_DATE,
    EMP_ID,
    EMP_TERM_DATE,
)

logger = logging.getLogger(__name__)


class CensusProcessor:
    """Handles loading and initial processing of census data files."""

    def __init__(self):
        self.column_mapping = {
            # Standard mappings from schema.py
            "ssn": EMP_ID,
            "employee_ssn": EMP_ID,
            "birth_date": EMP_BIRTH_DATE,
            "hire_date": EMP_HIRE_DATE,
            "termination_date": EMP_TERM_DATE,
            "gross_compensation": EMP_GROSS_COMP,
            # Additional mappings specific to our CSV structure
            "employee_birth_date": EMP_BIRTH_DATE,
            "employee_hire_date": EMP_HIRE_DATE,
            "employee_termination_date": EMP_TERM_DATE,
            "employee_gross_compensation": EMP_GROSS_COMP,
            "employee_deferral_rate": EMP_DEFERRAL_RATE,
        }

        self.required_columns = [EMP_ID, EMP_HIRE_DATE, EMP_BIRTH_DATE, EMP_GROSS_COMP]

    def load_census_file(self, census_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load census data from file, supporting both Parquet and CSV formats.

        Args:
            census_path: Path to census data file

        Returns:
            Raw census DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file cannot be read
        """
        census_path = Path(census_path) if isinstance(census_path, str) else census_path

        if not census_path.exists():
            raise FileNotFoundError(f"Census file not found: {census_path}")

        logger.debug("Reading census data from %s", census_path)

        # Try Parquet first, fall back to CSV
        try:
            census_df = pd.read_parquet(census_path)
            logger.info("Successfully loaded census data from Parquet file")
            return census_df
        except Exception as parquet_error:
            try:
                logger.info("Parquet read failed, attempting to read as CSV")
                file_extension = census_path.suffix.lower()

                if file_extension == ".csv":
                    census_df = pd.read_csv(census_path)
                    logger.info("Successfully loaded census data from CSV file")
                    return census_df
                else:
                    logger.error("Census file is neither a valid Parquet nor a CSV file.")
                    raise parquet_error
            except Exception as csv_error:
                logger.error(
                    "Failed to read census file as either Parquet or CSV: %s",
                    str(csv_error),
                    exc_info=True,
                )
                raise csv_error

    def standardize_column_names(self, census_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply column name standardization to census data.

        Args:
            census_df: Raw census DataFrame

        Returns:
            DataFrame with standardized column names
        """
        logger.info("Standardizing column names in census data")

        # Apply column mapping
        census_df = census_df.rename(columns=self.column_mapping)

        logger.info(f"After column mapping, available columns: {census_df.columns.tolist()}")

        return census_df

    def validate_required_columns(self, census_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all required columns exist in census data.

        Args:
            census_df: Census DataFrame with standardized columns

        Returns:
            Validated DataFrame

        Raises:
            ValueError: If required columns are missing
        """
        missing_columns = [col for col in self.required_columns if col not in census_df.columns]

        # Handle special case where employee_id might be in employee_ssn
        if EMP_ID in missing_columns and "employee_ssn" in census_df.columns:
            logger.info(f"Creating {EMP_ID} from employee_ssn column")
            census_df[EMP_ID] = census_df["employee_ssn"]
            missing_columns.remove(EMP_ID)

        if missing_columns:
            error_msg = f"Census data is missing required columns: {', '.join(missing_columns)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        return census_df

    def filter_active_employees(self, census_df: pd.DataFrame, start_year: int) -> pd.DataFrame:
        """
        Filter out employees terminated before the projection start year.

        Args:
            census_df: Census DataFrame
            start_year: Simulation start year

        Returns:
            Filtered DataFrame with active employees only
        """
        term_col = EMP_TERM_DATE if EMP_TERM_DATE in census_df.columns else None

        if term_col:
            # Convert termination dates to datetime
            census_df[term_col] = pd.to_datetime(census_df[term_col], errors="coerce")

            # Only include employees with no termination date or termination after start year
            before_filter = len(census_df)
            census_df = census_df[
                census_df[term_col].isna()
                | (census_df[term_col] > pd.Timestamp(f"{start_year}-01-01"))
            ]
            after_filter = len(census_df)

            logger.info(
                "Filtered out %d employees terminated before or at %d-01-01. Remaining: %d",
                before_filter - after_filter,
                start_year,
                after_filter,
            )
        else:
            logger.info(
                "No '%s' column found in census. Assuming all employees are active.", EMP_TERM_DATE
            )

        return census_df

    def process_census_data(self, census_path: Union[str, Path], start_year: int) -> pd.DataFrame:
        """
        Complete census processing pipeline.

        Args:
            census_path: Path to census data file
            start_year: Simulation start year

        Returns:
            Processed and validated census DataFrame
        """
        logger.info(
            "Processing census data from %s for start year %d", str(census_path), start_year
        )

        # Load file
        census_df = self.load_census_file(census_path)

        if census_df.empty:
            logger.warning("Census data is empty")
            return census_df

        logger.info(
            "Loaded census data with %d records. Columns: %s",
            len(census_df),
            census_df.columns.tolist(),
        )

        # Process data
        census_df = self.standardize_column_names(census_df)
        census_df = self.validate_required_columns(census_df)
        census_df = self.filter_active_employees(census_df, start_year)

        return census_df
