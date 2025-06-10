"""
Census data loading and preprocessing for snapshot creation.

Handles the loading, validation, and initial preprocessing of census data
from various file formats (Parquet, CSV).
"""

from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd

from .constants import CENSUS_COLUMN_MAPPINGS, DEFAULT_COMPENSATION
from .exceptions import CensusDataError
from .logging_utils import (
    get_snapshot_logger,
    log_dataframe_info,
    progress_context,
    timing_decorator,
)
from .models import SnapshotConfig, ValidationResult
from .types import (
    CensusLoaderConfig,
    ColumnName,
    CompensationAmount,
    FilePath,
    SimulationYear,
)
from .types import ValidationResult as ValidationResultType
from .validators import SnapshotValidator, validate_and_log_results

logger = get_snapshot_logger(__name__)


class CensusLoader:
    """Handles loading and preprocessing of census data.

    This class manages the loading, validation, and initial preprocessing of census
    data from various file formats, ensuring data quality and consistency.
    """

    def __init__(self, config: SnapshotConfig) -> None:
        """Initialize the census loader.

        Args:
            config: Snapshot configuration containing loading parameters.
        """
        self.config = config
        self.validator = SnapshotValidator(config)

    @timing_decorator(logger)
    def load_census_file(self, census_path: FilePath) -> pd.DataFrame:
        """
        Load census data from file with validation and preprocessing.

        Args:
            census_path: Path to the census data file

        Returns:
            DataFrame containing loaded and validated census data

        Raises:
            CensusDataError: If file cannot be loaded or is invalid
        """
        census_path = Path(census_path) if isinstance(census_path, str) else census_path

        # Validate file exists and format
        file_validation = self.validator.validate_census_file(str(census_path))
        validate_and_log_results(file_validation, "census file")

        logger.info(
            "Loading census data",
            file_path=str(census_path),
            file_size_mb=f"{census_path.stat().st_size / 1024 / 1024:.2f}",
        )

        try:
            # Load based on file extension
            if census_path.suffix.lower() == ".parquet":
                df = pd.read_parquet(census_path)
            elif census_path.suffix.lower() == ".csv":
                df = pd.read_csv(census_path)
            else:
                raise CensusDataError(f"Unsupported file format: {census_path.suffix}")

            # Log detailed DataFrame information
            log_dataframe_info(logger, df, "Raw census data")

            # Standardize column names before validation
            df = self.standardize_column_names(df)

            # Validate loaded data
            data_validation = self.validator.validate_census_data(df)
            validate_and_log_results(data_validation, "census data")

            # Log final DataFrame info
            log_dataframe_info(logger, df, "Processed census data", detailed=True)

            return df

        except CensusDataError:
            raise
        except Exception as e:
            raise CensusDataError(f"Failed to load census file {census_path}: {str(e)}")

    @timing_decorator(logger)
    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names using predefined mappings.

        Args:
            df: DataFrame with potentially non-standard column names

        Returns:
            DataFrame with standardized column names
        """
        logger.debug("Standardizing column names")

        # Create mapping for columns that exist in the dataframe
        rename_mapping = {}
        for source_col, target_col in CENSUS_COLUMN_MAPPINGS.items():
            if source_col.lower() in [col.lower() for col in df.columns]:
                # Find the actual column name (case-insensitive)
                actual_col = next(col for col in df.columns if col.lower() == source_col.lower())
                rename_mapping[actual_col] = target_col

        if rename_mapping:
            logger.debug(
                "Standardizing column names",
                mapping_count=len(rename_mapping),
                mappings=rename_mapping,
            )
            df = df.rename(columns=rename_mapping)
        else:
            logger.debug("No column name standardization needed")

        return df

    def validate_required_columns(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate that all required columns are present.

        Args:
            df: DataFrame to validate

        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])

        missing_columns = set(self.config.required_columns) - set(df.columns)
        if missing_columns:
            result.add_error(f"Missing required columns: {missing_columns}")

        return result

    @timing_decorator(logger)
    def filter_terminated_employees(self, df: pd.DataFrame, start_year: int) -> pd.DataFrame:
        """
        Filter out employees who were terminated before the start year.

        Args:
            df: DataFrame with employee data
            start_year: Starting year for simulation

        Returns:
            DataFrame with only relevant employees
        """
        # Use standardized column names
        EMP_TERM_DATE = "EMP_TERM_DATE"
        EMP_ACTIVE = "EMP_ACTIVE"

        initial_count = len(df)

        # Filter based on termination date if available
        if EMP_TERM_DATE in df.columns:
            # Convert termination dates to datetime if they're strings
            if df[EMP_TERM_DATE].dtype == "object":
                df[EMP_TERM_DATE] = pd.to_datetime(df[EMP_TERM_DATE], errors="coerce")

            # Filter out employees terminated before start year
            start_date = pd.Timestamp(f"{start_year}-01-01")
            terminated_before_start = df[EMP_TERM_DATE].notna() & (df[EMP_TERM_DATE] < start_date)
            df = df[~terminated_before_start].copy()

            filtered_count = initial_count - len(df)
            if filtered_count > 0:
                logger.info(
                    "Filtered terminated employees",
                    filtered_count=filtered_count,
                    start_year=start_year,
                    remaining_count=len(df),
                )

        # Also filter based on active status if available
        if EMP_ACTIVE in df.columns:
            active_initial = len(df)
            if df[EMP_ACTIVE].dtype == "object":
                # Convert string representations to boolean
                df[EMP_ACTIVE] = (
                    df[EMP_ACTIVE].astype(str).str.lower().isin(["true", "1", "yes", "active"])
                )

            # Keep only active employees or those with null active status
            df = df[df[EMP_ACTIVE].fillna(True)].copy()

            active_filtered = active_initial - len(df)
            if active_filtered > 0:
                logger.info(
                    "Filtered inactive employees",
                    filtered_count=active_filtered,
                    remaining_count=len(df),
                )

        logger.info(
            "Employee filtering complete",
            final_count=len(df),
            total_filtered=initial_count - len(df),
        )
        return df

    @timing_decorator(logger)
    def initialize_employee_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Initialize basic employee data structure with defaults.

        Args:
            df: DataFrame with census data

        Returns:
            DataFrame with initialized employee data
        """
        # Use standardized column names directly
        EMP_ID = "EMP_ID"
        EMP_HIRE_DATE = "EMP_HIRE_DATE"
        EMP_BIRTH_DATE = "EMP_BIRTH_DATE"
        EMP_GROSS_COMP = "EMP_GROSS_COMP"
        EMP_DEFERRAL_RATE = "EMP_DEFERRAL_RATE"
        EMP_ACTIVE = "EMP_ACTIVE"
        EMP_LEVEL = "EMP_LEVEL"
        EMP_EXITED = "EMP_EXITED"

        logger.debug("Initializing employee data structure")
        logger.debug(f"Available columns: {list(df.columns)}")

        # Ensure required columns exist
        if EMP_ID not in df.columns:
            logger.error(f"EMP_ID not found in columns: {list(df.columns)}")
            raise CensusDataError("Employee ID column (EMP_ID) is required")

        if EMP_HIRE_DATE not in df.columns:
            raise CensusDataError("Hire date column (EMP_HIRE_DATE) is required")

        # Initialize missing columns with defaults
        if EMP_GROSS_COMP not in df.columns:
            logger.warning(
                f"No compensation column found, using default: ${DEFAULT_COMPENSATION:,.0f}"
            )
            df[EMP_GROSS_COMP] = DEFAULT_COMPENSATION

        if EMP_DEFERRAL_RATE not in df.columns:
            logger.debug("No deferral rate column found, defaulting to 0.0")
            df[EMP_DEFERRAL_RATE] = 0.0

        if EMP_ACTIVE not in df.columns:
            logger.debug("No active status column found, defaulting to True")
            df[EMP_ACTIVE] = True

        if EMP_EXITED not in df.columns:
            logger.debug("No exited status column found, defaulting to False")
            df[EMP_EXITED] = False

        # Convert data types
        df = self._convert_data_types(df)

        return df

    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to appropriate data types."""
        # Use standardized column names
        EMP_HIRE_DATE = "EMP_HIRE_DATE"
        EMP_BIRTH_DATE = "EMP_BIRTH_DATE"
        EMP_TERM_DATE = "EMP_TERM_DATE"
        EMP_GROSS_COMP = "EMP_GROSS_COMP"
        EMP_DEFERRAL_RATE = "EMP_DEFERRAL_RATE"

        # Convert date columns
        date_columns = [EMP_HIRE_DATE, EMP_BIRTH_DATE, EMP_TERM_DATE]
        for col in date_columns:
            if col in df.columns and df[col].dtype == "object":
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Convert numeric columns
        numeric_columns = [EMP_GROSS_COMP, EMP_DEFERRAL_RATE]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df
