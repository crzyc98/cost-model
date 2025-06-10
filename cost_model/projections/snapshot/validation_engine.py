# cost_model/projections/snapshot/validation_engine.py
"""
Validation engine for snapshot creation and updates.
Handles data integrity checks, schema validation, and business rule validation.
"""
import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from cost_model.state.schema import (
    EMP_ACTIVE,
    EMP_AGE,
    EMP_BIRTH_DATE,
    EMP_GROSS_COMP,
    EMP_HIRE_DATE,
    EMP_ID,
    EMP_TENURE,
    SIMULATION_YEAR,
)
from cost_model.state.snapshot.constants import SNAPSHOT_COLS as SNAPSHOT_COL_NAMES
from cost_model.state.snapshot.constants import SNAPSHOT_DTYPES

logger = logging.getLogger(__name__)


class ValidationEngine:
    """Handles validation of snapshot data for integrity and business rules."""

    def __init__(self):
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []

    def reset_validation_state(self):
        """Reset validation error and warning lists."""
        self.validation_errors = []
        self.validation_warnings = []

    def validate_required_columns(self, df: pd.DataFrame, required_cols: List[str]) -> bool:
        """
        Validate that all required columns are present.

        Args:
            df: DataFrame to validate
            required_cols: List of required column names

        Returns:
            True if all required columns present, False otherwise
        """
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            self.validation_errors.append(f"Missing required columns: {missing_cols}")
            return False

        return True

    def validate_data_types(self, df: pd.DataFrame) -> bool:
        """
        Validate that columns have expected data types.

        Args:
            df: DataFrame to validate

        Returns:
            True if data types are valid, False otherwise
        """
        type_errors = []

        for col, expected_dtype in SNAPSHOT_DTYPES.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                # Allow some flexibility in type checking
                if not self._is_compatible_dtype(actual_dtype, str(expected_dtype)):
                    type_errors.append(f"{col}: expected {expected_dtype}, got {actual_dtype}")

        if type_errors:
            self.validation_warnings.extend(type_errors)
            return False

        return True

    def _is_compatible_dtype(self, actual: str, expected: str) -> bool:
        """Check if actual dtype is compatible with expected dtype."""
        # Define compatible type mappings
        compatible_types = {
            "object": ["string", "object"],
            "string": ["object", "string"],
            "datetime64[ns]": ["datetime64", "object"],
            "float64": ["float", "int", "float64", "float32"],
            "int64": ["int", "float", "int64", "int32"],
            "bool": ["bool", "boolean"],
            "Int64": ["int", "Int64", "float64"],  # Nullable integer
        }

        expected_clean = expected.lower()
        actual_clean = actual.lower()

        if expected_clean in compatible_types:
            return any(compat in actual_clean for compat in compatible_types[expected_clean])

        return expected_clean in actual_clean or actual_clean in expected_clean

    def validate_employee_ids(self, df: pd.DataFrame) -> bool:
        """
        Validate employee ID integrity.

        Args:
            df: DataFrame to validate

        Returns:
            True if employee IDs are valid, False otherwise
        """
        if EMP_ID not in df.columns:
            self.validation_errors.append(f"Employee ID column {EMP_ID} not found")
            return False

        # Check for null/empty employee IDs
        null_ids = df[EMP_ID].isna().sum()
        if null_ids > 0:
            self.validation_errors.append(f"Found {null_ids} null employee IDs")

        # Check for duplicate employee IDs
        duplicate_ids = df[EMP_ID].duplicated().sum()
        if duplicate_ids > 0:
            self.validation_errors.append(f"Found {duplicate_ids} duplicate employee IDs")

        # Check for empty string IDs
        if df[EMP_ID].dtype == "object" or "string" in str(df[EMP_ID].dtype):
            empty_ids = (df[EMP_ID].astype(str).str.strip() == "").sum()
            if empty_ids > 0:
                self.validation_errors.append(f"Found {empty_ids} empty employee IDs")

        return len(self.validation_errors) == 0

    def validate_dates(self, df: pd.DataFrame) -> bool:
        """
        Validate date columns for logical consistency.

        Args:
            df: DataFrame to validate

        Returns:
            True if dates are valid, False otherwise
        """
        date_cols = [EMP_HIRE_DATE, EMP_BIRTH_DATE]
        issues = []

        for col in date_cols:
            if col in df.columns:
                # Check for null dates in required columns
                null_dates = df[col].isna().sum()
                if null_dates > 0:
                    if col == EMP_HIRE_DATE:
                        issues.append(f"{col}: {null_dates} null hire dates (required)")
                    elif col == EMP_BIRTH_DATE:
                        issues.append(f"{col}: {null_dates} null birth dates (required)")

                # Check for future dates
                current_year = pd.Timestamp.now().year
                future_dates = (df[col] > pd.Timestamp(f"{current_year + 1}-12-31")).sum()
                if future_dates > 0:
                    issues.append(f"{col}: {future_dates} dates in far future")

                # Check for very old dates (before 1900)
                old_dates = (df[col] < pd.Timestamp("1900-01-01")).sum()
                if old_dates > 0:
                    issues.append(f"{col}: {old_dates} dates before 1900")

        # Check birth date vs hire date consistency
        if EMP_BIRTH_DATE in df.columns and EMP_HIRE_DATE in df.columns:
            valid_rows = df[EMP_BIRTH_DATE].notna() & df[EMP_HIRE_DATE].notna()
            if valid_rows.any():
                hire_before_birth = (
                    df.loc[valid_rows, EMP_HIRE_DATE] < df.loc[valid_rows, EMP_BIRTH_DATE]
                ).sum()
                if hire_before_birth > 0:
                    issues.append(f"Found {hire_before_birth} employees hired before birth")

                # Check for unrealistic hiring age (< 14 or > 80)
                ages_at_hire = (
                    df.loc[valid_rows, EMP_HIRE_DATE] - df.loc[valid_rows, EMP_BIRTH_DATE]
                ).dt.days / 365.25
                young_hires = (ages_at_hire < 14).sum()
                old_hires = (ages_at_hire > 80).sum()

                if young_hires > 0:
                    issues.append(f"Found {young_hires} employees hired before age 14")
                if old_hires > 0:
                    issues.append(f"Found {old_hires} employees hired after age 80")

        if issues:
            self.validation_warnings.extend(issues)
            return False

        return True

    def validate_compensation(self, df: pd.DataFrame) -> bool:
        """
        Validate compensation data for reasonable values.

        Args:
            df: DataFrame to validate

        Returns:
            True if compensation is valid, False otherwise
        """
        if EMP_GROSS_COMP not in df.columns:
            self.validation_errors.append(f"Compensation column {EMP_GROSS_COMP} not found")
            return False

        comp_col = df[EMP_GROSS_COMP]
        issues = []

        # Check for null compensation
        null_comp = comp_col.isna().sum()
        if null_comp > 0:
            issues.append(f"Found {null_comp} employees with null compensation")

        # Check for negative compensation
        negative_comp = (comp_col < 0).sum()
        if negative_comp > 0:
            issues.append(f"Found {negative_comp} employees with negative compensation")

        # Check for very low compensation (< $1000)
        very_low_comp = (comp_col < 1000).sum()
        if very_low_comp > 0:
            self.validation_warnings.append(
                f"Found {very_low_comp} employees with compensation < $1,000"
            )

        # Check for very high compensation (> $10M)
        very_high_comp = (comp_col > 10_000_000).sum()
        if very_high_comp > 0:
            self.validation_warnings.append(
                f"Found {very_high_comp} employees with compensation > $10M"
            )

        if issues:
            self.validation_errors.extend(issues)
            return False

        return True

    def validate_tenure_consistency(self, df: pd.DataFrame, simulation_year: int) -> bool:
        """
        Validate tenure calculations are consistent with hire dates.

        Args:
            df: DataFrame to validate
            simulation_year: Current simulation year

        Returns:
            True if tenure is consistent, False otherwise
        """
        if EMP_TENURE not in df.columns or EMP_HIRE_DATE not in df.columns:
            return True  # Skip if columns not present

        valid_rows = df[EMP_TENURE].notna() & df[EMP_HIRE_DATE].notna()
        if not valid_rows.any():
            return True

        # Calculate expected tenure
        start_date = pd.Timestamp(f"{simulation_year}-01-01")
        expected_tenure = ((start_date - df.loc[valid_rows, EMP_HIRE_DATE]).dt.days / 365.25).clip(
            lower=0
        )
        actual_tenure = df.loc[valid_rows, EMP_TENURE]

        # Allow small differences due to calculation methods
        tenure_diff = abs(expected_tenure - actual_tenure)
        large_differences = (tenure_diff > 0.1).sum()  # More than ~1 month difference

        if large_differences > 0:
            self.validation_warnings.append(
                f"Found {large_differences} employees with tenure/hire date inconsistencies"
            )
            return False

        return True

    def validate_snapshot(
        self, df: pd.DataFrame, simulation_year: int, required_cols: Optional[List[str]] = None
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Run complete validation suite on snapshot data.

        Args:
            df: DataFrame to validate
            simulation_year: Current simulation year
            required_cols: Optional list of required columns (defaults to core columns)

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.reset_validation_state()

        if required_cols is None:
            required_cols = [EMP_ID, EMP_HIRE_DATE, EMP_BIRTH_DATE, EMP_GROSS_COMP]

        logger.info(f"Running validation on snapshot with {len(df)} employees")

        # Run all validations
        validations = [
            self.validate_required_columns(df, required_cols),
            self.validate_data_types(df),
            self.validate_employee_ids(df),
            self.validate_dates(df),
            self.validate_compensation(df),
            self.validate_tenure_consistency(df, simulation_year),
        ]

        is_valid = all(validations) and len(self.validation_errors) == 0

        # Log results
        if self.validation_errors:
            logger.error(f"Validation failed with {len(self.validation_errors)} errors")
            for error in self.validation_errors:
                logger.error(f"  - {error}")

        if self.validation_warnings:
            logger.warning(f"Validation completed with {len(self.validation_warnings)} warnings")
            for warning in self.validation_warnings:
                logger.warning(f"  - {warning}")

        if is_valid and not self.validation_warnings:
            logger.info("Validation passed successfully")

        return is_valid, self.validation_errors.copy(), self.validation_warnings.copy()
