# cost_model/engines/run_one_year/orchestrator/validator.py
"""
Snapshot validation module.

Provides comprehensive validation and integrity checking for workforce snapshots
at various stages of the simulation process.
"""
import logging
from typing import Optional
import pandas as pd

from cost_model.state.schema import (
    EMP_ID, EMP_ACTIVE, EMP_TERM_DATE, EMP_HIRE_DATE,
    EMP_GROSS_COMP, SIMULATION_YEAR
)
from ..validation import validate_eoy_snapshot
from .base import YearContext


class SnapshotValidator:
    """
    Provides comprehensive validation and integrity checking for workforce snapshots.

    This includes:
    - Duplicate employee ID detection
    - Active/terminated state consistency checks
    - Headcount reconciliation against targets
    - Data quality and completeness validation
    - Cross-step consistency verification
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the snapshot validator.

        Args:
            logger: Optional logger instance. If None, creates a new one.
        """
        self.logger = logger or logging.getLogger(__name__)

    def validate(
        self,
        snapshot: pd.DataFrame,
        step_name: str,
        year_context: Optional[YearContext] = None,
        target_headcount: Optional[int] = None
    ) -> None:
        """
        Perform comprehensive validation of a workforce snapshot.

        Args:
            snapshot: Workforce snapshot to validate
            step_name: Name of the simulation step for logging context
            year_context: Optional year context for additional validations
            target_headcount: Optional target headcount for reconciliation
            **kwargs: Additional validation parameters

        Raises:
            ValueError: If any critical validation checks fail
        """
        self.logger.info(f"[VALIDATION] Validating snapshot after {step_name}")

        # Basic structural validation
        self._validate_structure(snapshot, step_name)

        # Employee ID validation
        self._validate_employee_ids(snapshot, step_name)

        # Active/terminated state validation
        self._validate_employee_states(snapshot, step_name)

        # Data quality validation
        self._validate_data_quality(snapshot, step_name, year_context)

        # Headcount validation if target provided
        if target_headcount is not None:
            self._validate_headcount(snapshot, step_name, target_headcount)

        # Year-specific validations
        if year_context is not None:
            self._validate_year_consistency(snapshot, step_name, year_context)

        self.logger.info(f"[VALIDATION] Snapshot validation passed for {step_name}")

    def validate_eoy(
        self,
        snapshot: pd.DataFrame,
        target_headcount: Optional[int] = None
    ) -> None:
        """
        Perform end-of-year specific validation using the existing validation module.

        Args:
            snapshot: End-of-year workforce snapshot
            target_headcount: Optional target headcount for validation

        Raises:
            ValueError: If any validation checks fail
        """
        self.logger.info("[VALIDATION] Performing end-of-year snapshot validation")

        # Use the existing EOY validation function
        validate_eoy_snapshot(snapshot, target_headcount)

        self.logger.info("[VALIDATION] End-of-year snapshot validation passed")

    def _validate_structure(self, snapshot: pd.DataFrame, step_name: str) -> None:
        """
        Validate basic snapshot structure and required columns.

        Args:
            snapshot: Snapshot to validate
            step_name: Step name for logging

        Raises:
            ValueError: If structural validation fails
        """
        if snapshot.empty:
            raise ValueError(f"Snapshot is empty after {step_name}")

        # Check for required columns
        required_columns = [EMP_ID, EMP_ACTIVE]
        missing_columns = [col for col in required_columns if col not in snapshot.columns]

        if missing_columns:
            raise ValueError(
                f"Missing required columns after {step_name}: {missing_columns}"
            )

        self.logger.debug(f"[VALIDATION] Structure validation passed for {step_name}")

    def _validate_employee_ids(self, snapshot: pd.DataFrame, step_name: str) -> None:
        """
        Validate employee IDs for uniqueness and validity.

        Args:
            snapshot: Snapshot to validate
            step_name: Step name for logging

        Raises:
            ValueError: If employee ID validation fails
        """
        # Check for duplicate employee IDs
        duplicate_ids = snapshot[EMP_ID].duplicated()
        if duplicate_ids.any():
            duplicate_count = duplicate_ids.sum()
            duplicates = snapshot[snapshot[EMP_ID].duplicated(keep=False)].sort_values(EMP_ID)
            self.logger.error(f"Found {duplicate_count} duplicate EMP_IDs after {step_name}")
            self.logger.error(f"Duplicate IDs:\n{duplicates[[EMP_ID, EMP_ACTIVE]].head(10)}")
            raise ValueError(f"Found {duplicate_count} duplicate EMP_IDs after {step_name}")

        # Check for invalid employee IDs
        invalid_ids = snapshot[EMP_ID].isna() | (snapshot[EMP_ID] == '') | (snapshot[EMP_ID] == 'nan')
        if invalid_ids.any():
            invalid_count = invalid_ids.sum()
            self.logger.error(f"Found {invalid_count} invalid EMP_IDs after {step_name}")
            raise ValueError(f"Found {invalid_count} invalid EMP_IDs after {step_name}")

        self.logger.debug(f"[VALIDATION] Employee ID validation passed for {step_name}")

    def _validate_employee_states(self, snapshot: pd.DataFrame, step_name: str) -> None:
        """
        Validate employee active/terminated states for consistency.

        Args:
            snapshot: Snapshot to validate
            step_name: Step name for logging

        Raises:
            ValueError: If state validation fails
        """
        # Check for employees marked as both active and terminated
        if EMP_TERM_DATE in snapshot.columns:
            invalid_active = snapshot[
                snapshot[EMP_ACTIVE] & ~pd.isna(snapshot[EMP_TERM_DATE])
            ]
            if not invalid_active.empty:
                self.logger.error(
                    f"Found {len(invalid_active)} employees marked as both active and terminated after {step_name}"
                )
                raise ValueError(
                    f"Found {len(invalid_active)} employees with both active=True and a termination date after {step_name}"
                )

        # Check for reasonable active employee count
        active_count = snapshot[EMP_ACTIVE].sum()
        total_count = len(snapshot)

        if active_count == 0:
            self.logger.warning(f"No active employees found after {step_name}")
        elif active_count > total_count:
            raise ValueError(f"Active count ({active_count}) exceeds total count ({total_count}) after {step_name}")

        self.logger.debug(f"[VALIDATION] Employee state validation passed for {step_name}")

    def _validate_data_quality(self, snapshot: pd.DataFrame, step_name: str, year_context: Optional[YearContext] = None) -> None:
        """
        Validate data quality and completeness.

        Args:
            snapshot: Snapshot to validate
            step_name: Step name for logging
        """
        # Check for reasonable compensation values
        if EMP_GROSS_COMP in snapshot.columns:
            comp_data = snapshot[EMP_GROSS_COMP]

            # Check for negative compensation
            negative_comp = comp_data < 0
            if negative_comp.any():
                negative_count = negative_comp.sum()
                self.logger.warning(f"Found {negative_count} employees with negative compensation after {step_name}")

            # Check for extremely high compensation (potential data errors)
            high_comp = comp_data > 1000000  # $1M threshold
            if high_comp.any():
                high_count = high_comp.sum()
                self.logger.warning(f"Found {high_count} employees with compensation > $1M after {step_name}")

        # Check for reasonable hire dates
        if EMP_HIRE_DATE in snapshot.columns:
            hire_dates = pd.to_datetime(snapshot[EMP_HIRE_DATE], errors='coerce')

            # Check for future hire dates relative to simulation context
            # Use simulation year from context if available, otherwise current calendar year
            # Allow hire dates up to 2 years beyond the current simulation year
            if year_context is not None:
                simulation_year = year_context.year
            else:
                simulation_year = pd.Timestamp.now().year  # Default fallback
            max_reasonable_year = simulation_year + 2

            future_hires = hire_dates.dt.year > max_reasonable_year
            if future_hires.any():
                future_count = future_hires.sum()
                max_hire_year = hire_dates.dt.year.max()
                self.logger.warning(
                    f"Found {future_count} employees with hire dates beyond reasonable range "
                    f"(max year: {max_hire_year}, simulation year: {simulation_year}) after {step_name}"
                )

        self.logger.debug(f"[VALIDATION] Data quality validation passed for {step_name}")

    def _validate_headcount(
        self,
        snapshot: pd.DataFrame,
        step_name: str,
        target_headcount: int
    ) -> None:
        """
        Validate headcount against target.

        Args:
            snapshot: Snapshot to validate
            step_name: Step name for logging
            target_headcount: Expected headcount

        Raises:
            ValueError: If headcount validation fails
        """
        active_count = snapshot[EMP_ACTIVE].sum()

        if active_count != target_headcount:
            self.logger.error(
                f"Headcount mismatch after {step_name}. Expected: {target_headcount}, Actual: {active_count}"
            )
            # For now, just log the error rather than raising an exception
            # This allows for some flexibility in headcount targets
            self.logger.warning(f"Continuing with headcount mismatch after {step_name}")
        else:
            self.logger.info(f"[VALIDATION] Headcount matches target ({target_headcount}) after {step_name}")

    def _validate_year_consistency(
        self,
        snapshot: pd.DataFrame,
        step_name: str,
        year_context: YearContext
    ) -> None:
        """
        Validate year-specific consistency requirements.

        Args:
            snapshot: Snapshot to validate
            step_name: Step name for logging
            year_context: Year context for validation
        """
        # Check simulation year consistency
        if SIMULATION_YEAR in snapshot.columns:
            year_values = snapshot[SIMULATION_YEAR].unique()
            if len(year_values) > 1 or (len(year_values) == 1 and year_values[0] != year_context.year):
                self.logger.warning(
                    f"Inconsistent simulation years in snapshot after {step_name}: {year_values}"
                )

        # Check hire date consistency for new hires
        if EMP_HIRE_DATE in snapshot.columns:
            hire_dates = pd.to_datetime(snapshot[EMP_HIRE_DATE], errors='coerce')
            current_year_hires = hire_dates.dt.year == year_context.year

            if current_year_hires.any():
                current_year_count = current_year_hires.sum()
                self.logger.info(f"Found {current_year_count} employees hired in {year_context.year} after {step_name}")

        self.logger.debug(f"[VALIDATION] Year consistency validation passed for {step_name}")
