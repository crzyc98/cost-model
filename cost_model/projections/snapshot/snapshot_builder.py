# cost_model/projections/snapshot/snapshot_builder.py
"""
Main snapshot builder orchestrator.
Coordinates census processing, data transformation, and validation to create snapshots.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

from cost_model.state.snapshot.constants import SNAPSHOT_COLS as SNAPSHOT_COL_NAMES
from cost_model.state.snapshot.constants import SNAPSHOT_DTYPES

from .census_processor import CensusProcessor
from .data_transformer import DataTransformer
from .validation_engine import ValidationEngine

logger = logging.getLogger(__name__)


class SnapshotBuilder:
    """
    Main orchestrator for building employee snapshots from census data.

    Coordinates the census processing, data transformation, and validation
    to produce clean, validated snapshot DataFrames.
    """

    def __init__(self):
        self.census_processor = CensusProcessor()
        self.data_transformer = DataTransformer()
        self.validation_engine = ValidationEngine()

    def create_initial_snapshot(
        self, start_year: int, census_path: Union[str, Path]
    ) -> pd.DataFrame:
        """
        Create the initial employee snapshot from census data.

        This is the main entry point that orchestrates the snapshot creation process
        by calling smaller, focused helper functions.

        Args:
            start_year: The starting year for the simulation
            census_path: Path to the census data file (Parquet or CSV format)

        Returns:
            DataFrame containing the initial employee snapshot

        Raises:
            FileNotFoundError: If the census file doesn't exist
            ValueError: If the census data is invalid or missing required columns
        """
        logger.info(
            "Creating initial snapshot for start year: %d from %s", start_year, str(census_path)
        )

        try:
            # Step 1: Load and validate census file
            raw_census_df = self._load_census_file(census_path)

            # Step 2: Preprocess census data
            census_df = self._preprocess_census_data(raw_census_df, start_year)

            # Step 3: Handle empty census edge case
            if census_df.empty:
                logger.warning("Census data is empty after preprocessing. Creating empty snapshot.")
                return self.create_empty_snapshot()

            # Step 4: Transform to snapshot format
            snapshot_df = self._transform_to_snapshot_format(census_df, start_year)

            # Step 5: Apply business rules and calculations
            snapshot_df = self._apply_business_rules(snapshot_df, start_year)

            # Step 6: Validate the final result
            self._validate_final_snapshot(snapshot_df, start_year)

            logger.info("Successfully created initial snapshot with %d employees", len(snapshot_df))
            return snapshot_df

        except Exception as e:
            logger.error("Failed to create initial snapshot: %s", str(e), exc_info=True)
            raise

    def _load_census_file(self, census_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load census data from file, supporting multiple formats.

        Args:
            census_path: Path to census data file

        Returns:
            Raw census DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        logger.debug("Loading census file: %s", str(census_path))
        return self.census_processor.load_census_file(census_path)

    def _preprocess_census_data(self, census_df: pd.DataFrame, start_year: int) -> pd.DataFrame:
        """
        Preprocess census data including column standardization and filtering.

        Args:
            census_df: Raw census DataFrame
            start_year: Simulation start year

        Returns:
            Preprocessed census DataFrame
        """
        logger.debug("Preprocessing census data")

        # Deduplicate columns
        census_df = census_df.loc[:, ~census_df.columns.duplicated()]

        # Standardize column names
        census_df = self.census_processor.standardize_column_names(census_df)

        # Validate required columns
        census_df = self.census_processor.validate_required_columns(census_df)

        # Filter active employees
        census_df = self.census_processor.filter_active_employees(census_df, start_year)

        return census_df

    def _transform_to_snapshot_format(
        self, census_df: pd.DataFrame, start_year: int
    ) -> pd.DataFrame:
        """
        Transform census data to snapshot format with all required columns.

        Args:
            census_df: Preprocessed census DataFrame
            start_year: Simulation start year

        Returns:
            Snapshot DataFrame with proper structure
        """
        logger.debug("Transforming census to snapshot format")
        return self.data_transformer.transform_census_to_snapshot(census_df, start_year)

    def _apply_business_rules(self, snapshot_df: pd.DataFrame, start_year: int) -> pd.DataFrame:
        """
        Apply business rules and calculations to the snapshot.

        Args:
            snapshot_df: Snapshot DataFrame
            start_year: Simulation start year

        Returns:
            Enhanced snapshot DataFrame with business rules applied
        """
        logger.debug("Applying business rules to snapshot")

        # Apply any additional business logic here
        # For now, this is a placeholder for future enhancements

        return snapshot_df

    def _validate_final_snapshot(self, snapshot_df: pd.DataFrame, start_year: int) -> None:
        """
        Validate the final snapshot for data integrity and business rules.

        Args:
            snapshot_df: Final snapshot DataFrame
            start_year: Simulation start year

        Raises:
            ValueError: If validation fails
        """
        logger.debug("Validating final snapshot")

        is_valid, errors, warnings = self.validation_engine.validate_snapshot(
            snapshot_df, start_year
        )

        if warnings:
            for warning in warnings:
                logger.warning(f"Snapshot validation warning: {warning}")

        if not is_valid:
            error_msg = f"Snapshot validation failed: {'; '.join(errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def build_enhanced_yearly_snapshot(
        self,
        start_of_year_snapshot: pd.DataFrame,
        end_of_year_snapshot: pd.DataFrame,
        year_events: pd.DataFrame,
        simulation_year: int,
    ) -> pd.DataFrame:
        """
        Build an enhanced yearly snapshot that includes all employees who were active
        at any point during the specified simulation year.

        This refactored implementation breaks down the original 600+ line function
        into focused, single-responsibility helper methods.

        Args:
            start_of_year_snapshot: Snapshot at the beginning of the year
            end_of_year_snapshot: Snapshot at the end of the year (after all events)
            year_events: Events that occurred during this simulation year
            simulation_year: The simulation year being processed

        Returns:
            Enhanced yearly snapshot DataFrame with all employees active during the year
        """
        logger.info("Building enhanced yearly snapshot for year %d", simulation_year)

        try:
            # Step 1: Identify employee populations
            employee_populations = self._identify_employee_populations(
                start_of_year_snapshot, year_events, simulation_year
            )

            # Step 2: Build core yearly snapshot from multiple sources
            yearly_snapshot = self._build_core_yearly_snapshot(
                start_of_year_snapshot,
                end_of_year_snapshot,
                year_events,
                employee_populations,
                simulation_year,
            )

            # Step 3: Set employee status and simulation year
            yearly_snapshot = self._set_employee_status_and_year(yearly_snapshot, simulation_year)

            # Step 4: Apply contribution calculations
            yearly_snapshot = self._apply_contribution_calculations(
                yearly_snapshot, simulation_year
            )

            # Step 5: Calculate tenure for all employees
            yearly_snapshot = self._calculate_tenure_for_all_employees(
                yearly_snapshot, simulation_year
            )

            # Step 6: Apply age calculations
            yearly_snapshot = self._apply_age_calculations(yearly_snapshot, simulation_year)

            # Step 7: Validate and log summary statistics
            self._validate_and_log_snapshot_summary(yearly_snapshot, simulation_year)

            logger.info(
                "Enhanced yearly snapshot for %d contains %d employees",
                simulation_year,
                len(yearly_snapshot),
            )
            return yearly_snapshot

        except Exception as e:
            logger.error(
                "Failed to build enhanced yearly snapshot for year %d: %s",
                simulation_year,
                str(e),
                exc_info=True,
            )
            raise

    def _identify_employee_populations(
        self, start_of_year_snapshot: pd.DataFrame, year_events: pd.DataFrame, simulation_year: int
    ) -> dict:
        """
        Identify different employee populations for the yearly snapshot.

        Returns:
            dict: Contains employee ID sets for different populations
        """
        from cost_model.state.schema import EMP_ACTIVE, EMP_ID, EVT_HIRE, EVT_TERM, EVT_TYPE

        logger.debug("Identifying employee populations for year %d", simulation_year)

        # Get employees active at start of year
        start_of_year_active = start_of_year_snapshot[start_of_year_snapshot[EMP_ACTIVE]][
            EMP_ID
        ].tolist()

        # Get employees hired during the year from events
        hire_events = (
            year_events[year_events[EVT_TYPE] == EVT_HIRE]
            if not year_events.empty
            else pd.DataFrame()
        )
        hired_during_year = hire_events[EMP_ID].tolist() if not hire_events.empty else []

        # Get employees terminated during the year from events
        term_events = (
            year_events[year_events[EVT_TYPE] == EVT_TERM]
            if not year_events.empty
            else pd.DataFrame()
        )
        terminated_during_year = term_events[EMP_ID].tolist() if not term_events.empty else []

        # Form the "Active During Year" set - anyone who was active at some point
        active_during_year = set(start_of_year_active + hired_during_year)

        logger.info(
            "Employee populations for year %d: SOY active=%d, hired=%d, terminated=%d, active during year=%d",
            simulation_year,
            len(start_of_year_active),
            len(hired_during_year),
            len(terminated_during_year),
            len(active_during_year),
        )

        return {
            "start_of_year_active": start_of_year_active,
            "hired_during_year": hired_during_year,
            "terminated_during_year": terminated_during_year,
            "active_during_year": list(active_during_year),
        }

    def _build_core_yearly_snapshot(
        self,
        start_of_year_snapshot: pd.DataFrame,
        end_of_year_snapshot: pd.DataFrame,
        year_events: pd.DataFrame,
        employee_populations: dict,
        simulation_year: int,
    ) -> pd.DataFrame:
        """
        Build the core yearly snapshot from multiple employee data sources.

        This is the most complex part of the original function and handles:
        - EOY employees
        - Missing employees from SOY
        - Terminated new hires reconstruction
        """
        logger.debug("Building core yearly snapshot")

        # For now, delegate to the original implementation to maintain functionality
        # This is the most complex part that would need careful step-by-step refactoring
        from cost_model.projections.snapshot_original import (
            build_enhanced_yearly_snapshot as original_build,
        )

        return original_build(
            start_of_year_snapshot, end_of_year_snapshot, year_events, simulation_year
        )

    def _set_employee_status_and_year(
        self, yearly_snapshot: pd.DataFrame, simulation_year: int
    ) -> pd.DataFrame:
        """
        Set employee status and simulation year for the snapshot.
        """
        from cost_model.projections.snapshot import _determine_employee_status_eoy
        from cost_model.state.schema import SIMULATION_YEAR

        logger.debug("Setting employee status and simulation year")

        # Determine end-of-year status for each employee
        yearly_snapshot["employee_status_eoy"] = yearly_snapshot.apply(
            lambda row: _determine_employee_status_eoy(row, simulation_year), axis=1
        )

        # Set simulation year
        yearly_snapshot[SIMULATION_YEAR] = simulation_year

        return yearly_snapshot

    def _apply_contribution_calculations(
        self, yearly_snapshot: pd.DataFrame, simulation_year: int
    ) -> pd.DataFrame:
        """
        Apply contribution calculations to the yearly snapshot.
        """
        logger.debug("Applying contribution calculations")

        # For now, return as-is. This would be implemented based on business requirements
        # This could involve calling contribution calculation engines

        return yearly_snapshot

    def _calculate_tenure_for_all_employees(
        self, yearly_snapshot: pd.DataFrame, simulation_year: int
    ) -> pd.DataFrame:
        """
        Calculate tenure and tenure bands for all employees in the snapshot.
        """
        logger.debug("Calculating tenure for all employees")

        # Use the existing data transformer for tenure calculations
        yearly_snapshot = self.data_transformer.calculate_tenure(yearly_snapshot, simulation_year)
        yearly_snapshot = self.data_transformer.assign_tenure_bands(yearly_snapshot)

        return yearly_snapshot

    def _apply_age_calculations(
        self, yearly_snapshot: pd.DataFrame, simulation_year: int
    ) -> pd.DataFrame:
        """
        Apply age and age band calculations to the snapshot.
        """
        logger.debug("Applying age calculations")

        # Use the existing data transformer for age calculations
        yearly_snapshot = self.data_transformer.calculate_age_information(
            yearly_snapshot, simulation_year
        )

        return yearly_snapshot

    def _validate_and_log_snapshot_summary(
        self, yearly_snapshot: pd.DataFrame, simulation_year: int
    ) -> None:
        """
        Validate the yearly snapshot and log summary statistics.
        """
        logger.debug("Validating and logging snapshot summary")

        # Log basic statistics
        logger.info("Yearly snapshot summary for %d:", simulation_year)
        logger.info("  Total employees: %d", len(yearly_snapshot))

        # Log status distribution if available
        if "employee_status_eoy" in yearly_snapshot.columns:
            status_counts = yearly_snapshot["employee_status_eoy"].value_counts()
            for status, count in status_counts.items():
                logger.info("  %s: %d", status, count)

        # Validate data integrity
        is_valid, errors, warnings = self.validation_engine.validate_snapshot(
            yearly_snapshot, simulation_year
        )

        if warnings:
            for warning in warnings:
                logger.warning("Snapshot validation warning: %s", warning)

        if not is_valid:
            for error in errors:
                logger.error("Snapshot validation error: %s", error)

    def validate_snapshot_integrity(
        self, snapshot_df: pd.DataFrame, simulation_year: int
    ) -> Tuple[bool, list, list]:
        """
        Validate snapshot integrity and business rules.

        Args:
            snapshot_df: Snapshot DataFrame to validate
            simulation_year: Current simulation year

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        return self.validation_engine.validate_snapshot(snapshot_df, simulation_year)

    def create_empty_snapshot(self) -> pd.DataFrame:
        """
        Create an empty snapshot with proper schema.

        Returns:
            Empty DataFrame with correct columns and types
        """
        logger.info("Creating empty snapshot")
        return pd.DataFrame(columns=SNAPSHOT_COL_NAMES).astype(SNAPSHOT_DTYPES)
