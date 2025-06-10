"""
Main snapshot building functionality.

Contains the refactored create_initial_snapshot and build_enhanced_yearly_snapshot
functions, broken down into smaller, focused components.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from .census_loader import CensusLoader
from .exceptions import SnapshotBuildError
from .logging_utils import (
    PerformanceMonitor,
    get_snapshot_logger,
    log_dataframe_info,
    timing_decorator,
)
from .models import SnapshotConfig, SnapshotMetrics
from .transformers import SnapshotTransformer
from .types import (
    ColumnMissingData,
    ColumnName,
    FilePath,
    SimulationYear,
    SnapshotCreationResult,
    YearlySnapshotResult,
)
from .validators import SnapshotValidator, validate_and_log_results

logger = get_snapshot_logger(__name__)


@timing_decorator(logger)
def create_initial_snapshot(start_year: SimulationYear, census_path: FilePath) -> pd.DataFrame:
    """
    Create the initial employee snapshot from census data.

    This is the main entry point that orchestrates the entire initial snapshot
    creation process using the refactored modular components.

    Args:
        start_year: The starting year for the simulation
        census_path: Path to the census data file (Parquet format)

    Returns:
        DataFrame containing the initial employee snapshot

    Raises:
        FileNotFoundError: If the census file doesn't exist
        ValueError: If the census data is invalid or missing required columns
        SnapshotBuildError: If snapshot creation fails
    """
    # Initialize performance monitoring
    performance_monitor = PerformanceMonitor(logger)
    performance_monitor.start_monitoring(f"create_initial_snapshot_year_{start_year}")

    logger.info(
        "Starting initial snapshot creation", start_year=start_year, census_path=str(census_path)
    )

    try:
        # Initialize configuration and components
        config = SnapshotConfig(start_year=start_year)
        loader = CensusLoader(config)
        transformer = SnapshotTransformer(config)
        validator = SnapshotValidator(config)

        performance_monitor.add_checkpoint("components_initialized")

        # Step 1: Load and validate census data (includes column standardization)
        logger.debug("Step 1: Loading census data")
        df = loader.load_census_file(census_path)
        performance_monitor.add_checkpoint("data_loaded", record_count=len(df))

        # Step 2: Validate required columns
        logger.debug("Step 2: Validating required columns")
        column_validation = loader.validate_required_columns(df)
        validate_and_log_results(column_validation, "required columns")
        performance_monitor.add_checkpoint("columns_validated")

        # Step 3: Filter terminated employees
        logger.debug("Step 3: Filtering terminated employees")
        initial_count = len(df)
        df = loader.filter_terminated_employees(df, start_year)
        performance_monitor.add_checkpoint(
            "employees_filtered", initial_count=initial_count, filtered_count=len(df)
        )

        # Step 4: Initialize employee data structure
        logger.debug("Step 4: Initializing employee data")
        df = loader.initialize_employee_data(df)
        performance_monitor.add_checkpoint("data_initialized")

        # Step 5: Infer missing data (job levels)
        logger.debug("Step 5: Inferring job levels")
        df = transformer.infer_job_levels(df)
        performance_monitor.add_checkpoint("job_levels_inferred")

        # Step 6: Normalize compensation by level
        logger.debug("Step 6: Normalizing compensation")
        df = transformer.normalize_compensation_by_level(df)
        performance_monitor.add_checkpoint("compensation_normalized")

        # Step 7: Calculate tenure and age
        logger.debug("Step 7: Calculating tenure and age")
        reference_date = datetime(start_year, 1, 1)
        df = transformer.apply_tenure_calculations(df, reference_date)
        df = transformer.apply_age_calculations(df, reference_date)
        performance_monitor.add_checkpoint("tenure_age_calculated")

        # Step 8: Calculate contributions
        logger.debug("Step 8: Calculating contributions")
        df = transformer.apply_contribution_calculations(df)
        performance_monitor.add_checkpoint("contributions_calculated")

        # Step 9: Create final snapshot DataFrame
        logger.debug("Step 9: Creating final snapshot")
        df = _create_snapshot_dataframe(df, start_year)
        performance_monitor.add_checkpoint("snapshot_finalized")

        # Step 10: Final validation and logging
        logger.debug("Step 10: Final validation")
        log_dataframe_info(logger, df, "Final initial snapshot", detailed=True)

        # Finalize performance monitoring
        final_metrics = performance_monitor.finish_monitoring(
            total_employees=len(df),
            final_columns=len(df.columns),
            missing_data_count=len(_count_missing_data(df)),
        )

        logger.info(
            "Initial snapshot created successfully",
            total_employees=len(df),
            total_columns=len(df.columns),
            performance_summary=final_metrics.get("final", {}),
        )

        return df

    except Exception as e:
        logger.error(f"Failed to create initial snapshot: {str(e)}")
        raise SnapshotBuildError(f"Initial snapshot creation failed: {str(e)}") from e


@timing_decorator(logger)
def build_enhanced_yearly_snapshot(
    start_of_year_snapshot: pd.DataFrame,
    end_of_year_snapshot: pd.DataFrame,
    year_events: pd.DataFrame,
    simulation_year: SimulationYear,
) -> pd.DataFrame:
    """
    Build enhanced yearly snapshot including all employees active during year.

    This is the refactored version that breaks down the complex yearly snapshot
    building process into focused, modular components.

    Args:
        start_of_year_snapshot: Snapshot at beginning of year
        end_of_year_snapshot: Snapshot at end of year (after all events)
        year_events: Events that occurred during this simulation year
        simulation_year: The simulation year being processed

    Returns:
        Enhanced yearly snapshot DataFrame

    Raises:
        SnapshotBuildError: If snapshot building fails
    """
    # Initialize performance monitoring
    performance_monitor = PerformanceMonitor(logger)
    performance_monitor.start_monitoring(f"build_enhanced_yearly_snapshot_year_{simulation_year}")

    logger.info(
        "Starting enhanced yearly snapshot creation",
        simulation_year=simulation_year,
        soy_employees=len(start_of_year_snapshot),
        eoy_employees=len(end_of_year_snapshot),
        events_count=len(year_events) if year_events is not None else 0,
    )

    try:
        # Import the new processor classes
        from .contributions_processor import ContributionsProcessor
        from .status_processor import StatusProcessor
        from .yearly_processor import YearlySnapshotProcessor

        # Initialize processors
        config = SnapshotConfig(start_year=simulation_year)
        yearly_processor = YearlySnapshotProcessor(config)
        contrib_processor = ContributionsProcessor()
        status_processor = StatusProcessor()
        transformer = SnapshotTransformer(config)
        performance_monitor.add_checkpoint("processors_initialized")

        # Step 1: Identify all employees active during the year
        logger.debug("Step 1: Identifying employees active during year")
        active_during_year_ids = yearly_processor.identify_employees_active_during_year(
            start_of_year_snapshot, year_events, simulation_year
        )
        performance_monitor.add_checkpoint(
            "employees_identified", active_count=len(active_during_year_ids)
        )

        # Step 2: Build base yearly snapshot from EOY data
        logger.debug("Step 2: Building base yearly snapshot")
        base_snapshot = yearly_processor.build_base_yearly_snapshot(
            end_of_year_snapshot, active_during_year_ids
        )
        performance_monitor.add_checkpoint("base_snapshot_built", base_employees=len(base_snapshot))

        # Step 3: Identify and reconstruct missing employees
        logger.debug("Step 3: Reconstructing missing employees")
        missing_ids = yearly_processor.identify_missing_employees(
            base_snapshot, active_during_year_ids
        )

        if missing_ids:
            missing_employees = yearly_processor.reconstruct_missing_employees(
                missing_ids, start_of_year_snapshot, year_events, simulation_year
            )

            if not missing_employees.empty:
                # Combine base snapshot with reconstructed employees
                yearly_snapshot = pd.concat([base_snapshot, missing_employees], ignore_index=True)
                performance_monitor.add_checkpoint(
                    "employees_reconstructed", reconstructed_count=len(missing_employees)
                )
            else:
                yearly_snapshot = base_snapshot
                performance_monitor.add_checkpoint("no_employees_reconstructed")
        else:
            yearly_snapshot = base_snapshot
            performance_monitor.add_checkpoint("no_missing_employees")

        # Step 4: Apply employee status determination
        logger.debug("Step 4: Determining employee status")
        yearly_snapshot = status_processor.apply_employee_status_eoy(yearly_snapshot)
        performance_monitor.add_checkpoint("status_determined")

        # Step 5: Apply contribution calculations
        logger.debug("Step 5: Calculating contributions")
        yearly_snapshot = contrib_processor.apply_contribution_calculations(yearly_snapshot)
        performance_monitor.add_checkpoint("contributions_calculated")

        # Step 6: Calculate tenure for all employees
        logger.debug("Step 6: Calculating tenure")
        yearly_snapshot = _apply_tenure_calculations(yearly_snapshot, simulation_year)
        performance_monitor.add_checkpoint("tenure_calculated")

        # Step 7: Apply age calculations
        logger.debug("Step 7: Calculating age")
        yearly_snapshot = _apply_age_calculations(yearly_snapshot, simulation_year)
        performance_monitor.add_checkpoint("age_calculated")

        # Step 8: Validate final snapshot
        logger.debug("Step 8: Validating yearly snapshot")
        _validate_yearly_snapshot(yearly_snapshot, status_processor, contrib_processor)
        performance_monitor.add_checkpoint("validation_complete")

        # Log final snapshot info and complete performance monitoring
        log_dataframe_info(logger, yearly_snapshot, "Final yearly snapshot", detailed=True)

        status_counts = yearly_snapshot.get("employee_status_eoy", pd.Series()).value_counts()
        final_metrics = performance_monitor.finish_monitoring(
            total_employees=len(yearly_snapshot),
            final_columns=len(yearly_snapshot.columns),
            status_distribution=status_counts.to_dict(),
        )

        logger.info(
            "Enhanced yearly snapshot completed successfully",
            simulation_year=simulation_year,
            total_employees=len(yearly_snapshot),
            status_distribution=status_counts.to_dict(),
            performance_summary=final_metrics.get("final", {}),
        )

        return yearly_snapshot

    except Exception as e:
        logger.error(f"Failed to build enhanced yearly snapshot: {str(e)}")
        raise SnapshotBuildError(f"Enhanced yearly snapshot building failed: {str(e)}") from e


def _create_snapshot_dataframe(df: pd.DataFrame, start_year: SimulationYear) -> pd.DataFrame:
    """
    Create the final snapshot DataFrame with proper column ordering and types.

    Args:
        df: DataFrame with processed employee data
        start_year: Starting year for simulation

    Returns:
        DataFrame formatted as proper snapshot
    """
    from cost_model.state.snapshot.constants import SNAPSHOT_COLS, SNAPSHOT_DTYPES

    from .constants import CENSUS_COLUMN_MAPPINGS

    logger.debug("Creating final snapshot DataFrame")

    # Add simulation year using original schema name
    df["simulation_year"] = start_year

    # Create specific reverse mapping for the known census file format
    reverse_mapping = {
        "EMP_ID": "employee_id",
        "EMP_HIRE_DATE": "employee_hire_date",
        "EMP_BIRTH_DATE": "employee_birth_date",
        "EMP_GROSS_COMP": "employee_gross_compensation",
        "EMP_TERM_DATE": "employee_termination_date",
        "EMP_DEFERRAL_RATE": "employee_deferral_rate",
        "EMP_ACTIVE": "active",
        "EMP_LEVEL": "employee_level",
    }

    # Map standardized columns back to original schema names
    rename_mapping = {}
    for col in df.columns:
        if col in reverse_mapping:
            original_name = reverse_mapping[col]
            rename_mapping[col] = original_name

    # Apply reverse mapping
    if rename_mapping:
        logger.debug(f"Mapping columns back to original schema: {rename_mapping}")
        df = df.rename(columns=rename_mapping)

    # Ensure all required snapshot columns exist
    for col in SNAPSHOT_COLS:
        if col not in df.columns:
            logger.debug(f"Adding missing snapshot column: {col}")
            df[col] = np.nan

    # Select and order columns according to snapshot schema
    df = df[SNAPSHOT_COLS].copy()

    # Apply proper data types
    for col, dtype in SNAPSHOT_DTYPES.items():
        if col in df.columns:
            try:
                if dtype == "datetime64[ns]":
                    df[col] = pd.to_datetime(df[col])
                elif dtype == "bool":
                    df[col] = df[col].astype(bool)
                elif dtype in ["int64", "Int64"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                elif dtype == "float64":
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                else:
                    df[col] = df[col].astype(dtype)
            except Exception as e:
                logger.warning(f"Could not convert column {col} to {dtype}: {e}")

    return df


def _apply_tenure_calculations(
    snapshot: pd.DataFrame, simulation_year: SimulationYear
) -> pd.DataFrame:
    """
    Apply tenure calculations for yearly snapshot.

    Args:
        snapshot: Yearly snapshot DataFrame
        simulation_year: Year being processed

    Returns:
        Updated snapshot with tenure calculations
    """
    logger.debug("Applying tenure calculations for yearly snapshot")

    snapshot = snapshot.copy()
    hire_date_col = "employee_hire_date"
    term_date_col = "employee_termination_date"
    status_col = "employee_status_eoy"
    tenure_col = "employee_tenure"
    tenure_band_col = "employee_tenure_band"

    # Calculate tenure based on employee status
    end_of_year = pd.Timestamp(f"{simulation_year}-12-31")

    for idx, row in snapshot.iterrows():
        hire_date = row.get(hire_date_col)
        term_date = row.get(term_date_col)
        status = row.get(status_col, "ACTIVE")

        if pd.isna(hire_date):
            snapshot.at[idx, tenure_col] = 0.0
            snapshot.at[idx, tenure_band_col] = "UNKNOWN"
            continue

        try:
            hire_dt = pd.to_datetime(hire_date)

            # For terminated employees, use termination date
            if status == "TERMINATED" and pd.notna(term_date):
                term_dt = pd.to_datetime(term_date)
                tenure_days = (term_dt - hire_dt).days
            else:
                # For active employees, use end of year
                tenure_days = (end_of_year - hire_dt).days

            tenure_years = max(0, tenure_days / 365.25)
            snapshot.at[idx, tenure_col] = tenure_years

            # Calculate tenure band
            if tenure_years < 1:
                snapshot.at[idx, tenure_band_col] = "NEW_HIRE"
            elif tenure_years < 5:
                snapshot.at[idx, tenure_band_col] = "EARLY_CAREER"
            elif tenure_years < 15:
                snapshot.at[idx, tenure_band_col] = "MID_CAREER"
            elif tenure_years < 25:
                snapshot.at[idx, tenure_band_col] = "SENIOR"
            else:
                snapshot.at[idx, tenure_band_col] = "VETERAN"

        except Exception as e:
            logger.warning(f"Could not calculate tenure for employee {row.get('employee_id')}: {e}")
            snapshot.at[idx, tenure_col] = 0.0
            snapshot.at[idx, tenure_band_col] = "UNKNOWN"

    logger.debug(f"Applied tenure calculations for {len(snapshot)} employees")
    return snapshot


def _apply_age_calculations(
    snapshot: pd.DataFrame, simulation_year: SimulationYear
) -> pd.DataFrame:
    """
    Apply age calculations for yearly snapshot.

    Args:
        snapshot: Yearly snapshot DataFrame
        simulation_year: Year being processed

    Returns:
        Updated snapshot with age calculations
    """
    logger.debug("Applying age calculations for yearly snapshot")

    snapshot = snapshot.copy()
    end_of_year = pd.Timestamp(f"{simulation_year}-12-31")

    try:
        # Try to use existing age calculation function
        from cost_model.state.age import apply_age

        snapshot = apply_age(snapshot, as_of=end_of_year)
        logger.debug("Applied age calculations using existing function")

    except Exception as e:
        logger.warning(f"Existing age function failed, using manual calculation: {e}")
        snapshot = _manual_age_calculation(snapshot, end_of_year)

    return snapshot


def _manual_age_calculation(snapshot: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """Manual age calculation when existing function fails."""
    birth_date_col = "employee_birth_date"
    age_col = "employee_age"
    age_band_col = "employee_age_band"

    if birth_date_col not in snapshot.columns:
        logger.warning("No birth date column available for age calculation")
        snapshot[age_col] = np.nan
        snapshot[age_band_col] = "UNKNOWN"
        return snapshot

    # Calculate age in years
    birth_dates = pd.to_datetime(snapshot[birth_date_col], errors="coerce")
    age_days = (reference_date - birth_dates).dt.days
    snapshot[age_col] = age_days / 365.25

    # Handle invalid ages
    invalid_age = (snapshot[age_col] < 0) | (snapshot[age_col] > 100)
    if invalid_age.any():
        logger.warning(f"Found {invalid_age.sum()} employees with invalid ages")
        snapshot.loc[invalid_age, age_col] = np.nan

    # Calculate age bands
    def calculate_age_band(age):
        if pd.isna(age):
            return "UNKNOWN"
        elif age < 30:
            return "YOUNG"
        elif age < 40:
            return "EARLY_CAREER"
        elif age < 50:
            return "MID_CAREER"
        elif age < 60:
            return "SENIOR"
        elif age < 67:
            return "PRE_RETIREMENT"
        else:
            return "POST_RETIREMENT"

    snapshot[age_band_col] = snapshot[age_col].apply(calculate_age_band)

    return snapshot


def _validate_yearly_snapshot(snapshot: pd.DataFrame, status_processor, contrib_processor) -> None:
    """
    Validate the yearly snapshot and log results.

    Args:
        snapshot: Completed yearly snapshot
        status_processor: Status processor for validation
        contrib_processor: Contributions processor for validation
    """
    logger.debug("Validating yearly snapshot")

    # Validate employee status
    status_validation = status_processor.validate_employee_status(snapshot)
    if status_validation["warnings"]:
        for warning in status_validation["warnings"]:
            logger.warning(f"Status validation: {warning}")
    if status_validation["errors"]:
        for error in status_validation["errors"]:
            logger.error(f"Status validation: {error}")

    # Validate contributions
    contrib_validation = contrib_processor.validate_contributions(snapshot)
    if contrib_validation["warnings"]:
        for warning in contrib_validation["warnings"]:
            logger.warning(f"Contribution validation: {warning}")
    if contrib_validation["errors"]:
        for error in contrib_validation["errors"]:
            logger.error(f"Contribution validation: {error}")

    # Calculate and log status metrics
    status_metrics = status_processor.calculate_status_metrics(snapshot)
    logger.info(f"Yearly snapshot validation complete: {status_metrics.get('status_counts', {})}")


def _count_missing_data(df: pd.DataFrame) -> ColumnMissingData:
    """Count missing data by column."""
    missing_counts = {}
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            missing_counts[col] = null_count
    return missing_counts
