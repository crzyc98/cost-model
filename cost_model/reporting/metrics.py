# cost_model/reporting/metrics.py
"""
Functions to calculate summary metrics from simulation results.
"""

import pandas as pd
import logging
from typing import Dict, Any
import numpy as np

# Import canonical column names from schema
from cost_model.state.schema import (
    SUMMARY_YEAR, EMP_ID, EMP_GROSS_COMP, EMP_STATUS_EOY, EMP_BIRTH_DATE, EMP_DEFERRAL_RATE,
    EMP_CONTR, EMPLOYER_CORE_CONTRIB, EMPLOYER_MATCH_CONTRIB, SIMULATION_YEAR, ACTIVE_STATUS,
    EMP_PLAN_YEAR_COMP
)

# Define status constants for employee_status_eoy
EMPLOYEE_STATUS_TERMINATED = "Terminated"
EMPLOYEE_STATUS_ACTIVE = "Active"
EMPLOYEE_STATUS_INACTIVE = "Inactive"

logger = logging.getLogger(__name__)


def calculate_summary_metrics(
    all_results_df: pd.DataFrame,
    config: Dict[str, Any],  # Or specific config model if preferred
) -> pd.DataFrame:
    """
    Calculates summary metrics across all simulation years.

    Args:
        all_results_df: DataFrame containing combined results from all years,
                        with a 'simulation_year' column.
        config: The scenario configuration dictionary or object.

    Returns:
        A DataFrame containing summary metrics, indexed by year.
    """
    if all_results_df is None or all_results_df.empty:
        logger.warning(
            "Input DataFrame for summary metrics is empty. Returning empty results."
        )
        return pd.DataFrame()

    logger.info(
        f"Calculating summary metrics for {len(all_results_df)} total records across years..."
    )

    # Log available columns for debugging
    logger.info(f"Available columns in all_results_df: {sorted(all_results_df.columns.tolist())}")
    logger.info(f"DataFrame shape: {all_results_df.shape}")
    logger.info(f"Simulation years: {sorted(all_results_df[SIMULATION_YEAR].unique()) if SIMULATION_YEAR in all_results_df.columns else 'SIMULATION_YEAR column missing'}")

    # Enhanced debugging for key columns
    key_columns_to_check = [EMP_STATUS_EOY, EMP_GROSS_COMP, EMP_CONTR, EMPLOYER_CORE_CONTRIB, EMPLOYER_MATCH_CONTRIB]
    for col in key_columns_to_check:
        if col in all_results_df.columns:
            logger.info(f"Column '{col}' found - sample values: {all_results_df[col].head().tolist()}")
            logger.info(f"Column '{col}' - unique values: {all_results_df[col].unique()[:10]}")  # First 10 unique values
            logger.info(f"Column '{col}' - null count: {all_results_df[col].isnull().sum()}")
            if col == EMP_STATUS_EOY:
                logger.info(f"Column '{col}' - value counts: {all_results_df[col].value_counts().to_dict()}")
        else:
            logger.warning(f"Column '{col}' NOT FOUND in all_results_df")

    # Check for alternative column names that might exist
    alternative_names = {
        'employee_status_eoy': EMP_STATUS_EOY,
        'employee_gross_compensation': EMP_GROSS_COMP,
        'employee_contribution': EMP_CONTR,
        'employer_core_contribution': EMPLOYER_CORE_CONTRIB,
        'employer_match_contribution': EMPLOYER_MATCH_CONTRIB
    }

    for alt_name, canonical_name in alternative_names.items():
        if alt_name in all_results_df.columns and canonical_name not in all_results_df.columns:
            logger.info(f"Found alternative column name '{alt_name}' for '{canonical_name}'")

    # --- Ensure correct data types ---
    # Handle birth date column with flexible naming
    birth_date_col = None
    for col in [EMP_BIRTH_DATE, "employee_birth_date"]:
        if col in all_results_df.columns:
            birth_date_col = col
            break

    if birth_date_col:
        all_results_df[birth_date_col] = pd.to_datetime(
            all_results_df[birth_date_col], errors="coerce"
        )
        logger.debug(f"Using birth date column: {birth_date_col}")
    else:
        logger.warning("No birth date column found. Age calculations will be skipped.")

    # Handle eligibility column
    eligibility_col = None
    for col in ["is_eligible", "eligible"]:
        if col in all_results_df.columns:
            eligibility_col = col
            # Handle NaN values before converting to bool
            all_results_df[col] = all_results_df[col].fillna(False).astype(bool)
            break

    if not eligibility_col:
        logger.warning("No eligibility column found. Creating default eligibility (all True).")
        all_results_df["is_eligible"] = True
        eligibility_col = "is_eligible"

    # Handle numeric columns with flexible naming
    numeric_col_mapping = {
        EMP_DEFERRAL_RATE: ["employee_deferral_rate", EMP_DEFERRAL_RATE],
        EMP_CONTR: ["employee_contribution", EMP_CONTR],
        EMPLOYER_MATCH_CONTRIB: ["employer_match_contribution", EMPLOYER_MATCH_CONTRIB],
        EMPLOYER_CORE_CONTRIB: ["employer_core_contribution", EMPLOYER_CORE_CONTRIB],
        EMP_GROSS_COMP: ["employee_gross_compensation", EMP_GROSS_COMP],
        EMP_PLAN_YEAR_COMP: ["employee_plan_year_compensation", EMP_PLAN_YEAR_COMP],
    }

    for canonical_name, possible_names in numeric_col_mapping.items():
        found_col = None
        for col in possible_names:
            if col in all_results_df.columns:
                found_col = col
                break

        if found_col:
            # Log original values before conversion
            logger.info(f"Column '{found_col}' - original sample values: {all_results_df[found_col].head().tolist()}")
            logger.info(f"Column '{found_col}' - original data type: {all_results_df[found_col].dtype}")
            logger.info(f"Column '{found_col}' - null count before conversion: {all_results_df[found_col].isnull().sum()}")

            # Convert to numeric
            original_values = all_results_df[found_col].copy()
            all_results_df[found_col] = pd.to_numeric(
                all_results_df[found_col], errors="coerce"
            ).fillna(0)

            # Log conversion results
            converted_nulls = pd.to_numeric(original_values, errors="coerce").isnull().sum()
            if converted_nulls > 0:
                logger.warning(f"Column '{found_col}' - {converted_nulls} values converted to NaN during numeric conversion")
                # Show some examples of problematic values
                problematic_mask = pd.to_numeric(original_values, errors="coerce").isnull()
                if problematic_mask.any():
                    problematic_values = original_values[problematic_mask].unique()[:5]
                    logger.warning(f"Column '{found_col}' - sample problematic values: {problematic_values.tolist()}")

            logger.info(f"Column '{found_col}' - final sample values: {all_results_df[found_col].head().tolist()}")
            logger.info(f"Column '{found_col}' - final sum: {all_results_df[found_col].sum()}")
            logger.info(f"Column '{found_col}' - final mean: {all_results_df[found_col].mean()}")

            # Create canonical name alias if different
            if found_col != canonical_name:
                all_results_df[canonical_name] = all_results_df[found_col]
            logger.info(f"Using {canonical_name}: {found_col}")
        else:
            logger.warning(f"Column {canonical_name} not found. Creating with default value 0.")
            all_results_df[canonical_name] = 0.0

    # --- Feature Engineering ---
    # Calculate age if birth date is available
    if birth_date_col and birth_date_col in all_results_df.columns:
        all_results_df["age"] = (
            all_results_df[SIMULATION_YEAR]
            - all_results_df[birth_date_col].dt.year
        )
    else:
        logger.warning("Age calculation skipped due to missing birth date.")
        all_results_df["age"] = np.nan

    # Calculate participation
    all_results_df["is_participant"] = all_results_df[eligibility_col] & (
        all_results_df[EMP_DEFERRAL_RATE] > 0
    )

    # Row-level ER and total compensation
    all_results_df["total_er_contribution"] = (
        all_results_df[EMPLOYER_MATCH_CONTRIB]
        + all_results_df[EMPLOYER_CORE_CONTRIB]
    )
    all_results_df["total_compensation"] = (
        all_results_df[EMP_GROSS_COMP]
        + all_results_df[EMP_CONTR]
        + all_results_df["total_er_contribution"]
    )

    # --- Group and Aggregate ---
    # Calculate terminations by counting employees with 'Terminated' status
    terminations_by_year = None
    if EMP_STATUS_EOY in all_results_df.columns:
        # Log the unique values in the status column for debugging
        status_values = all_results_df[EMP_STATUS_EOY].value_counts()
        logger.info(f"Employee status distribution: {status_values.to_dict()}")

        # Count terminated employees using the canonical constant
        terminated_mask = all_results_df[EMP_STATUS_EOY] == EMPLOYEE_STATUS_TERMINATED
        logger.info(f"Total terminated employees found: {terminated_mask.sum()}")

        if terminated_mask.sum() > 0:
            terminations_by_year = (
                all_results_df[terminated_mask]
                .groupby(SIMULATION_YEAR)
                .size()
                .reset_index(name='total_terminations')
            )
            logger.info(f"Calculated terminations by year: {terminations_by_year.to_dict('records')}")
        else:
            logger.warning("No terminated employees found in the data")
    else:
        logger.warning(f"Column {EMP_STATUS_EOY} not found. Terminations will be set to 0.")

    # Determine employee ID column
    emp_id_col = EMP_ID if EMP_ID in all_results_df.columns else "employee_id"
    if emp_id_col not in all_results_df.columns:
        logger.error("No employee ID column found. Cannot calculate headcount.")
        return pd.DataFrame()

    # Log data before aggregation
    logger.info(f"Data summary before aggregation:")
    logger.info(f"  - Total rows: {len(all_results_df)}")
    logger.info(f"  - {EMP_GROSS_COMP} sum: {all_results_df[EMP_GROSS_COMP].sum()}")
    logger.info(f"  - {EMP_GROSS_COMP} mean: {all_results_df[EMP_GROSS_COMP].mean()}")
    logger.info(f"  - {EMP_CONTR} sum: {all_results_df[EMP_CONTR].sum()}")
    logger.info(f"  - total_er_contribution sum: {all_results_df['total_er_contribution'].sum()}")
    logger.info(f"  - total_compensation sum: {all_results_df['total_compensation'].sum()}")

    summary_metrics = (
        all_results_df.groupby(SIMULATION_YEAR)
        .agg(
            active_headcount=(emp_id_col, "size"),
            eligible_count=(eligibility_col, "sum"),
            participant_count=("is_participant", "sum"),
            total_employee_gross_compensation=(EMP_GROSS_COMP, "sum"),
            total_plan_year_compensation=(EMP_PLAN_YEAR_COMP, "sum"),
            total_ee_contribution=(EMP_CONTR, "sum"),
            total_er_contribution=("total_er_contribution", "sum"),
            total_compensation=("total_compensation", "sum"),
            avg_compensation=(EMP_GROSS_COMP, "mean"),  # Use expected output name
            avg_plan_year_compensation=(EMP_PLAN_YEAR_COMP, "mean"),
            avg_age_active=("age", "mean"),
        )
        .reset_index()
    )

    # Log aggregation results
    logger.info(f"Aggregation results:")
    for _, row in summary_metrics.iterrows():
        year = row[SIMULATION_YEAR]
        logger.info(f"  Year {year}:")
        logger.info(f"    - active_headcount: {row['active_headcount']}")
        logger.info(f"    - total_employee_gross_compensation: {row['total_employee_gross_compensation']}")
        logger.info(f"    - total_plan_year_compensation: {row['total_plan_year_compensation']}")
        logger.info(f"    - avg_compensation: {row['avg_compensation']}")
        logger.info(f"    - avg_plan_year_compensation: {row['avg_plan_year_compensation']}")
        logger.info(f"    - total_compensation: {row['total_compensation']}")

    # Add terminations data
    if terminations_by_year is not None and not terminations_by_year.empty:
        summary_metrics = pd.merge(
            summary_metrics, terminations_by_year,
            on=SIMULATION_YEAR, how="left"
        )
    else:
        summary_metrics['total_terminations'] = 0

    # Fill any NaN values in total_terminations with 0
    summary_metrics['total_terminations'] = summary_metrics['total_terminations'].fillna(0).astype(int)

    # Calculate participation rate
    summary_metrics["participation_rate"] = np.where(
        summary_metrics["eligible_count"] > 0,
        summary_metrics["participant_count"] / summary_metrics["eligible_count"],
        0,
    )

    # Calculate average deferral rate for participants
    participants_df = all_results_df[all_results_df["is_participant"]]
    if not participants_df.empty:
        avg_deferral_rate = (
            participants_df.groupby(SIMULATION_YEAR)[EMP_DEFERRAL_RATE]
            .mean()
            .reset_index()
        )
        avg_deferral_rate = avg_deferral_rate.rename(
            columns={EMP_DEFERRAL_RATE: "avg_deferral_rate_participants"}
        )

        # Merge with summary metrics
        summary_metrics = pd.merge(
            summary_metrics, avg_deferral_rate, on=SIMULATION_YEAR, how="left"
        )
    else:
        summary_metrics["avg_deferral_rate_participants"] = 0

    # Log summary before final processing
    logger.info(f"Summary metrics calculated for years: {sorted(summary_metrics[SIMULATION_YEAR].unique())}")
    logger.debug(f"Summary metrics columns: {summary_metrics.columns.tolist()}")
    logger.debug(f"Sample summary data:\n{summary_metrics.head().to_string()}")

    # Round values for presentation
    rounding_dict = {
        "avg_compensation": 0,  # Updated to match our output column name
        "avg_plan_year_compensation": 0,  # New prorated compensation average
        "avg_age_active": 1,
        "participation_rate": 4,
        "avg_deferral_rate_participants": 4,
        "total_ee_contribution": 0,
        "total_er_contribution": 0,
        "total_employee_gross_compensation": 0,
        "total_plan_year_compensation": 0,  # New prorated compensation total
        "total_compensation": 0,
    }

    for col, decimals in rounding_dict.items():
        if col in summary_metrics.columns:
            summary_metrics[col] = summary_metrics[col].round(decimals)

    # Ensure we have the expected output columns for projection_cli_summary_statistics.parquet
    # Based on the user's problem description, these are the key columns that should be populated:
    # - year (from SUMMARY_YEAR)
    # - active_headcount
    # - total_terminations
    # - avg_compensation
    # - total_compensation
    # - new_hires (if available)
    # - terminated_employees (if available)

    # Add canonical year column as first column
    summary_metrics.insert(0, SUMMARY_YEAR, summary_metrics[SIMULATION_YEAR])

    # Log final results for debugging
    logger.info("Final summary metrics:")
    for _, row in summary_metrics.iterrows():
        year = row[SUMMARY_YEAR]
        active_hc = row.get('active_headcount', 'N/A')
        total_terms = row.get('total_terminations', 'N/A')
        avg_comp = row.get('avg_compensation', 'N/A')
        avg_plan_comp = row.get('avg_plan_year_compensation', 'N/A')
        total_comp = row.get('total_compensation', 'N/A')
        total_plan_comp = row.get('total_plan_year_compensation', 'N/A')
        logger.info(f"  Year {year}: active_headcount={active_hc}, total_terminations={total_terms}, "
                   f"avg_compensation={avg_comp}, avg_plan_year_compensation={avg_plan_comp}, "
                   f"total_compensation={total_comp}, total_plan_year_compensation={total_plan_comp}")

    # Set index to simulation_year for consistency with existing code
    summary_metrics.set_index(SIMULATION_YEAR, inplace=True)

    logger.info("Summary metric calculation complete.")
    return summary_metrics
