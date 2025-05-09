# cost_model/reporting/metrics.py
"""
Functions to calculate summary metrics from simulation results.
"""

import pandas as pd
import logging
from typing import Dict, Any
import numpy as np

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

    # --- Ensure correct data types ---
    all_results_df["employee_birth_date"] = pd.to_datetime(
        all_results_df["employee_birth_date"], errors="coerce"
    )
    all_results_df["is_eligible"] = all_results_df["is_eligible"].astype(bool)

    numeric_cols = [
        "employee_deferral_rate",
        "employee_contribution",
        "employer_match_contribution",
        "employer_core_contribution",
        "employee_gross_compensation",
    ]
    for col in numeric_cols:
        all_results_df[col] = pd.to_numeric(
            all_results_df[col], errors="coerce"
        ).fillna(0)

    # --- Feature Engineering ---
    all_results_df["age"] = (
        all_results_df["simulation_year"]
        - all_results_df["employee_birth_date"].dt.year
    )
    all_results_df["is_participant"] = all_results_df["is_eligible"] & (
        all_results_df["employee_deferral_rate"] > 0
    )

    # Row-level ER and total compensation
    all_results_df["total_er_contribution"] = (
        all_results_df["employer_match_contribution"]
        + all_results_df["employer_core_contribution"]
    )
    all_results_df["total_compensation"] = (
        all_results_df["employee_gross_compensation"]
        + all_results_df["employee_contribution"]
        + all_results_df["total_er_contribution"]
    )

    # --- Group and Aggregate ---
    summary_metrics = (
        all_results_df.groupby("simulation_year")
        .agg(
            active_headcount=("employee_id", "size"),
            eligible_count=("is_eligible", "sum"),
            participant_count=("is_participant", "sum"),
            total_employee_gross_compensation=("employee_gross_compensation", "sum"),
            total_ee_contribution=("employee_contribution", "sum"),
            total_er_contribution=("total_er_contribution", "sum"),
            total_compensation=("total_compensation", "sum"),
            avg_compensation_active=("employee_gross_compensation", "mean"),
            avg_age_active=("age", "mean"),
        )
        .reset_index()
    )

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
            participants_df.groupby("simulation_year")["employee_deferral_rate"]
            .mean()
            .reset_index()
        )
        avg_deferral_rate = avg_deferral_rate.rename(
            columns={"employee_deferral_rate": "avg_deferral_rate_participants"}
        )

        # Merge with summary metrics
        summary_metrics = pd.merge(
            summary_metrics, avg_deferral_rate, on="simulation_year", how="left"
        )
    else:
        summary_metrics["avg_deferral_rate_participants"] = 0

    # Round values for presentation
    rounding_dict = {
        "avg_compensation_active": 0,
        "avg_age_active": 1,
        "participation_rate": 4,
        "avg_deferral_rate_participants": 4,
        "total_ee_contribution": 0,
        "total_er_contribution": 0,
        "total_employee_gross_compensation": 0,
        "total_compensation": 0,
    }

    for col, decimals in rounding_dict.items():
        if col in summary_metrics.columns:
            summary_metrics[col] = summary_metrics[col].round(decimals)

    # Final renaming to desired output format
    final_rename = {
        "active_headcount": "Active Headcount",
        "eligible_count": "Eligible Count",
        "participant_count": "Participant Count",
        "total_employee_gross_compensation": "Total Employee Gross Compensation",
        "total_ee_contribution": "Total EE Contribution",
        "total_er_contribution": "Total ER Contribution",
        "total_compensation": "Total Compensation",
        "avg_compensation_active": "Avg Compensation (Active)",
        "avg_age_active": "Avg Age (Active)",
        "participation_rate": "Participation Rate",
        "avg_deferral_rate_participants": "Avg Deferral Rate (Participants)",
    }

    summary_metrics.rename(columns=final_rename, inplace=True)

    # Add 'Projection Year' column as first column
    summary_metrics.insert(0, "Projection Year", summary_metrics["simulation_year"])
    # Set index to simulation_year
    summary_metrics.set_index("simulation_year", inplace=True)

    logger.info("Summary metric calculation complete.")
    return summary_metrics
