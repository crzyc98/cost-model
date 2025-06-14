# cost_model/engines/run_one_year/finalize.py
"""
Finalization module for run_one_year package.

Handles final event collection, new hire terminations, and snapshot finalization.
"""
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from cost_model.state.schema import (
    ACTIVE_STATUS,
    AE_OPT_OUT_DATE,
    AE_OPTED_OUT,
    AE_WINDOW_END,
    AE_WINDOW_START,
    AI_ENROLLED,
    AI_OPTED_OUT,
    AUTO_ENROLLED,
    AUTO_REENROLLED,
    AVG_DEFERRAL_PART,
    AVG_DEFERRAL_TOTAL,
    BECAME_ELIGIBLE_DURING_YEAR,
    CFG,
    COLA_PCT,
    COMP_RAISE_PCT,
    ELIGIBILITY_ENTRY_DATE,
    EMP_ACTIVE,
    EMP_BIRTH_DATE,
    EMP_CAPPED_COMP,
    EMP_CONTR,
    EMP_DEFERRAL_RATE,
    EMP_EXITED,
    EMP_GROSS_COMP,
    EMP_HIRE_DATE,
    EMP_ID,
    EMP_LEVEL,
    EMP_LEVEL_SOURCE,
    EMP_PLAN_YEAR_COMP,
    EMP_SSN,
    EMP_TENURE,
    EMP_TENURE_BAND,
    EMP_TERM_DATE,
    EMPLOYER_CORE,
    EMPLOYER_MATCH,
    ENROLLMENT_DATE,
    ENROLLMENT_METHOD,
    EVENT_COLS,
    EVT_COLA,
    EVT_COMP,
    EVT_CONTRIB,
    EVT_CONTRIB_INCR,
    EVT_HIRE,
    EVT_NEW_HIRE_TERM,
    EVT_PROMOTION,
    EVT_RAISE,
    EVT_TERM,
    FIRST_CONTRIBUTION_DATE,
    HOURS_WORKED,
    INACTIVE_STATUS,
    IS_ELIGIBLE,
    IS_PARTICIPATING,
    NEW_HIRE_TERMINATION_RATE,
    PCT_EMP_COST_CAP,
    PCT_EMP_COST_PLAN,
    PROACTIVE_ENROLLED,
    RATE_PARTICIP_ELIG,
    RATE_PARTICIP_TOTAL,
    SIMULATION_YEAR,
    SNAPSHOT_COLS,
    SNAPSHOT_DTYPES,
    STATUS_COL,
    SUM_CAP_COMP,
    SUM_CONTRIB,
    SUM_ELIGIBLE,
    SUM_EMP_CONTR,
    SUM_EMP_CORE,
    SUM_EMP_COST,
    SUM_EMP_MATCH,
    SUM_HEADCOUNT,
    SUM_PARTICIPATING,
    SUM_PLAN_COMP,
    TERM_RATE,
    WINDOW_CLOSED_DURING_YEAR,
)

from .utils import dbg


def apply_new_hire_terminations(
    snap_with_hires: pd.DataFrame,
    hazard_slice: pd.DataFrame,
    year_rng: np.random.Generator,
    year: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies terminations to newly hired employees.

    Args:
        snap_with_hires: Snapshot with new hires
        hazard_slice: Hazard table slice for the current year
        year_rng: Random number generator
        year: Current simulation year

    Returns:
        Tuple containing (termination_events_df, updated_snapshot)
    """
    logger = logging.getLogger(__name__)

    # Log the hazard_slice to see what rates we have
    logger.info(
        f"[NEW_HIRE_TERM] Year {year}: Hazard slice columns: {hazard_slice.columns.tolist()}"
    )

    # Get the new hire termination rate from hazard table
    # CRITICAL FIX: Use NEW_HIRE_TERMINATION_RATE instead of NEW_HIRE_TERM_RATE
    if NEW_HIRE_TERMINATION_RATE in hazard_slice.columns:
        nh_term_rate = hazard_slice[NEW_HIRE_TERMINATION_RATE].iloc[
            0
        ]  # Get first value if multiple
        logger.info(
            f"[NEW_HIRE_TERM] Year {year}: Found {NEW_HIRE_TERMINATION_RATE} in hazard_slice: {nh_term_rate}"
        )
    else:
        nh_term_rate = 0.0
        logger.warning(
            f"[NEW_HIRE_TERM] Year {year}: {NEW_HIRE_TERMINATION_RATE} not found in hazard_slice, using {nh_term_rate}"
        )

    # Check if we have the job_level_source column to identify new hires
    if (
        EMP_LEVEL_SOURCE in snap_with_hires.columns
        and snap_with_hires[EMP_LEVEL_SOURCE].notna().any()
    ):
        # Use job_level_source if it has non-null values
        new_hire_mask = snap_with_hires[EMP_LEVEL_SOURCE] == "hire"
        logger.info(
            f"[NEW_HIRE_TERM] Year {year}: Found {new_hire_mask.sum()} new hires via {EMP_LEVEL_SOURCE}='hire'"
        )
        if new_hire_mask.any():
            logger.info(
                f"[NEW_HIRE_TERM] Sample of {EMP_LEVEL_SOURCE} values: {snap_with_hires.loc[new_hire_mask, EMP_LEVEL_SOURCE].head(5).tolist()}"
            )
    else:
        # Fallback: if job_level_source is missing or all NaN, look for employees hired this year
        logger.warning(f"{EMP_LEVEL_SOURCE} not found or all NaN. Falling back to hire date check.")
        new_hire_mask = snap_with_hires[EMP_HIRE_DATE].dt.year == year
        logger.info(
            f"[NEW_HIRE_TERM] Year {year}: Found {new_hire_mask.sum()} new hires via {EMP_HIRE_DATE}.year == {year}"
        )
        if new_hire_mask.any():
            logger.info(
                f"[NEW_HIRE_TERM] Sample of hire dates: {snap_with_hires.loc[new_hire_mask, EMP_HIRE_DATE].dt.strftime('%Y-%m-%d').head(5).tolist()}"
            )

    # Log columns in the snapshot for debugging
    logger.debug(f"[NEW_HIRE_TERM] Snapshot columns: {snap_with_hires.columns.tolist()}")
    if EMP_LEVEL_SOURCE in snap_with_hires.columns:
        logger.debug(
            f"[NEW_HIRE_TERM] Unique {EMP_LEVEL_SOURCE} values: {snap_with_hires[EMP_LEVEL_SOURCE].unique().tolist()}"
        )

    new_hire_count = new_hire_mask.sum()

    if new_hire_count == 0:
        logger.info(f"[NEW_HIRE_TERM] Year {year}: No new hires found to terminate.")
        return pd.DataFrame(), snap_with_hires

    # Log some info about new hires
    if new_hire_count > 0:
        new_hires = snap_with_hires[new_hire_mask]
        logger.info(f"[NEW_HIRE_TERM] Year {year}: New hire IDs: {new_hires.index.tolist()}")
        if EMP_LEVEL in new_hires.columns:
            logger.info(
                f"[NEW_HIRE_TERM] Year {year}: New hire job levels: {new_hires[EMP_LEVEL].value_counts().to_dict()}"
            )
        else:
            logger.warning(
                f"[NEW_HIRE_TERM] Year {year}: '{EMP_LEVEL}' column not found in new hires data"
            )

    # Calculate how many new hires to terminate
    nh_term_count = int(round(new_hire_count * nh_term_rate))

    # Debug: Log the calculation
    logger.info(
        f"[NEW_HIRE_TERM] Year {year}: New hire count: {new_hire_count}, Term rate: {nh_term_rate}, Calculated nh_term_count: {nh_term_count}"
    )

    # If we have new hires but nh_term_count is 0 due to rounding, log a warning
    if new_hire_count > 0 and nh_term_count == 0 and nh_term_rate > 0:
        logger.warning(
            f"[NEW_HIRE_TERM] Year {year}: New hires found but nh_term_count rounded to 0. Consider increasing {NEW_HIRE_TERMINATION_RATE}."
        )

    # For debugging: If we have new hires but nh_term_count is 0, log more details
    if new_hire_count > 0 and nh_term_count == 0:
        sample_cols = [EMP_ID, EMP_HIRE_DATE, EMP_LEVEL]
        sample_data = snap_with_hires[new_hire_mask][
            [col for col in sample_cols if col in snap_with_hires.columns]
        ]
        logger.info(
            f"[NEW_HIRE_TERM] Year {year}: New hires found but nh_term_count is 0. New hires sample: {sample_data.head().to_dict('records')}"
        )

    logger.info(
        f"[NEW_HIRE_TERM] Year {year}: Terminating {nh_term_count} of {new_hire_count} new hires (rate={nh_term_rate:.4f}, rounded from {new_hire_count * nh_term_rate:.2f})"
    )

    if nh_term_count == 0:
        logger.info(f"[NEW_HIRE_TERM] Year {year}: No new hires to terminate (nh_term_count=0)")
        return pd.DataFrame(columns=EVENT_COLS), snap_with_hires

    # Get the new hire indices and randomly select nh_term_count to terminate
    new_hire_indices = snap_with_hires[new_hire_mask].index
    term_indices = year_rng.choice(new_hire_indices, size=nh_term_count, replace=False)

    logger.info(
        f"[NEW_HIRE_TERM] Year {year}: Selected new hires to terminate: {term_indices.tolist()}"
    )

    # Create termination events
    term_events = []
    for idx in term_indices:
        emp_id = snap_with_hires.loc[idx, EMP_ID]
        hire_date = snap_with_hires.loc[idx, EMP_HIRE_DATE]

        # Calculate termination date (randomly during the year)
        days_into_year = year_rng.integers(1, 365)
        term_date = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=days_into_year)

        # Calculate tenure in days
        tenure_days = (term_date - hire_date).days

        # Update snapshot with termination
        snap_with_hires.loc[idx, EMP_TERM_DATE] = term_date
        snap_with_hires.loc[idx, EMP_ACTIVE] = False

        # Get job level and compensation if available, with proper NA handling
        job_level = None
        if EMP_LEVEL in snap_with_hires.columns and not pd.isna(
            snap_with_hires.loc[idx, EMP_LEVEL]
        ):
            try:
                job_level = int(snap_with_hires.loc[idx, EMP_LEVEL])
            except (ValueError, TypeError):
                job_level = None

        job_family = (
            snap_with_hires.loc[idx, "job_family"]
            if "job_family" in snap_with_hires.columns
            and not pd.isna(snap_with_hires.loc[idx, "job_family"])
            else None
        )

        compensation = None
        if EMP_GROSS_COMP in snap_with_hires.columns and not pd.isna(
            snap_with_hires.loc[idx, EMP_GROSS_COMP]
        ):
            try:
                compensation = float(snap_with_hires.loc[idx, EMP_GROSS_COMP])
            except (ValueError, TypeError):
                compensation = None

        # Employee role is no longer tracked - removed as part of schema refactoring

        # Safely get values with NA handling
        def safe_get_float(col, default=None):
            if col not in snap_with_hires.columns or pd.isna(snap_with_hires.loc[idx, col]):
                return default
            try:
                return float(snap_with_hires.loc[idx, col])
            except (ValueError, TypeError):
                return default

        def safe_get_value(col, default=None):
            if col not in snap_with_hires.columns or pd.isna(snap_with_hires.loc[idx, col]):
                return default
            try:
                value = snap_with_hires.loc[idx, col]
                return float(value) if isinstance(value, (int, float, np.number)) else value
            except (ValueError, TypeError):
                return value if not pd.isna(value) else default

        # Get values with proper NA handling
        comp_raise_pct = (
            safe_get_float(COMP_RAISE_PCT) if COMP_RAISE_PCT in snap_with_hires.columns else None
        )
        deferral_rate = (
            safe_get_float(EMP_DEFERRAL_RATE)
            if EMP_DEFERRAL_RATE in snap_with_hires.columns
            else None
        )
        tenure_band = (
            safe_get_value(EMP_TENURE_BAND)
            if (EMP_TENURE_BAND in snap_with_hires.columns)
            else None
        )

        # Create termination event with detailed metadata
        event = {
            "event_id": f"NH_TERM_{year}_{emp_id}_{term_date.strftime('%Y%m%d')}",
            "event_time": term_date,
            "employee_id": emp_id,
            "event_type": EVT_NEW_HIRE_TERM,
            "value_num": tenure_days,  # Store tenure_days as a numeric value for easier querying
            "value_json": json.dumps(
                {
                    "reason": "new_hire_termination",
                    "tenure_days": tenure_days,
                    "hire_date": hire_date.strftime("%Y-%m-%d") if not pd.isna(hire_date) else None,
                    "term_date": term_date.strftime("%Y-%m-%d") if not pd.isna(term_date) else None,
                    "job_level": job_level,
                    "job_family": job_family,
                    "gross_compensation": compensation,
                    "comp_raise_pct": comp_raise_pct,
                    "deferral_rate": deferral_rate,
                    "tenure_band": tenure_band,
                },
                default=str,
            ),
            "meta": f"New-hire termination for {emp_id} after {tenure_days} days in {year}",
            "simulation_year": year,
        }
        term_events.append(event)

        logger.debug(
            f"[NEW_HIRE_TERM] Year {year}: New hire termination: {emp_id} on {term_date.date()} (tenure: {tenure_days} days)"
        )

    # Create events dataframe with consistent columns
    if term_events:
        # Create DataFrame and ensure all columns exist
        term_events_df = pd.DataFrame(term_events)
        for col in EVENT_COLS:
            if col not in term_events_df.columns:
                term_events_df[col] = None

        # Reorder columns
        term_events_df = term_events_df[EVENT_COLS]
        logger.info(
            f"[NEW_HIRE_TERM] Year {year}: Created {len(term_events_df)} new-hire termination events"
        )
    else:
        term_events_df = pd.DataFrame(columns=EVENT_COLS)
        logger.info(f"[NEW_HIRE_TERM] Year {year}: No new-hire termination events created")

    return term_events_df, snap_with_hires


def build_full_event_log(
    plan_rule_events: pd.DataFrame,
    comp_term_events: pd.DataFrame,
    hires_events: pd.DataFrame,
    new_hire_term_events: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """
    Combines all event types into a complete event log for the year.

    Args:
        plan_rule_events: Plan rule events
        comp_term_events: Compensation and termination events
        hires_events: Hiring events
        new_hire_term_events: New hire termination events
        year: Current simulation year

    Returns:
        DataFrame containing all events for the year with consistent columns
    """
    logger = logging.getLogger(__name__)

    # Initialize an empty DataFrame with the expected columns
    full_event_log = pd.DataFrame(columns=EVENT_COLS)

    # List of all event dataframes to combine
    all_events = []

    # Helper function to ensure consistent columns in event dataframes
    def prepare_events_df(df: pd.DataFrame, event_type: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=EVENT_COLS)

        # Make a copy to avoid modifying the original
        df = df.copy()

        # Ensure all required columns exist
        for col in EVENT_COLS:
            if col not in df.columns:
                df[col] = None

        # Ensure event_type is set if not already
        if "event_type" in df.columns:
            df["event_type"] = df["event_type"].fillna(event_type)
        else:
            df["event_type"] = event_type

        # Ensure simulation_year is set
        if "simulation_year" in df.columns:
            df["simulation_year"] = df["simulation_year"].fillna(year)
        else:
            df["simulation_year"] = year

        # Convert event_time to datetime if it exists
        if "event_time" in df.columns and pd.api.types.is_datetime64_any_dtype(df["event_time"]):
            df["event_time"] = pd.to_datetime(df["event_time"])

        # Ensure value_num is numeric
        if "value_num" in df.columns:
            df["value_num"] = pd.to_numeric(df["value_num"], errors="coerce")

        # Ensure value_json is a string
        if "value_json" in df.columns:
            df["value_json"] = df["value_json"].astype(str)

        # Ensure meta is a string
        if "meta" in df.columns:
            df["meta"] = df["meta"].astype(str)

        # Select only the columns we want, in the right order
        return df[EVENT_COLS]

    # Prepare each event type
    if plan_rule_events is not None and not plan_rule_events.empty:
        all_events.append(prepare_events_df(plan_rule_events, EVT_CONTRIB))

    if comp_term_events is not None and not comp_term_events.empty:
        all_events.append(prepare_events_df(comp_term_events, EVT_COMP))

    if hires_events is not None and not hires_events.empty:
        all_events.append(prepare_events_df(hires_events, EVT_HIRE))

    if new_hire_term_events is not None and not new_hire_term_events.empty:
        all_events.append(prepare_events_df(new_hire_term_events, EVT_TERM))

    # Filter out any empty dataframes
    all_events = [df for df in all_events if not df.empty]

    # If no events, return empty DataFrame with expected columns
    if not all_events:
        return pd.DataFrame(columns=EVENT_COLS)

    # Combine all events
    full_event_log = pd.concat(all_events, ignore_index=True)

    # Sort by event_time if available
    if "event_time" in full_event_log.columns:
        full_event_log = full_event_log.sort_values("event_time")

    # Ensure we only return the columns we want, in the right order
    return full_event_log[EVENT_COLS]


def finalize_snapshot(
    snapshot: pd.DataFrame, year: int, config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Applies final transformations to the snapshot before saving.

    Args:
        snap_df: The snapshot DataFrame
        year: Current simulation year
        config: Configuration dictionary

    Returns:
        Finalized snapshot DataFrame with standardized column names and types
    """
    logger = logging.getLogger(__name__)

    if snapshot.empty:
        logger.warning(f"[YR={year}] Empty snapshot, nothing to finalize")
        return pd.DataFrame(columns=SNAPSHOT_COLS)

    # Make a copy to avoid modifying the original
    snap_df = snapshot.copy()

    # Ensure all required columns exist with appropriate defaults
    for col in SNAPSHOT_COLS:
        if col not in snap_df.columns:
            # Set default values based on column type
            if col in SNAPSHOT_DTYPES:
                if SNAPSHOT_DTYPES[col] == "datetime64[ns]":
                    snap_df[col] = pd.NaT
                elif SNAPSHOT_DTYPES[col] == "float64":
                    snap_df[col] = 0.0
                elif SNAPSHOT_DTYPES[col] == "int64":
                    snap_df[col] = 0
                elif SNAPSHOT_DTYPES[col] == "bool":
                    snap_df[col] = False
                else:
                    snap_df[col] = None
            else:
                snap_df[col] = None

    # Convert columns to proper data types with robust NA handling
    for col, dtype in SNAPSHOT_DTYPES.items():
        if col not in snap_df.columns:
            continue

        try:
            if dtype == "datetime64[ns]":
                # For datetime, convert to datetime and handle NAs
                if snap_df[col].isna().all():
                    snap_df[col] = pd.NaT
                else:
                    snap_df[col] = pd.to_datetime(snap_df[col], errors="coerce")

            elif dtype == "float64":
                # For float, convert to numeric and fill NAs with 0.0
                if snap_df[col].isna().all():
                    snap_df[col] = 0.0
                else:
                    # First convert to string to handle any non-numeric values
                    str_series = snap_df[col].astype(str)
                    # Replace empty strings and 'nan' with '0.0'
                    str_series = str_series.replace(["", "nan", "None", "N/A", "NA", "NaT"], "0.0")
                    # Convert to numeric
                    snap_df[col] = pd.to_numeric(str_series, errors="coerce").fillna(0.0)

            elif dtype == "int64":
                # For int, convert to numeric, fill NAs with 0, then convert to int
                if snap_df[col].isna().all():
                    snap_df[col] = 0
                else:
                    # First convert to string to handle any non-numeric values
                    str_series = snap_df[col].astype(str)
                    # Replace empty strings and 'nan' with '0'
                    str_series = str_series.replace(["", "nan", "None", "N/A", "NA", "NaT"], "0")
                    # Convert to numeric and then to int - use infer_objects before fillna to avoid downcasting warning
                    numeric_series = pd.to_numeric(str_series, errors="coerce")
                    inferred_series = numeric_series.infer_objects(copy=False)
                    snap_df[col] = inferred_series.fillna(0).astype("int64")

            elif dtype == "bool":
                # For boolean, convert to boolean and fill NAs with False
                if snap_df[col].isna().all():
                    snap_df[col] = False
                else:
                    # Convert to string and handle common boolean representations
                    str_series = snap_df[col].astype(str).str.lower().str.strip()
                    # Map common true/false strings to boolean
                    true_values = ["true", "t", "yes", "y", "1", "1.0"]
                    false_values = ["false", "f", "no", "n", "0", "0.0", ""]

                    mask_true = str_series.isin(true_values)
                    mask_false = str_series.isin(false_values) | str_series.isna()

                    snap_df[col] = False  # Default to False
                    snap_df.loc[mask_true, col] = True

            else:
                # For other types, convert to string and handle NAs
                if snap_df[col].isna().all():
                    snap_df[col] = None
                else:
                    snap_df[col] = snap_df[col].astype("object")

        except Exception as e:
            logger.warning(f"Error converting column '{col}' to {dtype}: {str(e)}")
            # Set default value based on type
            if dtype == "datetime64[ns]":
                snap_df[col] = pd.NaT
            elif dtype == "float64":
                snap_df[col] = 0.0
            elif dtype == "int64":
                snap_df[col] = 0
            elif dtype == "bool":
                snap_df[col] = False
            else:
                snap_df[col] = None

    # Ensure required columns have valid values
    if EMP_ID in snap_df.columns:
        snap_df[EMP_ID] = snap_df[EMP_ID].fillna("")

    # Ensure active status is consistent with termination date
    if EMP_ACTIVE in snap_df.columns and EMP_TERM_DATE in snap_df.columns:
        snap_df[EMP_ACTIVE] = snap_df[EMP_TERM_DATE].isna()

    # Ensure simulation year is set
    snap_df[SIMULATION_YEAR] = year

    # Select only the columns we want, in the right order
    result_cols = [col for col in SNAPSHOT_COLS if col in snap_df.columns]
    snap_df = snap_df[result_cols].copy()

    # Ensure proper column order
    final_cols = [col for col in SNAPSHOT_COLS if col in snap_df.columns]
    snap_df = snap_df[final_cols]

    logger.info(f"[YR={year}] Finalized snapshot with {len(snap_df)} employees")
    return snap_df
