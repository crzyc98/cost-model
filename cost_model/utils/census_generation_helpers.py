"""
Helper functions for census generation extracted from scripts/generate_census.py
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import truncnorm
from typing import Set, Generator  # Added Union

# Attempt to import column constants, provide fallbacks
try:
    from utils.columns import (
        EMP_SSN,
        EMP_ROLE,
        EMP_BIRTH_DATE,
        EMP_HIRE_DATE,
        EMP_TERM_DATE,
        EMP_GROSS_COMP,
        EMP_PLAN_YEAR_COMP,
        EMP_CAPPED_COMP,
        EMP_DEFERRAL_RATE,
        EMP_CONTR,
        EMPLOYER_CORE,
        EMPLOYER_MATCH,
    )
except ImportError:
    print(
        "Warning: Could not import column constants from utils. Using string literals."
    )
    EMP_SSN, EMP_ROLE, EMP_BIRTH_DATE, EMP_HIRE_DATE, EMP_TERM_DATE = (
        "employee_ssn",
        "employee_role",
        "employee_birth_date",
        "employee_hire_date",
        "employee_termination_date",
    )
    EMP_GROSS_COMP, EMP_PLAN_YEAR_COMP, EMP_CAPPED_COMP = (
        "employee_gross_compensation",
        "employee_plan_year_compensation",
        "employee_capped_compensation",
    )
    EMP_DEFERRAL_RATE, EMP_CONTR = "employee_deferral_rate", "employee_contribution"
    EMPLOYER_CORE, EMPLOYER_MATCH = (
        "employer_core_contribution",
        "employer_match_contribution",
    )


logger = logging.getLogger(__name__)


# --- Generate single employee record ---
def generate_employee_record(
    year: int,
    plan_end_date: datetime,
    is_new_hire: bool,
    existing_ssns: Set[str],
    min_working_age: int,
    max_working_age: int,
    age_mean: float,
    age_std_dev: float,
    tenure_mean_years: float,
    tenure_std_dev_years: float,
    max_tenure_years: float,
    role_distribution: dict,
    role_compensation_params: dict,
    deferral_distribution: dict,
    unique_id: int,
    rng: Generator,
) -> dict:
    """Generates a dictionary representing a single employee."""
    record = {}
    roles = list(role_distribution.keys())
    role_probs = list(role_distribution.values())
    # Ensure keys are floats for deferral rates before using
    deferral_rates = [float(k) for k in deferral_distribution.keys()]
    deferral_probs = list(deferral_distribution.values())

    # Assign Role
    assigned_role = rng.choice(roles, p=role_probs)
    record[EMP_ROLE] = assigned_role
    # Get role params safely, provide empty dict if role missing
    role_params = role_compensation_params.get(assigned_role, {})

    # Generate SSN (ensure uniqueness)
    prefix = "NH" if is_new_hire else "EX"  # EX for initial population
    ssn = f"DUMMY_{prefix}_{rng.integers(100000, 999999)}_{unique_id:06d}"
    retry_count = 0
    while ssn in existing_ssns and retry_count < 100:
        ssn = f"DUMMY_{prefix}_{rng.integers(100000, 999999)}_{unique_id:06d}_{rng.integers(100)}"
        retry_count += 1
    if ssn in existing_ssns:  # Extremely unlikely after retries
        logger.error(
            f"Could not generate unique SSN after {retry_count} retries. ID: {unique_id}"
        )
        raise ValueError(f"Could not generate unique SSN after {retry_count} retries.")
    existing_ssns.add(ssn)
    record[EMP_SSN] = ssn  # Use the SSN constant

    # Generate Age/Birth Date
    safe_age_std_dev = max(age_std_dev, 1e-6)
    age_a, age_b = (min_working_age - age_mean) / safe_age_std_dev, (
        max_working_age - age_mean
    ) / safe_age_std_dev
    age = (
        truncnorm.rvs(
            age_a, age_b, loc=age_mean, scale=safe_age_std_dev, random_state=rng
        )
        if age_std_dev > 0
        else age_mean
    )
    age = np.clip(int(round(age)), min_working_age, max_working_age)
    # Store age calculated as of plan_end_date (used for comp calc below)
    record["internal_age_at_year_end"] = age

    birth_year = plan_end_date.year - age
    birth_month = rng.integers(1, 13)
    birth_day = rng.integers(1, 29)  # Avoid issues with Feb 30/31 etc.
    try:
        record[EMP_BIRTH_DATE] = datetime(birth_year, birth_month, birth_day)
    except ValueError:
        logger.warning(
            f"Generated invalid birth date {birth_year}-{birth_month}-{birth_day}, using Jan 1."
        )
        record[EMP_BIRTH_DATE] = datetime(birth_year, 1, 1)  # Fallback

    # Generate Hire Date and Tenure
    if is_new_hire:
        # Hire date within the current year
        plan_start_date = datetime(year, 1, 1)
        # Ensure hire date is after birth date + min working age
        min_possible_hire_date = record[EMP_BIRTH_DATE] + timedelta(
            days=365.25 * min_working_age
        )
        hire_start_bound = max(plan_start_date, min_possible_hire_date)
        # Ensure hire date is not after plan end date
        if hire_start_bound > plan_end_date:
            logger.warning(
                f"Min possible hire date {hire_start_bound.date()} is after plan end {plan_end_date.date()} for generated age {age}. Setting hire date to plan end date."
            )
            hire_date = plan_end_date
        else:
            days_in_hire_period = max(0, (plan_end_date - hire_start_bound).days) + 1
            hire_offset = rng.integers(0, days_in_hire_period)
            hire_date = hire_start_bound + timedelta(
                days=int(hire_offset)
            )  # Convert hire_offset to int

        record[EMP_HIRE_DATE] = hire_date
        # Calculate tenure based on plan_end_date for consistency
        tenure_years = max(0, (plan_end_date - hire_date).days / 365.25)
    else:  # Existing employee (from initial generation or previous year)
        # Generate tenure first, then calculate hire date
        safe_tenure_std_dev = max(tenure_std_dev_years, 1e-6)
        # Ensure tenure doesn't imply hiring before min working age or birth date
        max_possible_tenure = age - min_working_age
        effective_max_tenure = min(max_tenure_years, max_possible_tenure)

        if effective_max_tenure <= 0:
            tenure_years = 0
        else:
            # Generate tenure using truncated normal distribution
            tenure_a, tenure_b = (0 - tenure_mean_years) / safe_tenure_std_dev, (
                effective_max_tenure - tenure_mean_years
            ) / safe_tenure_std_dev
            tenure_years = (
                truncnorm.rvs(
                    tenure_a,
                    tenure_b,
                    loc=tenure_mean_years,
                    scale=safe_tenure_std_dev,
                    random_state=rng,
                )
                if tenure_std_dev_years > 0
                else tenure_mean_years
            )
            tenure_years = np.clip(tenure_years, 0, effective_max_tenure)

        # Calculate hire date based on tenure relative to plan_end_date
        hire_date = plan_end_date - timedelta(days=tenure_years * 365.25)
        # Ensure hire date is not before birth date + min working age
        min_possible_hire_date = record[EMP_BIRTH_DATE] + timedelta(
            days=365.25 * min_working_age
        )
        record[EMP_HIRE_DATE] = max(hire_date, min_possible_hire_date)

    # Store tenure calculated as of plan_end_date (used for comp calc below)
    record["internal_tenure_at_year_end"] = tenure_years

    # Calculate Compensation (consistent logic for new/existing based on current age/tenure)
    age_experience_years = max(0, record["internal_age_at_year_end"] - min_working_age)
    # Use role params with defaults from config if role-specific ones are missing
    comp_defaults = {
        "comp_base_salary": 50000,
        "comp_increase_per_age_year": 300,
        "comp_increase_per_tenure_year": 500,
        "comp_log_mean_factor": 1.0,
        "comp_spread_sigma": 0.20,
        "comp_min_salary": 28000,
    }
    current_role_params = comp_defaults.copy()
    current_role_params.update(role_params)  # Overwrite defaults with role specifics

    target_comp = (
        current_role_params["comp_base_salary"]
        + (age_experience_years * current_role_params["comp_increase_per_age_year"])
        + (
            record["internal_tenure_at_year_end"]
            * current_role_params["comp_increase_per_tenure_year"]
        )
    )

    # Use lognormal with safe sigma and base
    safe_sigma = max(current_role_params["comp_spread_sigma"], 1e-6)
    safe_base = max(1000, target_comp * current_role_params["comp_log_mean_factor"])
    log_mean = np.log(safe_base)

    comp = rng.lognormal(mean=log_mean, sigma=safe_sigma)
    record[EMP_GROSS_COMP] = round(max(current_role_params["comp_min_salary"], comp), 2)

    # Assign deferral rate
    record[EMP_DEFERRAL_RATE] = rng.choice(deferral_rates, p=deferral_probs)
    # Termination date is handled in the main simulation loop
    record[EMP_TERM_DATE] = pd.NaT

    # Remove temporary internal fields before returning
    record.pop("internal_age_at_year_end", None)
    record.pop("internal_tenure_at_year_end", None)

    return record


# --- Calculate derived fields ---
def calculate_derived_fields(
    df: pd.DataFrame,
    year: int,
    limits: dict,
    nec_rate: float,
    match_rate: float,
    match_cap_deferral_perc: float,
) -> pd.DataFrame:  # Ensure return type hint
    """Calculates plan year comp, capped comp, and contributions.
    Handles potential NaT in EMP_TERM_DATE for proration.
    """
    plan_end_date = datetime(year, 12, 31)
    plan_start_date = datetime(year, 1, 1)
    # Get limits for the year, fallback to latest known if year is missing
    limits_for_year = limits.get(year)
    if limits_for_year is None:
        latest_known_year = max(limits.keys())
        limits_for_year = limits[latest_known_year]
        logger.warning(
            f"IRS limits for year {year} not found. Using limits from {latest_known_year}."
        )

    comp_limit = limits_for_year["comp_limit"]
    deferral_limit = limits_for_year["deferral_limit"]
    catch_up_limit = limits_for_year["catch_up"]

    df_calc = df.copy()
    logger.debug(
        f"Calculating derived fields for {len(df_calc)} records for year {year}."
    )

    # Ensure date columns are datetime, coercing errors
    df_calc[EMP_HIRE_DATE] = pd.to_datetime(df_calc[EMP_HIRE_DATE], errors="coerce")
    df_calc[EMP_TERM_DATE] = pd.to_datetime(df_calc[EMP_TERM_DATE], errors="coerce")
    df_calc[EMP_BIRTH_DATE] = pd.to_datetime(df_calc[EMP_BIRTH_DATE], errors="coerce")

    # Check for missing critical dates after coercion
    if df_calc[EMP_HIRE_DATE].isnull().any():
        logger.warning("Found rows with missing or invalid Hire Date after coercion.")
    if df_calc[EMP_BIRTH_DATE].isnull().any():
        logger.warning("Found rows with missing or invalid Birth Date after coercion.")

    # Calculate Plan Year Compensation (Prorated for hire/term date)
    df_calc["calc_plan_start_date"] = plan_start_date
    df_calc["calc_plan_end_date"] = plan_end_date

    # Effective term date is term date if present, otherwise plan end date
    df_calc["calc_effective_term_date"] = df_calc[EMP_TERM_DATE].fillna(plan_end_date)

    # Determine service start/end within the plan year
    # Ensure hire date is not NaT before comparison
    df_calc["calc_service_start"] = df_calc[
        [EMP_HIRE_DATE, "calc_plan_start_date"]
    ].max(
        axis=1, skipna=False
    )  # Propagate NaT if hire date is NaT
    df_calc.loc[df_calc[EMP_HIRE_DATE].isna(), "calc_service_start"] = (
        pd.NaT
    )  # Explicitly set NaT if hire date missing

    df_calc["calc_service_end"] = df_calc[
        ["calc_effective_term_date", "calc_plan_end_date"]
    ].min(axis=1, skipna=False)

    # Calculate days worked in the plan year, handle NaT dates
    valid_service_dates = (
        df_calc["calc_service_start"].notna() & df_calc["calc_service_end"].notna()
    )
    df_calc["calc_days_in_plan_year"] = 0  # Default to 0
    df_calc.loc[valid_service_dates, "calc_days_in_plan_year"] = (
        df_calc.loc[valid_service_dates, "calc_service_end"]
        - df_calc.loc[valid_service_dates, "calc_service_start"]
    ).dt.days + 1
    df_calc["calc_days_in_plan_year"] = df_calc["calc_days_in_plan_year"].clip(
        lower=0
    )  # Ensure non-negative

    total_days_in_year = (plan_end_date - plan_start_date).days + 1
    df_calc["calc_proration_factor"] = (
        df_calc["calc_days_in_plan_year"] / total_days_in_year
    )

    # Ensure gross comp is numeric
    df_calc[EMP_GROSS_COMP] = pd.to_numeric(
        df_calc[EMP_GROSS_COMP], errors="coerce"
    ).fillna(0.0)
    df_calc[EMP_PLAN_YEAR_COMP] = (
        df_calc[EMP_GROSS_COMP] * df_calc["calc_proration_factor"]
    ).round(2)

    # Apply Compensation Limit
    df_calc[EMP_CAPPED_COMP] = df_calc[EMP_PLAN_YEAR_COMP].clip(upper=comp_limit)

    # Calculate Contributions
    # Calculate Age at year-end
    valid_birth_date_mask = df_calc[EMP_BIRTH_DATE].notna()
    df_calc["calc_age_at_year_end"] = -1  # Default age if birth date is missing
    df_calc.loc[valid_birth_date_mask, "calc_age_at_year_end"] = df_calc.loc[
        valid_birth_date_mask
    ].apply(
        lambda row: (
            plan_end_date.year
            - row[EMP_BIRTH_DATE].year
            - (
                (plan_end_date.month, plan_end_date.day)
                < (row[EMP_BIRTH_DATE].month, row[EMP_BIRTH_DATE].day)
            )
        ),
        axis=1,
    )
    df_calc["calc_age_at_year_end"] = df_calc["calc_age_at_year_end"].astype(int)

    # Deferrals
    eligible_for_catch_up = df_calc["calc_age_at_year_end"] >= 50
    max_deferral = np.where(
        eligible_for_catch_up, deferral_limit + catch_up_limit, deferral_limit
    )

    # Ensure deferral rate is numeric
    df_calc[EMP_DEFERRAL_RATE] = pd.to_numeric(
        df_calc[EMP_DEFERRAL_RATE], errors="coerce"
    ).fillna(0.0)

    # Calculate deferral based on Plan Year Comp, apply IRS limits
    df_calc["calc_potential_deferral"] = (
        df_calc[EMP_PLAN_YEAR_COMP] * (df_calc[EMP_DEFERRAL_RATE] / 100.0)
    ).round(2)
    df_calc[EMP_CONTR] = df_calc["calc_potential_deferral"].clip(
        lower=0, upper=max_deferral
    )  # Apply limits

    # NEC
    df_calc[EMPLOYER_CORE] = (
        (df_calc[EMP_PLAN_YEAR_COMP] * nec_rate).round(2).clip(lower=0)
    )  # Ensure non-negative

    # Match
    # Match based on plan year comp up to match cap percentage
    deferral_eligible_for_match = (
        df_calc[EMP_PLAN_YEAR_COMP] * match_cap_deferral_perc
    ).round(2)
    # Consider only the actual deferral amount up to the eligibility cap
    actual_deferral_for_match = df_calc[EMP_CONTR].clip(
        upper=deferral_eligible_for_match
    )
    df_calc[EMPLOYER_MATCH] = (
        (actual_deferral_for_match * match_rate).round(2).clip(lower=0)
    )  # Ensure non-negative

    # Select and order final columns
    final_cols = [
        EMP_SSN,
        EMP_ROLE,
        EMP_BIRTH_DATE,
        EMP_HIRE_DATE,
        EMP_TERM_DATE,
        EMP_GROSS_COMP,
        EMP_PLAN_YEAR_COMP,
        EMP_CAPPED_COMP,
        EMP_DEFERRAL_RATE,
        EMP_CONTR,
        EMPLOYER_CORE,
        EMPLOYER_MATCH,
        # Add any other desired output columns here
    ]
    # Return only the final columns that actually exist in the dataframe
    # Also drop temporary calculation columns
    cols_to_keep = [col for col in final_cols if col in df_calc.columns]
    return df_calc[cols_to_keep]  # *** Ensure this returns the DataFrame ***


# --- Select weighted terminations ---
def select_weighted_terminations(
    df: pd.DataFrame, num_to_terminate: int, rng: Generator
) -> pd.Index:  # Return type is index
    """Selects indices to terminate weighted by tenure and compensation."""
    eligible_df = df.copy()
    if num_to_terminate <= 0:
        return pd.Index([])  # Return empty index if none to terminate
    if num_to_terminate >= len(eligible_df):
        logger.warning(
            f"Requesting {num_to_terminate} terminations, but only {len(eligible_df)} eligible. Terminating all."
        )
        return eligible_df.index  # Return all indices

    # --- Refined Weighted Termination Logic ---
    LOW_TENURE_THRESHOLD = 2.0
    LOW_COMP_PERCENTILE = 0.25
    HIGH_RISK_WEIGHT = 10.0
    LOW_RISK_WEIGHT = 1.0

    # Tenure Risk
    if "tenure" not in eligible_df.columns:
        eligible_df["tenure"] = 0.0  # Add temp if missing
    eligible_df["low_tenure"] = (
        pd.to_numeric(eligible_df["tenure"], errors="coerce").fillna(0.0)
        < LOW_TENURE_THRESHOLD
    )

    # Compensation Risk
    eligible_df["low_comp"] = False  # Default
    comp_col = EMP_GROSS_COMP
    role_col = EMP_ROLE
    if comp_col in eligible_df.columns and pd.api.types.is_numeric_dtype(
        eligible_df[comp_col]
    ):
        if role_col in eligible_df.columns and eligible_df[role_col].nunique() > 1:
            try:
                eligible_df["comp_percentile_rank"] = eligible_df.groupby(role_col)[
                    comp_col
                ].rank(pct=True, method="average")
                eligible_df["low_comp"] = (
                    eligible_df["comp_percentile_rank"] < LOW_COMP_PERCENTILE
                )
                logger.debug("Calculated comp risk by role.")
            except Exception as e:
                logger.warning(
                    f"Error calculating comp percentile by role: {e}. Falling back to global."
                )
                eligible_df["comp_percentile_rank"] = eligible_df[comp_col].rank(
                    pct=True, method="average"
                )
                eligible_df["low_comp"] = (
                    eligible_df["comp_percentile_rank"] < LOW_COMP_PERCENTILE
                )
        else:  # Fallback or initial calc
            eligible_df["comp_percentile_rank"] = eligible_df[comp_col].rank(
                pct=True, method="average"
            )
            eligible_df["low_comp"] = (
                eligible_df["comp_percentile_rank"] < LOW_COMP_PERCENTILE
            )
            logger.debug("Calculated comp risk globally.")
    else:
        logger.warning(
            f"Column '{comp_col}' not found or not numeric. Cannot calculate comp risk."
        )

    # Assign Weights
    eligible_df["is_at_risk"] = eligible_df["low_tenure"] & eligible_df["low_comp"]
    eligible_df["term_weight"] = np.where(
        eligible_df["is_at_risk"], HIGH_RISK_WEIGHT, LOW_RISK_WEIGHT
    )
    at_risk_count = eligible_df["is_at_risk"].sum()
    logger.info(
        "Identified %d employees as 'at-risk' (low tenure & low comp).", at_risk_count
    )

    # Calculate Probabilities
    total_weight = eligible_df["term_weight"].sum()
    probabilities = None
    if total_weight > 0 and np.isfinite(total_weight):
        probabilities = eligible_df["term_weight"] / total_weight
        if np.any(~np.isfinite(probabilities)) or np.isclose(probabilities.sum(), 0):
            probabilities = None  # Invalid probabilities
        elif not np.isclose(probabilities.sum(), 1.0):
            logger.warning(
                "Termination probabilities sum to %.4f. Renormalizing.",
                probabilities.sum(),
            )
            probabilities = probabilities / probabilities.sum()  # Renormalize

    if probabilities is None:
        logger.warning("Using uniform termination probability due to invalid weights.")

    # Perform Weighted Choice
    try:
        indices_to_terminate = rng.choice(
            eligible_df.index,
            size=num_to_terminate,
            replace=False,
            p=probabilities,  # None defaults to uniform
        )
        return indices_to_terminate
    except Exception as e:  # Broader exception catch for choice issues
        logger.warning(
            f"Error during weighted choice: {e}. Falling back to simple random selection."
        )
        # Fallback to simple random sampling without weights
        return rng.choice(eligible_df.index, size=num_to_terminate, replace=False)
