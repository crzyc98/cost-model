"""
cost_model/rules/contributions.py

Calculates employee and employer contributions for the simulation year.
"""

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

from cost_model.rules.validators import ContributionsRule, MatchRule, NonElectiveRule
from cost_model.state.schema import (
    EMP_ID,
    EMP_GROSS_COMP,
    EMP_DEFERRAL_RATE,
    EMP_CONTR,
    EMPLOYER_MATCH,
    EMPLOYER_CORE,
    EMP_CAPPED_COMP,
    EMP_PLAN_YEAR_COMP,
    EMP_HIRE_DATE,
    EMP_TERM_DATE,
    EMP_STATUS_EOY,
    EMP_ACTIVE,
    EMP_BIRTH_DATE,
    ACTIVE_STATUS,
    to_nullable_bool,
)
from cost_model.utils.date_utils import calculate_age
from cost_model.utils.constants import ACTIVE_STATUSES

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def apply(
    df: pd.DataFrame,
    contrib_rules: ContributionsRule,
    match_rules: MatchRule,
    nec_rules: NonElectiveRule,
    irs_limits: Dict[int, Dict[str, float]],
    simulation_year: int,
    year_start: pd.Timestamp,
    year_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Vectorized contributions calculation:
      1. Prorate compensation for terminations
      2. Calculate plan-year & capped compensation
      3. Determine catch-up eligibility and limits
      4. Compute employee pre-tax, NEC, match, and total contributions
    """
    logger.info(f"=== APPLY_CONTRIBUTIONS DEBUG START for {simulation_year} ===")
    logger.info(f"Input DataFrame shape: {df.shape}")
    logger.info(f"Input DataFrame columns: {df.columns.tolist()}")
    logger.info(f"Input DataFrame head (first 3 rows):")
    logger.info(f"{df.head(3).to_string()}")

    # Log key input columns
    key_input_cols = [EMP_GROSS_COMP, EMP_DEFERRAL_RATE, EMP_STATUS_EOY, EMP_ACTIVE, EMP_BIRTH_DATE]
    for col in key_input_cols:
        if col in df.columns:
            logger.info(f"Column '{col}' - sample values: {df[col].head().tolist()}")
            logger.info(f"Column '{col}' - null count: {df[col].isnull().sum()}")
            logger.info(f"Column '{col}' - dtype: {df[col].dtype}")
        else:
            logger.warning(f"Expected input column '{col}' NOT FOUND")

    # Log rule objects
    logger.info(f"contrib_rules: {contrib_rules}")
    logger.info(f"match_rules: {match_rules}")
    logger.info(f"nec_rules: {nec_rules}")
    logger.info(f"irs_limits keys: {list(irs_limits.keys()) if irs_limits else 'None'}")

    # 0) Find and resolve any duplicate column names up front
    dupes = df.columns[df.columns.duplicated()].unique().tolist()
    if dupes:
        logger.warning("Duplicate columns detected: %r", dupes)
        df = df.loc[:, ~df.columns.duplicated(keep="last")]
        logger.info("Columns after deduplication: %r", df.columns.tolist())

    # 1) Check for duplicate indices
    dupes = df.index.duplicated()
    if dupes.any():
        logger.error(
            "Duplicate indices found in input DataFrame: %d duplicates", dupes.sum()
        )
        logger.debug("Duplicate index values: %s", df.index[dupes].unique())

    logger.info(f"Calculating contributions for {simulation_year}")

    # --- Extract IRS limits for the year ---
    year_limits_obj = irs_limits.get(
        simulation_year
    )  # Get the IRSYearLimits object or None

    if year_limits_obj:  # Check if limits exist for the year
        comp_limit = getattr(year_limits_obj, "compensation_limit", 345000)
        def_limit = getattr(year_limits_obj, "deferral_limit", 23000)
        catch_limit = getattr(year_limits_obj, "catchup_limit", 7500)
        catch_age = getattr(year_limits_obj, "catchup_eligibility_age", 50)
        overall_limit = getattr(
            year_limits_obj, "overall_limit", None
        )  # Use getattr with default None
        logger.debug(
            f"Using IRS Limits for {simulation_year}: Comp={comp_limit}, Deferral={def_limit}, Catchup={catch_limit}@{catch_age}, Overall={overall_limit}"
        )
    else:
        logger.warning(
            f"IRS limits for {simulation_year} not found in configuration. Using fallback defaults."
        )
        # Fallback defaults if year's limits are missing
        comp_limit = 345000
        def_limit = 23000
        catch_limit = 7500
        catch_age = 50
        overall_limit = None  # Default overall limit if not specified

    # --- 1) Initialize output columns with defaults ---
    defaults: Dict[str, Any] = {
        EMP_PLAN_YEAR_COMP: 0.0,
        EMP_CAPPED_COMP: 0.0,
        EMP_CONTR: 0.0,
        EMPLOYER_MATCH: 0.0,
        EMPLOYER_CORE: 0.0,
        "total_contributions": 0.0,
        "is_catch_up_eligible": False,
        "effective_deferral_limit": 0.0,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            dtype = "boolean" if isinstance(default, bool) else float
            df.loc[:, col] = pd.Series(default, index=df.index, dtype=dtype)
        # ensure boolean defaults are proper nullable bool
        if isinstance(default, bool):
            df.loc[:, col] = to_nullable_bool(df[col])

    # --- 2) Defensive numeric coercion for key contribution columns ---
    for col in [
        EMP_DEFERRAL_RATE,
        EMP_CONTR,
        EMPLOYER_MATCH,
        EMPLOYER_CORE,
        EMP_PLAN_YEAR_COMP,
        EMP_CAPPED_COMP,
        "total_contributions",
        "effective_deferral_limit",
    ]:
        val = df.get(col, None)
        if isinstance(val, pd.Series):
            df.loc[:, col] = pd.to_numeric(val, errors="coerce").fillna(0.0)
        elif isinstance(val, (list, tuple, np.ndarray)):
            df.loc[:, col] = (
                pd.Series(val, index=df.index)
                .pipe(pd.to_numeric, errors="coerce")
                .fillna(0.0)
            )
        else:
            df.loc[:, col] = 0.0

    # --- 3) Proration of compensation ---
    total_days_in_year = (
        pd.to_datetime(year_end) - pd.to_datetime(year_start)
    ).days + 1

    # Ensure date columns are in datetime format
    hire_date_col = pd.to_datetime(df.get(EMP_HIRE_DATE), errors="coerce")
    term_date_col = pd.to_datetime(df.get(EMP_TERM_DATE), errors="coerce")

    # Determine the actual start and end of work within the simulation year for each employee
    # Effective start date for the year is the later of year_start or hire_date
    effective_start_date_in_year = pd.Series(year_start, index=df.index)
    if EMP_HIRE_DATE in df.columns and hire_date_col.notna().any():
        effective_start_date_in_year = np.maximum(
            effective_start_date_in_year, hire_date_col.fillna(pd.Timestamp.min)
        )

    # Effective end date for the year is the earlier of year_end or term_date
    effective_end_date_in_year = pd.Series(year_end, index=df.index)
    if EMP_TERM_DATE in df.columns and term_date_col.notna().any():
        effective_end_date_in_year = np.minimum(
            effective_end_date_in_year, term_date_col.fillna(pd.Timestamp.max)
        )

    # Calculate days_worked based on the effective employment period within the year
    df["days_worked"] = (
        effective_end_date_in_year - effective_start_date_in_year
    ).dt.days + 1

    # If effective_end_date_in_year is before effective_start_date_in_year, it means no overlap, so days_worked is 0.
    # This also handles cases where term_date is before year_start, or hire_date is after year_end.
    no_overlap_mask = effective_end_date_in_year < effective_start_date_in_year
    df.loc[no_overlap_mask, "days_worked"] = 0

    # Clip days_worked to be within [0, total_days_in_year]
    df.loc[:, "days_worked"] = df["days_worked"].clip(lower=0, upper=total_days_in_year)

    # If original termination date was present but unparseable (became NaT),
    # these individuals should not be considered as having worked correctly.
    if EMP_TERM_DATE in df.columns:  # only if original column exists
        original_term_dates_series = df[EMP_TERM_DATE]
        # Check for non-None/non-NaN in original, and NaT in coerced
        unparseable_term_date_mask = (
            original_term_dates_series.notna() & term_date_col.isna()
        )
        if unparseable_term_date_mask.any():
            df.loc[unparseable_term_date_mask, "days_worked"] = 0
            logger.warning(
                f"Found {unparseable_term_date_mask.sum()} records with unparseable original termination dates "
                f"(e.g., became NaT after coercion). Their 'days_worked' has been set to 0."
            )

    # If original hire date was present but unparseable (became NaT),
    # this could also lead to incorrect days_worked (e.g., treating them as starting Jan 1st by default).
    # Setting days_worked to 0 is a safe default if hire date is essential and unparseable.
    if EMP_HIRE_DATE in df.columns:  # only if original column exists
        original_hire_dates_series = df[EMP_HIRE_DATE]
        # Check for non-None/non-NaN in original, and NaT in coerced
        unparseable_hire_date_mask = (
            original_hire_dates_series.notna() & hire_date_col.isna()
        )
        if unparseable_hire_date_mask.any():
            df.loc[unparseable_hire_date_mask, "days_worked"] = 0
            logger.warning(
                f"Found {unparseable_hire_date_mask.sum()} records with unparseable original hire dates "
                f"(e.g., became NaT after coercion). Their 'days_worked' has been set to 0."
            )

    df.loc[:, "proration"] = 0.0  # Initialize proration column
    # Avoid division by zero if total_days_in_year is somehow zero (should not happen)
    if total_days_in_year > 0:
        df.loc[:, "proration"] = df["days_worked"] / total_days_in_year

    # Ensure EMP_GROSS_COMP is numeric before multiplication
    current_gross_comp = pd.to_numeric(df.get(EMP_GROSS_COMP), errors="coerce").fillna(
        0.0
    )
    df.loc[:, EMP_PLAN_YEAR_COMP] = current_gross_comp * df["proration"]

    # Calculate capped compensation based on prorated comp_limit
    # comp_limit is already defined earlier in the function
    prorated_comp_limit = comp_limit * df["proration"]
    df.loc[:, EMP_CAPPED_COMP] = np.minimum(df[EMP_PLAN_YEAR_COMP], prorated_comp_limit)

    # --- 4) Catch-up eligibility & effective limits ---
    df.loc[:, "current_age"] = calculate_age(df.get(EMP_BIRTH_DATE), year_end)

    # Determine active status from available columns
    # Priority: EMP_STATUS_EOY > EMP_ACTIVE > legacy 'status' > fallback to all active
    if EMP_STATUS_EOY in df.columns:
        # Use EMP_STATUS_EOY if available (uses schema values: 'Active', 'Terminated', 'Inactive')
        active_mask = df[EMP_STATUS_EOY] == ACTIVE_STATUS  # ACTIVE_STATUS = "Active"
    elif EMP_ACTIVE in df.columns:
        # Use EMP_ACTIVE boolean column if available
        active_mask = df[EMP_ACTIVE].fillna(False)
    elif "status" in df.columns:
        # Fallback to legacy 'status' column if it exists (uses enum values)
        active_mask = df["status"].isin(ACTIVE_STATUSES)
    else:
        # Last resort: assume all employees are active and log warning
        logger.warning("No status column found (EMP_STATUS_EOY, EMP_ACTIVE, or 'status'). Assuming all employees are active.")
        active_mask = pd.Series(True, index=df.index)

    catch_mask = active_mask & (df["current_age"] >= catch_age)

    df.loc[catch_mask, "is_catch_up_eligible"] = True
    df.loc[:, "effective_deferral_limit"] = def_limit
    df.loc[catch_mask, "effective_deferral_limit"] += catch_limit

    # --- 5) Employee pre-tax contributions ---
    potential = df[EMP_CAPPED_COMP] * df[EMP_DEFERRAL_RATE]
    df.loc[:, EMP_CONTR] = np.minimum(potential, df["effective_deferral_limit"])
    logger.info(f"Employee contributions calculated - sample potential: {potential.head().tolist()}")
    logger.info(f"Employee contributions calculated - sample final: {df[EMP_CONTR].head().tolist()}")

    # --- 6) Employer non-elective contributions (NEC) ---
    df.loc[:, EMPLOYER_CORE] = df[EMP_CAPPED_COMP] * nec_rules.rate
    logger.info(f"NEC rate: {nec_rules.rate}")
    logger.info(f"Employer core contributions calculated - sample: {df[EMPLOYER_CORE].head().tolist()}")

    # --- 7) Employer match contributions ---
    df.loc[:, EMPLOYER_MATCH] = 0.0
    tiers = match_rules.tiers
    dollar_cap = match_rules.dollar_cap
    logger.info(f"Match tiers: {tiers}")
    logger.info(f"Match dollar cap: {dollar_cap}")
    if tiers:
        caps = np.array([t.cap_deferral_pct for t in tiers])
        rates = np.array([t.match_rate for t in tiers])
        prev = np.concatenate(([0.0], caps[:-1]))
        dr = df[EMP_DEFERRAL_RATE].to_numpy()[:, None]
        cc = df[EMP_CAPPED_COMP].to_numpy()[:, None]
        alloc = np.clip(np.minimum(dr, caps) - prev, 0.0, None)
        match_amt = (alloc * cc * rates).sum(axis=1)
        logger.info(f"Match calculation - sample deferral rates: {dr[:3].flatten()}")
        logger.info(f"Match calculation - sample capped comp: {cc[:3].flatten()}")
        logger.info(f"Match calculation - sample match amounts: {match_amt[:3]}")
        if dollar_cap is not None:
            match_amt = np.minimum(match_amt, dollar_cap)
        df.loc[active_mask, EMPLOYER_MATCH] = match_amt[active_mask]
        logger.info(f"Employer match contributions calculated - sample: {df[EMPLOYER_MATCH].head().tolist()}")
    else:
        logger.info("No match tiers defined - match contributions remain 0")

    # --- 8) Total contributions & overall limit check ---
    df.loc[:, "total_contributions"] = df[EMP_CONTR] + df[EMPLOYER_MATCH] + df[EMPLOYER_CORE]
    logger.debug(
        "Null values - EMP_CONTR: %d, EMPLOYER_MATCH: %d, EMPLOYER_CORE: %d",
        df[EMP_CONTR].isna().sum(),
        df[EMPLOYER_MATCH].isna().sum(),
        df[EMPLOYER_CORE].isna().sum(),
    )
    if overall_limit is not None:
        over = df["total_contributions"] > overall_limit
        if over.any():
            logger.warning(
                "%d employees exceed overall limit $%s, scaling back ER credits",
                over.sum(),
                overall_limit,
            )
            excess = df.loc[over, "total_contributions"] - overall_limit
            er_tot = df.loc[over, EMPLOYER_MATCH] + df.loc[over, EMPLOYER_CORE]
            frac = np.where(er_tot > 0, 1 - excess / er_tot, 0)
            df.loc[over, EMPLOYER_MATCH] *= frac
            df.loc[over, EMPLOYER_CORE] *= frac
            df.loc[over, "total_contributions"] = overall_limit

    # --- DETAILED OUTPUT LOGGING ---
    logger.info(f"=== APPLY_CONTRIBUTIONS FINAL RESULTS ===")
    logger.info(f"Output DataFrame shape: {df.shape}")

    # Log contribution column results
    contrib_cols = [EMP_CONTR, EMPLOYER_MATCH, EMPLOYER_CORE]
    for col in contrib_cols:
        if col in df.columns:
            logger.info(f"Column '{col}' - sample values: {df[col].head().tolist()}")
            logger.info(f"Column '{col}' - dtype: {df[col].dtype}")
            logger.info(f"Column '{col}' - null count: {df[col].isnull().sum()}")
            logger.info(f"Column '{col}' - non-zero count: {(df[col] != 0).sum()}")
            logger.info(f"Column '{col}' - min/max: {df[col].min()}/{df[col].max()}")
        else:
            logger.error(f"Expected output column '{col}' NOT FOUND")

    # Log a few sample rows for debugging
    logger.info(f"Sample output rows (first 3):")
    sample_cols = [EMP_ID, EMP_GROSS_COMP, EMP_DEFERRAL_RATE, EMP_CONTR, EMPLOYER_MATCH, EMPLOYER_CORE]
    available_cols = [col for col in sample_cols if col in df.columns]
    logger.info(f"{df[available_cols].head(3).to_string()}")

    logger.info(f"=== APPLY_CONTRIBUTIONS DEBUG END ===")

    # --- Cleanup intermediate columns ---
    df = df.drop(columns=["proration", "current_age"])
    return df
