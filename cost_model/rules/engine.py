# cost_model/rules/engine.py
"""
Engine for applying plan rules (Phase 2) to a population snapshot.
Orchestrates eligibility, auto-enrollment, auto-increase, and contributions.
"""

import logging
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

# Use relative imports for rules and validators within the rules package
try:
    # Import the rule modules directly
    from cost_model.config.models import IRSYearLimits

    from . import auto_enrollment, auto_increase, contributions, eligibility

    # Import Pydantic models for validation/structure access
    from .validators import (
        AutoEnrollmentRule,
        AutoIncreaseRule,
        ContributionsRule,
        EligibilityRule,
        MatchRule,
        NonElectiveRule,
    )

    RULE_MODULES_LOADED = True
except ImportError as e:
    print(f"Error importing rule modules: {e}")
    RULE_MODULES_LOADED = False

    # Define dummy modules with placeholder apply functions
    class DummyRuleModule:
        def __init__(self, name, error):
            self.name = name
            self.error = error

        def apply(self, *args, **kwargs):
            raise NotImplementedError(
                f"Rule module '{self.name}' apply function called, but module failed to import: {self.error}"
            )

    # Assign dummy modules if import failed
    eligibility = DummyRuleModule("eligibility", e)
    auto_enrollment = DummyRuleModule("auto_enrollment", e)
    auto_increase = DummyRuleModule("auto_increase", e)
    contributions = DummyRuleModule("contributions", e)

    # Need dummy validators too if they failed
    class EligibilityRule:
        pass

    class AutoEnrollmentRule:
        pass

    class AutoIncreaseRule:
        pass

    class ContributionsRule:
        pass

    class MatchRule:
        pass

    class NonElectiveRule:
        pass

    class IRSYearLimits:
        pass


# Import common elements from state schema (centralized constants)
from cost_model.state.schema import (
    ACTIVE_STATUS,
    ELIGIBILITY_ENTRY_DATE,
    EMP_DEFERRAL_RATE,
    EMP_TERM_DATE,
    INACTIVE_STATUS,
    IS_ELIGIBLE,
    STATUS_COL,
)

logger = logging.getLogger(__name__)

# --- Helper Functions --- #


def set_status_based_on_termination(df, sim_year, log=None):
    """Sets employee status to Active or Inactive based on termination date relative to year end."""
    log = log or logger
    plan_end_date = pd.Timestamp(f"{sim_year}-12-31")
    # Ensure termination date is datetime
    if EMP_TERM_DATE not in df.columns:
        df[EMP_TERM_DATE] = pd.NaT
    df[EMP_TERM_DATE] = pd.to_datetime(df[EMP_TERM_DATE], errors="coerce")

    active_mask = df[EMP_TERM_DATE].isna() | (df[EMP_TERM_DATE] > plan_end_date)
    df[STATUS_COL] = np.where(active_mask, ACTIVE_STATUS, INACTIVE_STATUS)
    log.debug(
        f"Status set for year {sim_year}: Active={(active_mask).sum()}, Inactive={(~active_mask).sum()}"
    )
    return df


def _normalize_bool_flag(df, col, is_date=False):
    """Ensures a column exists and normalizes it to boolean or NaT for dates."""
    if col not in df.columns:
        df[col] = pd.NaT if is_date else False

    if is_date:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    else:
        # Normalize to boolean using astype('boolean'), which is future-proof and avoids deprecated replace()
        # First, map common truthy/falsy string/number values to Python bools
        df[col] = df[col].map(
            {
                True: True,
                False: False,
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "": False,
                np.nan: False,
            }
        )
        # Convert to pandas nullable boolean type, filling potential remaining NaNs with False
        try:
            df[col] = df[col].astype("boolean").fillna(False)
        except TypeError:
            logger.warning(
                f"Could not robustly convert column '{col}' to boolean, defaulting unhandled values to False."
            )
            df[col] = df[col].apply(lambda x: bool(x) if pd.notna(x) else False).astype("boolean")

    return df


# --- Main Rule Application Engine --- #


def apply_rules_for_year(
    population_df: pd.DataFrame,
    year_config: Mapping[str, Any],
    sim_year: int,
    parent_logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Applies a sequence of plan rule functions to the population DataFrame for a given year.

    Args:
        population_df: DataFrame representing population state AFTER dynamics for the year.
        year_config: The fully resolved configuration object for the scenario (e.g., a Pydantic model instance).
        sim_year: The current simulation year.
        parent_logger: Optional logger instance.

    Returns:
        DataFrame with plan rule effects applied (eligibility, contributions, etc.).
    """
    log = parent_logger or logger
    log.info(
        f"--- Applying Plan Rules for Simulation Year {sim_year} --- RML: {RULE_MODULES_LOADED}"
    )

    if population_df.empty:
        log.warning(
            f"Input DataFrame for year {sim_year} rules application is empty. Returning empty."
        )
        return population_df

    df = population_df.copy()

    # Get year start/end dates
    sim_year_start_date = pd.Timestamp(f"{sim_year}-01-01")
    sim_year_end_date = pd.Timestamp(f"{sim_year}-12-31")

    # Safely get plan rules configuration using getattr
    # Assumes year_config has a 'plan_rules' attribute which is a Pydantic model or dict
    plan_rules_config = getattr(year_config, "plan_rules", None)
    if not plan_rules_config:
        log.warning(
            f"No 'plan_rules' found in year_config for {sim_year}. Skipping all rule applications."
        )
        return df

    # Safely get IRS limits (assuming it's a dict keyed by year: {2024: {...}, 2025: {...}})
    # Expecting irs_limits to be an attribute of plan_rules_config
    irs_limits_raw = getattr(plan_rules_config, "irs_limits", {})

    # Convert raw dictionary IRS limits to IRSYearLimits model objects
    # The contributions.apply function expects Dict[int, IRSYearLimits] not Dict[int, Dict[str, Any]]
    irs_limits_all_years = {}
    if irs_limits_raw and isinstance(irs_limits_raw, dict):
        from ..config.models import IRSYearLimits

        for year, limits_dict in irs_limits_raw.items():
            if isinstance(limits_dict, dict):
                try:
                    # Convert dictionary to IRSYearLimits model
                    irs_limits_all_years[year] = IRSYearLimits(**limits_dict)
                    log.debug(f"Successfully converted IRS limits for year {year}: {limits_dict}")
                except Exception as e:
                    log.warning(
                        f"Failed to convert IRS limits for year {year}: {e}. Skipping this year."
                    )
            elif hasattr(limits_dict, "compensation_limit"):
                # Already an IRSYearLimits object
                irs_limits_all_years[year] = limits_dict
            else:
                log.warning(
                    f"Invalid IRS limits format for year {year}: {type(limits_dict)}. Skipping this year."
                )

    if not irs_limits_all_years:
        log.warning(
            f"IRS limits missing or could not be converted for {sim_year}. Contribution calculations might fail."
        )

    # === Status & Pre-processing ===
    # 1. Determine Active/Inactive Status based on termination date
    df = set_status_based_on_termination(df, sim_year, log=log)

    # 2. Normalize common boolean/flag columns (ensure they exist and have correct type)
    log.debug(
        "Normalizing flags: is_participating, ae_opted_out, ai_opted_out, is_eligible, eligibility_entry_date"
    )
    df = _normalize_bool_flag(df, "is_participating")
    df = _normalize_bool_flag(df, "ae_opted_out")
    df = _normalize_bool_flag(df, "ai_opted_out")
    df = _normalize_bool_flag(df, IS_ELIGIBLE)
    df = _normalize_bool_flag(df, ELIGIBILITY_ENTRY_DATE, is_date=True)

    # 3. Ensure Deferral Rate exists and is numeric
    if EMP_DEFERRAL_RATE not in df.columns:
        log.warning(f"Column '{EMP_DEFERRAL_RATE}' not found. Adding with default 0.0.")
        df[EMP_DEFERRAL_RATE] = 0.0
    df[EMP_DEFERRAL_RATE] = pd.to_numeric(df[EMP_DEFERRAL_RATE], errors="coerce").fillna(0.0)

    # === Apply Rules Sequentially ===

    # 4. Eligibility
    # Safely get eligibility config (should be an EligibilityRules instance or None)
    elig_config = getattr(plan_rules_config, "eligibility", None)
    if elig_config:
        try:
            log.debug(f"Applying eligibility rules with config: {elig_config}")
            df = eligibility.apply(df, elig_config, sim_year_end_date)
            log.info(f"After eligibility.apply, columns: {list(df.columns)}")
        except Exception as e:
            log.error(f"Error applying Eligibility rules: {e}", exc_info=True)
    else:
        log.warning(
            "No 'eligibility' configuration found in plan_rules. Skipping eligibility rules."
        )

    # 5. Auto-Enrollment (AE)
    ae_config = getattr(plan_rules_config, "auto_enrollment", None)
    if ae_config:
        try:
            if getattr(ae_config, "enabled", False):
                log.debug(f"Applying Auto-Enrollment rules with config: {ae_config}")
                df = auto_enrollment.apply(df, ae_config, sim_year_start_date, sim_year_end_date)
                log.info(f"After auto_enrollment.apply, columns: {list(df.columns)}")
            else:
                log.info(f"Auto-Enrollment disabled in config for {sim_year}.")
        except Exception as e:
            log.error(f"Error applying Auto-Enrollment rules: {e}", exc_info=True)
    else:
        log.info(f"No 'auto_enrollment' configuration found in plan_rules for {sim_year}.")

    # 6. Auto-Increase (AI)
    # Safely get AI config (should be AutoIncreaseRules instance or None)
    ai_config = getattr(plan_rules_config, "auto_increase", None)
    if ai_config:
        try:
            # No need to re-validate with AutoIncreaseRule(**...); ai_config is already the model
            if getattr(ai_config, "enabled", False):
                log.debug(f"Applying Auto-Increase rules with config: {ai_config}")
                # Call apply via the imported module, passing the config model and year
                df = auto_increase.apply(df, ai_config, sim_year)
            else:
                log.info(f"Auto-Increase disabled in config for {sim_year}.")
        except Exception as e:
            log.error(f"Error applying Auto-Increase rules: {e}", exc_info=True)
    else:
        log.info(f"No 'auto_increase' configuration found in plan_rules for {sim_year}.")

    # 7. Contributions (Employee, Employer Match, Employer Non-Elective)
    # This step relies on contributions.apply handling compensation calculations internally
    contrib_config = getattr(plan_rules_config, "contributions", None)
    match_config = getattr(plan_rules_config, "employer_match", None)
    nec_config = getattr(plan_rules_config, "employer_nec", None)
    days_worked_col = None
    if contrib_config and match_config and nec_config and irs_limits_all_years:
        try:
            # No need to re-validate; contrib_config, match_config, nec_config are already models
            log.debug(f"Applying contributions with config: {contrib_config}")
            df = contributions.apply(
                df,
                contrib_config,
                match_config,
                nec_config,
                irs_limits_all_years,
                sim_year,
                sim_year_start_date,
                sim_year_end_date,
            )
            log.info(f"After contributions.apply, columns: {list(df.columns)}")
            # Save days_worked column if present
            if "days_worked" in df.columns:
                days_worked_col = df["days_worked"].copy()
        except Exception as e:
            log.error(f"Error applying Contributions rules: {e}", exc_info=True)
    elif not irs_limits_all_years:
        log.warning(
            f"Skipping contribution calculation for {sim_year} due to missing or invalid IRS limits configuration."
        )
    else:
        log.info(
            f"Contributions calculation skipped for {sim_year} due to missing contribution/match/nec configurations."
        )

    # Ensure days_worked is present in the final DataFrame
    if days_worked_col is not None:
        df["days_worked"] = days_worked_col
    elif "days_worked" not in df.columns:
        log.warning("days_worked column missing after rule application; check contributions logic.")

    # === Final Cleanup ===
    # (Add checks for expected output columns if necessary)

    log.info(f"--- Finished Applying Plan Rules for Simulation Year {sim_year} ---")
    return df
