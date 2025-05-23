# utils/plan_rules_engine.py
# This file contains the logic for applying plan rules (Phase II)

import logging
import pandas as pd
import numpy as np
from typing import Mapping, Any

# Import specific rule application functions using relative paths
from .rules.eligibility import apply as apply_eligibility
from .rules.auto_enrollment import apply as apply_auto_enrollment
from .rules.auto_increase import apply as apply_auto_increase
from .rules.contributions import apply as apply_contributions

# Import rule validators/data classes
from .rules.validators import (
    EligibilityRule,
    AutoEnrollmentRule,
    ContributionsRule,
    MatchRule,
    NonElectiveRule,
)

# Import constants
from cost_model.state.schema import STATUS_COL, EMP_TERM_DATE
from .constants import ACTIVE_STATUS, INACTIVE_STATUS

logger = logging.getLogger(__name__)


# Phase II: Apply plan rules (eligibility, auto_enroll, auto_increase, contributions)
def apply_plan_rules(
    df: pd.DataFrame, scenario_config: Mapping[str, Any], year_num: int
) -> pd.DataFrame:
    """
    Applies a sequence of plan rule functions to the DataFrame for a given year.
    Modifies the DataFrame in place for eligibility, AE, AI, and contributions.
    """
    sim_year = scenario_config["start_year"] + year_num - 1
    start_date = pd.Timestamp(f"{sim_year}-01-01")
    end_date = pd.Timestamp(f"{sim_year}-12-31")

    # Always reset status based on termination date (active if no termination)
    df[STATUS_COL] = np.where(df[EMP_TERM_DATE].isna(), ACTIVE_STATUS, INACTIVE_STATUS)

    # ensure deferral_rate exists
    if "deferral_rate" not in df and "pre_tax_deferral_percentage" in df:
        df["deferral_rate"] = df["pre_tax_deferral_percentage"]
    elif "deferral_rate" not in df:  # Create if neither exists
        df["deferral_rate"] = 0.0

    # normalize participation and enrollment flags using pandas nullable BooleanDtype
    for col in ["is_participating", "ae_opted_out", "ai_opted_out"]:
        # get existing col or create empty BooleanDtype Series
        series = (
            df[col] if col in df.columns else pd.Series(index=df.index, dtype="boolean")
        )
        # cast to BooleanDtype and fill missing as False
        df[col] = series.astype("boolean").fillna(False)

    # Eligibility
    # Use .get() for safer access to potentially missing keys
    elig_config = scenario_config.get("plan_rules", {}).get("eligibility", {})
    if elig_config:  # Only apply if config exists
        elig_rules = EligibilityRule(**elig_config)
        df = apply_eligibility(df, elig_rules, end_date)

    # Auto-enrollment
    ae_config = scenario_config.get("plan_rules", {}).get("auto_enrollment", {})
    if ae_config and ae_config.get("enabled", False):
        ae_rules = AutoEnrollmentRule(**ae_config)
        df = apply_auto_enrollment(df, ae_rules, start_date, end_date)

    # Auto-increase
    ai_config = scenario_config.get("plan_rules", {}).get("auto_increase", {})
    if ai_config and ai_config.get("enabled", False):
        # Note: apply_auto_increase expects the whole plan_rules dict
        df = apply_auto_increase(df, scenario_config.get("plan_rules", {}), sim_year)

    # Contributions
    contrib_config = scenario_config.get("plan_rules", {}).get("contributions", {})
    if contrib_config and contrib_config.get("enabled", False):
        contrib_rules = ContributionsRule(**contrib_config)
        match_config = scenario_config.get("plan_rules", {}).get("employer_match", {})
        nec_config = scenario_config.get("plan_rules", {}).get("employer_nec", {})
        irs_limits = scenario_config.get("plan_rules", {}).get("irs_limits", {})

        if match_config and nec_config and irs_limits:  # Check sub-configs exist
            match_rules = MatchRule(**match_config)
            nec_rules = NonElectiveRule(**nec_config)
            df = apply_contributions(
                df,
                contrib_rules,
                match_rules,
                nec_rules,
                irs_limits,
                sim_year,
                start_date,
                end_date,
            )
        else:
            logger.warning(
                "Contributions enabled, but missing match, NEC, or IRS limit config. Skipping calculation."
            )

    return df
