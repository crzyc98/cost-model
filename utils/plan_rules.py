# utils/plan_rules.py

from typing import Any, Dict
import pandas as pd
from pandas import DataFrame

from utils.rules.validators import (
    EligibilityRule,
    AutoEnrollmentRule,
    AutoIncreaseRule,
    ContributionsRule,
    PlanRules,
    OutcomeDistribution
)
from utils.rules.eligibility    import apply as _apply_eligibility
from utils.rules.auto_enrollment import apply as _apply_auto_enrollment
from utils.rules.auto_increase   import apply as _apply_auto_increase
from utils.rules.contributions   import apply as _apply_contributions


def apply_eligibility(
    df: DataFrame,
    scenario_config: Dict[str, Any],
    simulation_year_end: pd.Timestamp
) -> DataFrame:
    """
    Extracts eligibility config and runs eligibility rule.
    """
    pr = scenario_config.get("plan_rules", {}) or {}
    elig_cfg = pr.get('eligibility', {}) or {}
    rules = EligibilityRule(**elig_cfg) if isinstance(elig_cfg, dict) else elig_cfg
    return _apply_eligibility(df, rules, simulation_year_end)


def apply_auto_enrollment(
    df: DataFrame,
    plan_rules: Dict[str, Any],
    year_start: pd.Timestamp,
    year_end: pd.Timestamp
) -> DataFrame:
    """Facade to auto-enrollment rules."""
    from utils.rules.validators import AutoEnrollmentRule, OutcomeDistribution
    
    ae_cfg = plan_rules.get('auto_enrollment', {}) or {}
    
    # Convert nested outcome_distribution dict to model
    if isinstance(ae_cfg.get('outcome_distribution'), dict):
        ae_cfg['outcome_distribution'] = OutcomeDistribution(**ae_cfg['outcome_distribution'])
    
    ae_rules = AutoEnrollmentRule(**ae_cfg)
    
    # Make the match rate available in the DF
    df = df.copy()
    df['rate_for_max_match'] = ae_rules.increase_to_match_rate
    
    return _apply_auto_enrollment(df, ae_rules, year_start, year_end)


def apply_auto_increase(
    df: DataFrame,
    plan_rules: Dict[str, Any],
    simulation_year: int
) -> DataFrame:
    """Facade to auto-increase rules."""
    from utils.rules.auto_increase import apply as _apply_auto_increase
    from utils.rules.validators import AutoIncreaseRule

    ai_cfg_dict = plan_rules.get('auto_increase', {})
    ai_rules = AutoIncreaseRule(**ai_cfg_dict)
    return _apply_auto_increase(df, ai_rules, simulation_year)


def calculate_contributions(
    df: DataFrame,
    scenario_config: Dict[str, Any],
    simulation_year: int
) -> DataFrame:
    """
    Extracts auto_increase config and runs auto-increase rule.
    """
    pr = scenario_config.get('plan_rules', {}) or {}
    ai_cfg = pr.get('auto_increase', {}) or {}
    rules = AutoIncreaseRule(**ai_cfg) if isinstance(ai_cfg, dict) else ai_cfg
    return _apply_auto_increase(df, rules, simulation_year)


def calculate_contributions(
    df: DataFrame,
    scenario_config: Dict[str, Any],
    simulation_year: int,
    year_start: pd.Timestamp,
    year_end: pd.Timestamp
) -> DataFrame:
    """
    Extracts contributions config and runs contributions rule.
    """
    pr = scenario_config.get('plan_rules', {}) or {}
    c_cfg = pr.get('contributions', {}) or {}
    rules = ContributionsRule(**c_cfg) if isinstance(c_cfg, dict) else c_cfg
    return _apply_contributions(df, rules, simulation_year, year_start, year_end)


def apply_plan_change_deferral_response(
    df: DataFrame,
    current_scenario_config: Dict[str, Any],
    baseline_scenario_config: Dict[str, Any],
    simulation_year: int,
    start_year: int
) -> DataFrame:
    """
    Stub for any “plan change” deferral logic.  No-op by default.
    """
    return df
