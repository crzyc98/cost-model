# utils/plan_rules.py

from typing import Any, Dict
import pandas as pd
from pandas import DataFrame

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
    Extracts plan_rules from scenario_config and runs eligibility.
    """
    plan_rules = scenario_config.get("plan_rules", {})
    return _apply_eligibility(df, plan_rules, simulation_year_end)


def apply_auto_enrollment(
    df: DataFrame,
    scenario_config: Dict[str, Any],
    simulation_year_start: pd.Timestamp,
    simulation_year_end: pd.Timestamp
) -> DataFrame:
    """
    Extracts plan_rules from scenario_config and runs auto-enrollment.
    """
    plan_rules = scenario_config.get("plan_rules", {})
    return _apply_auto_enrollment(df, plan_rules, simulation_year_start, simulation_year_end)


def apply_auto_increase(
    df: DataFrame,
    scenario_config: Dict[str, Any],
    simulation_year: int
) -> DataFrame:
    """
    Extracts plan_rules from scenario_config and runs auto-increase.
    """
    plan_rules = scenario_config.get("plan_rules", {})
    return _apply_auto_increase(df, plan_rules, simulation_year)


def calculate_contributions(
    df: DataFrame,
    scenario_config: Dict[str, Any],
    simulation_year: int,
    year_start: pd.Timestamp,
    year_end: pd.Timestamp
) -> DataFrame:
    """
    Extracts plan_rules from scenario_config and runs contributions.
    """
    plan_rules = scenario_config.get("plan_rules", {})
    return _apply_contributions(df, plan_rules, simulation_year, year_start, year_end)


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