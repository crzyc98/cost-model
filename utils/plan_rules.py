# /utils/plan_rules.py

from typing import Any, Dict
import pandas as pd
from pandas import DataFrame

from utils.rules.eligibility    import apply as apply_eligibility
from utils.rules.auto_enrollment import apply as _apply_auto_enrollment
from utils.rules.auto_increase   import apply as _apply_auto_increase
from utils.rules.contributions   import apply as apply_contributions

# Facade functions for compatibility with legacy code

def determine_eligibility(
    df: DataFrame,
    scenario_config: Dict[str, Any],
    simulation_year_end_date: pd.Timestamp
) -> DataFrame:
    """Facade to eligibility rules."""
    plan_rules = scenario_config.get('plan_rules', {})
    return apply_eligibility(df, plan_rules, simulation_year_end_date)

def apply_auto_enrollment(
    df: DataFrame,
    plan_rules: Dict[str, Any],
    year_start: pd.Timestamp,
    year_end: pd.Timestamp
) -> DataFrame:
    """Facade to auto-enrollment rules."""
    return _apply_auto_enrollment(df, plan_rules, year_start, year_end)

def apply_auto_increase(
    df: DataFrame,
    plan_rules: Dict[str, Any],
    simulation_year: int
) -> DataFrame:
    """Facade to auto-increase rules."""
    return _apply_auto_increase(df, plan_rules, simulation_year)

def calculate_contributions(
    df: DataFrame,
    scenario_config: Dict[str, Any],
    simulation_year: int,
    year_start: pd.Timestamp,
    year_end: pd.Timestamp
) -> DataFrame:
    """Facade to contributions rules."""
    return apply_contributions(df, scenario_config, simulation_year, year_start, year_end)

# If apply_plan_change_deferral_response is needed, provide a stub or delegate

def apply_plan_change_deferral_response(df, current_scenario_config, baseline_scenario_config, simulation_year, start_year):
    # TODO: Implement or delegate if logic exists elsewhere
    # For now, return df unchanged (no-op)
    return df
