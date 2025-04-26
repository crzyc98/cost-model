from utils.rules.eligibility import apply as apply_eligibility
from utils.rules.auto_enrollment import apply as apply_auto_enrollment
from utils.rules.auto_increase import apply as apply_auto_increase
from utils.rules.contributions import apply as apply_contributions
from utils.rules.response import apply as apply_response

# Facade functions for compatibility with legacy code

def determine_eligibility(df, scenario_config, simulation_year_end_date):
    """Facade: delegate to eligibility rule."""
    plan_rules = scenario_config.get('plan_rules', {})
    return apply_eligibility(df, plan_rules, simulation_year_end_date)

def calculate_contributions(df, scenario_config, simulation_year, year_start_date, year_end_date):
    """Facade: delegate to contributions rule."""
    return apply_contributions(df, scenario_config, simulation_year, year_start_date, year_end_date)

# If apply_plan_change_deferral_response is needed, provide a stub or delegate

def apply_plan_change_deferral_response(df, current_scenario_config, baseline_scenario_config, simulation_year, start_year):
    # TODO: Implement or delegate if logic exists elsewhere
    # For now, return df unchanged (no-op)
    return df
