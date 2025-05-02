"""
rules/response.py
Models existing participants' potential deferral rate increase in response to 
a more generous match formula compared to a baseline scenario.
"""
import numpy as np
import logging
from cost_model.rules.formula_parsers import parse_match_formula

logger = logging.getLogger(__name__)

def apply(df, current_scenario_config, baseline_scenario_config, simulation_year, start_year):
    """
    Models existing participants' potential deferral rate increase in response to 
    a more generous match formula compared to a baseline scenario.

    Updates 'deferral_rate' column inplace.
    """
    logger.info(f"Applying plan change response logic for {simulation_year}...")

    plan_rules = current_scenario_config.get('plan_rules', {})
    response_config = plan_rules.get('match_change_response', {})
    if not response_config.get('enabled', False):
        logger.info("Plan change response logic is disabled.")
        return df

    if not baseline_scenario_config:
        logger.error("Baseline scenario config not provided. Cannot apply plan change response logic.")
        return df

    # Run only in the first year
    if simulation_year != start_year:
        return df

    # Ensure required columns
    if 'is_participating' not in df.columns or 'deferral_rate' not in df.columns:
        logger.warning("Required columns missing. Cannot apply plan change response logic.")
        return df
    if 'status' not in df.columns:
        is_active = True
    else:
        is_active = (df['status'] == 'Active')

    # Compare match formulas
    curr = plan_rules.get('employer_match_formula', "")
    baseline_plan_rules = baseline_scenario_config.get('plan_rules', {})
    base = baseline_plan_rules.get('employer_match_formula', "")
    if curr == base:
        logger.info("Match formulas same as baseline. No response needed.")
        return df

    curr_rate, curr_cap = parse_match_formula(curr)
    base_rate, base_cap = parse_match_formula(base)

    is_more_generous = (curr_cap > base_cap) or (curr_cap == base_cap and curr_rate > base_rate) or (curr and not base)
    if not is_more_generous:
        logger.info(f"Current match ('{curr}') not more generous than baseline ('{base}').")
        return df

    logger.info(f"Detected more generous match ('{curr}' vs '{base}'). Applying response.")

    optimal_rate = curr_cap
    if optimal_rate <= 0:
        logger.warning("Cannot determine optimal deferral rate. Skipping response.")
        return df

    mask = is_active & df.get('is_participating', False) & (df['deferral_rate'] > 0) & (df['deferral_rate'] < optimal_rate)
    indices = df.index[mask]
    num = len(indices)
    if num == 0:
        logger.info("No participants deferring below optimal rate.")
        return df

    logger.info(f"{num} participants may increase deferral to {optimal_rate:.2%}.")
    prob = response_config.get('increase_probability', 0.0)
    target = response_config.get('increase_target', 'optimal')

    decision = np.random.rand(num) < prob
    chosen = indices[decision]
    if len(chosen):
        logger.info(f"Applying deferral increase to {len(chosen)} participants.")
        if target == 'optimal':
            df.loc[chosen, 'deferral_rate'] = optimal_rate
        else:
            logger.warning(f"Unsupported increase_target '{target}'. No change applied.")
    return df
