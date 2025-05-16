# cost_model/projections/hazard.py
"""
Hazard module for generating hazard tables used in workforce projections.
QuickStart: see docs/cost_model/projections/hazard.md
"""

import pandas as pd
import logging
from typing import List
from cost_model.utils.columns import (
    EMP_LEVEL,
    EMP_TENURE_BAND,
    SIMULATION_YEAR,
    TERM_RATE,
    COMP_RAISE_PCT,
    NEW_HIRE_TERM_RATE,
    COLA_PCT,
    CFG
)

logger = logging.getLogger(__name__)

def build_hazard_table(
    years: List[int],
    initial_snapshot: pd.DataFrame,
    global_params,
    plan_rules_config
) -> pd.DataFrame:
    """Generates the hazard table based on configuration and initial snapshot."""
    logger.info("Generating hazard table...")
    from cost_model.utils.columns import EMP_TENURE_BAND
    if EMP_LEVEL in initial_snapshot.columns and EMP_TENURE_BAND in initial_snapshot.columns:
        unique_levels_tenures = initial_snapshot[[EMP_LEVEL, EMP_TENURE_BAND]].drop_duplicates().to_dict('records')
    else:
        logger.warning(f"'{EMP_LEVEL}' or '{EMP_TENURE_BAND}' not in initial snapshot. Using default '1'/'all'.")
        unique_levels_tenures = [{EMP_LEVEL: 1, EMP_TENURE_BAND: 'all'}]

    # Robustly check for annual_termination_rate, comp_raise_pct, and nh_term_rate
    if hasattr(global_params, 'annual_termination_rate'):
        global_term_rate = global_params.annual_termination_rate
    else:
        logger.warning("global_params missing 'annual_termination_rate'. Using default 0.10. Available attributes: %s", dir(global_params))
        global_term_rate = 0.10
    if hasattr(global_params, 'annual_compensation_increase_rate'):
        global_comp_raise_pct = global_params.annual_compensation_increase_rate
    else:
        logger.warning("global_params missing 'annual_compensation_increase_rate'. Using default 0.03. Available attributes: %s", dir(global_params))
        global_comp_raise_pct = 0.03
    if hasattr(global_params, 'new_hire_termination_rate'):
        global_nh_term_rate = global_params.new_hire_termination_rate
    else:
        logger.warning("global_params missing 'new_hire_termination_rate'. Using default 0.25. Available attributes: %s", dir(global_params))
        global_nh_term_rate = 0.25
    logger.info(f"Using global rates: Term={global_term_rate}, CompPct={global_comp_raise_pct}, NH_Term={global_nh_term_rate}")

    records = []
    for year in years:
        for combo in unique_levels_tenures:
            records.append({
                SIMULATION_YEAR: year,
                EMP_LEVEL: combo[EMP_LEVEL],
                EMP_TENURE_BAND: combo[EMP_TENURE_BAND],
                TERM_RATE: global_term_rate,
                COMP_RAISE_PCT: global_comp_raise_pct,
                NEW_HIRE_TERM_RATE: global_nh_term_rate,
                COLA_PCT: getattr(global_params, 'cola_pct', 0.0),
                CFG: plan_rules_config
            })
    if records:
        df = pd.DataFrame(records)
        logger.info(f"Hazard table with {len(records)} rows.")
    else:
        cols = [SIMULATION_YEAR, EMP_LEVEL, EMP_TENURE_BAND, TERM_RATE, COMP_RAISE_PCT, NEW_HIRE_TERM_RATE, COLA_PCT, CFG]
        df = pd.DataFrame(columns=cols)
        logger.warning("Empty hazard table created.")
    return df
