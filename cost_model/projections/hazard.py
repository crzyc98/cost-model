# cost_model/projections/hazard.py
"""
Hazard module for generating hazard tables used in workforce projections.
QuickStart: see docs/cost_model/projections/hazard.md
"""

import pandas as pd
import logging
from typing import List
from cost_model.utils.columns import EMP_ROLE

logger = logging.getLogger(__name__)

def build_hazard_table(
    years: List[int],
    initial_snapshot: pd.DataFrame,
    global_params,
    plan_rules_config
) -> pd.DataFrame:
    """Generates the hazard table based on configuration and initial snapshot."""
    logger.info("Generating hazard table...")
    from cost_model.utils.columns import EMP_TENURE
    if EMP_ROLE in initial_snapshot.columns and 'tenure_band' in initial_snapshot.columns:
        unique_roles_tenures = initial_snapshot[[EMP_ROLE, 'tenure_band']].drop_duplicates().to_dict('records')
    else:
        logger.warning(f"'{EMP_ROLE}' or 'tenure_band' not in initial snapshot. Using default 'all'/'all'.")
        unique_roles_tenures = [{EMP_ROLE: 'all', 'tenure_band': 'all'}]

    global_term_rate = getattr(global_params, 'annual_termination_rate', 0.10)
    global_comp_raise_pct = getattr(global_params, 'annual_compensation_increase_rate', 0.03)
    global_nh_term_rate = getattr(global_params, 'new_hire_termination_rate', 0.25)
    logger.info(f"Using global rates: Term={global_term_rate}, CompPct={global_comp_raise_pct}")

    records = []
    for year in years:
        for combo in unique_roles_tenures:
            records.append({
                'simulation_year': year,
                EMP_ROLE: combo[EMP_ROLE],
                'tenure_band': combo['tenure_band'],
                'term_rate': global_term_rate,
                'comp_raise_pct': global_comp_raise_pct,
                'new_hire_termination_rate': global_nh_term_rate,
                'cola_pct': getattr(global_params, 'cola_pct', 0.0),
                'cfg': plan_rules_config
            })
    if records:
        df = pd.DataFrame(records)
        logger.info(f"Hazard table with {len(records)} rows.")
    else:
        cols = ['simulation_year', EMP_ROLE, 'tenure_band', 'term_rate', 'comp_raise_pct', 'new_hire_termination_rate', 'cola_pct', 'cfg']
        df = pd.DataFrame(columns=cols)
        logger.warning("Empty hazard table created.")
    return df
