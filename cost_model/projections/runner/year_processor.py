# /cost_model/projections/runner/year_processor.py
"""
Handles the processing of each year in the projection.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy.random import Generator

logger = logging.getLogger(__name__)

from cost_model.engines.run_one_year import run_one_year
from cost_model.projections.hazard import build_hazard_table
from cost_model.state.schema import (
    EVT_COLA,
    EVT_COMP,
    EVT_CONTRIB,
    EVT_HIRE,
    EVT_PROMOTION,
    EVT_RAISE,
    EVT_TERM,
)

from .summaries import make_yearly_summaries


def process_year(
    year: int,
    current_snapshot: pd.DataFrame,
    cumulative_log: pd.DataFrame,
    global_params: Dict[str, Any],
    plan_rules: Dict[str, Any],
    rng: Generator,
    years: List[int],
    census_path: str,
    ee_contrib_event_types: List[str],
) -> Tuple[Tuple[pd.DataFrame, None], pd.DataFrame, Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    """
    Process a single year in the projection.

    Returns:
      - new_snapshot (as a tuple for compatibility)
      - updated_cumulative_log
      - core_summary
      - employment_summary
      - year_eoy_rows
    """
    # 1. Build hazard table
    hazard_table = build_hazard_table([year], current_snapshot, global_params, plan_rules)

    # 2. Run one year simulation (events + exact final_snapshot)
    year_events, final_snapshot = run_one_year(
        event_log=cumulative_log,
        prev_snapshot=current_snapshot,
        year=year,
        global_params=global_params,
        plan_rules=plan_rules,
        hazard_table=hazard_table,
        rng=rng,
        census_template_path=census_path,
        rng_seed_offset=0,
        deterministic_term=False,
    )

    # 3. Take the exact final_snapshot returned
    new_snapshot = final_snapshot

    # 4. Append this year's events to the cumulative log
    updated_cumulative_log = pd.concat([cumulative_log, year_events], ignore_index=True)

    # 5. Filter out any employees terminated in prior years so we can report EOY
    year_eoy_rows = new_snapshot.copy()
    if "employee_termination_date" in year_eoy_rows.columns:
        mask = year_eoy_rows["employee_termination_date"].isna() | (
            pd.to_datetime(year_eoy_rows["employee_termination_date"]).dt.year == year
        )
        filtered_count = len(year_eoy_rows) - mask.sum()
        year_eoy_rows = year_eoy_rows.loc[mask]
        logger.info(
            f"[{year}] Dropped {filtered_count} employees "
            "terminated in prior years from EOY snapshot"
        )

    # 6. Build summaries
    from cost_model.state.schema import EMP_ACTIVE

    start_headcount = int(current_snapshot[EMP_ACTIVE].sum())

    core_summary, employment_summary = make_yearly_summaries(
        snapshot=new_snapshot, year_events=year_events, year=year, start_headcount=start_headcount
    )

    # 7. Return in the expected shape
    return (
        (new_snapshot, None),
        updated_cumulative_log,
        core_summary,
        employment_summary,
        year_eoy_rows,
    )
