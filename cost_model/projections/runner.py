# cost_model/projections/runner.py
"""
Core projection engine runner module.
QuickStart: see docs/cost_model/projections/runner.md
"""

import pandas as pd
import numpy as np
import logging
from types import SimpleNamespace
from typing import Dict, Tuple, Any, List  
from datetime import datetime  

# --- Core Simulation Engine ---
from cost_model.engines.run_one_year import run_one_year
from cost_model.projections.hazard import build_hazard_table
from cost_model.projections.snapshot import update_snapshot_with_events
from cost_model.projections.summaries.core import build_core_summary
from cost_model.projections.summaries.employment import (
    build_employment_status_summary,
    build_employment_status_snapshot
)
from cost_model.projections.utils import assign_employment_status, filter_prior_terminated
from cost_model.state.event_log import EVT_TERM, EMP_ID as EVENT_EMP_ID, EVENT_COLS, EVT_COMP, EVT_CONTRIB  
try:
    from cost_model.state.event_log import EVT_HIRE
except ImportError:
    EVT_HIRE = 'hire'

# --- State and Utility Imports ---
from cost_model.utils.columns import (
    EMP_ID, EMP_HIRE_DATE, EMP_ROLE, EMP_TERM_DATE,
    EMP_BIRTH_DATE, EMP_GROSS_COMP, EMP_DEFERRAL_RATE,
    EMP_ACTIVE, EMP_TENURE_BAND
)

# Define event priority mapping for snapshot updates
EVENT_PRIORITY = {
    EVT_HIRE: 0,
    EVT_TERM: 1,
    EVT_COMP: 2,
    EVT_CONTRIB: 3,
}

logger = logging.getLogger(__name__)

def run_projection_engine(
    config_ns: SimpleNamespace,
    initial_snapshot_df: pd.DataFrame,
    initial_event_log_df: pd.DataFrame
) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Runs the multi-year projection.
    (Args and Returns docstring remains the same)
    """
    global_params = config_ns.global_parameters
    plan_rules_config = config_ns.plan_rules

    start_year = getattr(global_params, 'start_year', 2025)
    projection_years_count = getattr(global_params, 'projection_years', 5)
    random_seed = getattr(global_params, 'random_seed', 42)

    logger.info(f"Running {projection_years_count}-year projection starting from {start_year}.")
    logger.info(f"Random seed: {random_seed}")
    
    # Initialize random number generator
    rng = np.random.default_rng(random_seed)

    # Initialize the current snapshot and current event log
    current_snapshot = initial_snapshot_df.copy()
    logger.info(f"[RUNNER] Initial snapshot: {len(current_snapshot)} total rows, {current_snapshot[EMP_ACTIVE].sum()} active rows")

    # Validate and use the 'active' column from create_initial_snapshot directly.
    # create_initial_snapshot is responsible for correctly interpreting the census, including 'active' status.
    if EMP_ACTIVE not in current_snapshot.columns:
        logger.error("CRITICAL: Initial snapshot from create_initial_snapshot is missing the 'active' column. This should not happen. Falling back to all active.")
        current_snapshot[EMP_ACTIVE] = pd.Series([True] * len(current_snapshot), index=current_snapshot.index)
    elif not pd.api.types.is_bool_dtype(current_snapshot[EMP_ACTIVE].dtype):
        logger.error(f"CRITICAL: Initial snapshot 'active' column (type: {current_snapshot[EMP_ACTIVE].dtype}) is not boolean. Attempting conversion or falling back.")
        try:
            # Attempt to convert to boolean. This is a fix-up for unexpected 'active' column types.
            current_snapshot[EMP_ACTIVE] = current_snapshot[EMP_ACTIVE].astype(bool)
        except Exception as e_conv:
            logger.error(f"Failed to convert 'active' column to bool: {e_conv}. Falling back to all active.")
            current_snapshot[EMP_ACTIVE] = pd.Series([True] * len(current_snapshot), index=current_snapshot.index)
    
    # Log the definitive initial active count taken from the snapshot's 'active' column.
    active_count_for_runner = current_snapshot[EMP_ACTIVE].sum()
    logger.info(f"Runner: Initial active headcount from snapshot's 'active' column: {active_count_for_runner}")

    # Verify consistency between 'active' column and EMP_TERM_DATE if possible, but 'active' column is primary.
    if EMP_TERM_DATE in current_snapshot.columns:
        active_via_term_date_count = current_snapshot[EMP_TERM_DATE].isna().sum()
        if active_via_term_date_count != active_count_for_runner:
            logger.warning(f"Initial Snapshot: Mismatch between active count from '{EMP_TERM_DATE}'.isna() ({active_via_term_date_count}) "
                           f"and 'active' column ({active_count_for_runner}). Proceeding with 'active' column as definitive.")
    else:
        logger.warning(f"Initial Snapshot: '{EMP_TERM_DATE}' column not present. Cannot verify 'active' column consistency.")

    current_cumulative_event_log = pd.DataFrame([], columns=EVENT_COLS)

    projection_sim_years = list(range(start_year, start_year + projection_years_count))

    # 0) grab the census file path for hire.run()
    census_template_path = (
        getattr(global_params, 'census_template_path', None)
        or getattr(config_ns, 'census', None)
    )
    if not census_template_path:
        logger.warning("'census_template_path' not provided in global_parameters or --census flag. Proceeding with None. If hiring logic is invoked, this may cause an error.")

    yearly_eoy_snapshots: Dict[int, pd.DataFrame] = {}
    summary_results_list = []
    employment_status_summary_data = []  

    ee_contrib_event_types = []
    if hasattr(plan_rules_config, 'enrollment'):
        ee_contrib_event_types.append(getattr(plan_rules_config.enrollment, 'event_type_contribution_auto', None))
        ee_contrib_event_types.append(getattr(plan_rules_config.enrollment, 'event_type_contribution_manual', None))
    if hasattr(plan_rules_config, 'contribution_increase'):
        ee_contrib_event_types.append(getattr(plan_rules_config.contribution_increase, 'event_type_contribution_increase', None))
    ee_contrib_event_types.append('EVT_CONTRIB_INCR')
    ee_contrib_event_types = [et for et in ee_contrib_event_types if et is not None and isinstance(et, str)]
    logger.debug(f"EE Contribution event types being tracked: {ee_contrib_event_types}")

    if initial_snapshot_df is not None and EMP_ACTIVE in initial_snapshot_df.columns:
        last_year_active_headcount = initial_snapshot_df[initial_snapshot_df[EMP_ACTIVE]].shape[0]
    else:
        last_year_active_headcount = 0
        logger.warning("[RUNNER] Initial snapshot is missing or has no 'active' column. First year growth rate might be off.")

    from cost_model.utils.columns import EMP_ROLE
    for yr_idx, current_sim_year in enumerate(projection_sim_years):
        logger.info(f"--- Simulating Year {current_sim_year} (Index {yr_idx}) ---")
        logger.debug(f"SOY {current_sim_year} - Snapshot shape: {current_snapshot.shape}, Active: {current_snapshot[EMP_ACTIVE].sum() if EMP_ACTIVE in current_snapshot else 'N/A'}")
        logger.debug(f"SOY {current_sim_year} - Cumulative Event Log shape: {current_cumulative_event_log.shape}")

        # --- Patch: Regenerate hazard table for the current year using the current snapshot ---
        hazard_table = build_hazard_table([current_sim_year], current_snapshot, global_params, plan_rules_config)
        # --- Check for missing (role, tenure_band) combos ---
        missing_combos = []
        if EMP_ROLE in current_snapshot.columns and EMP_TENURE_BAND in current_snapshot.columns:
            snapshot_combos = set(tuple(x) for x in current_snapshot[[EMP_ROLE, EMP_TENURE_BAND]].drop_duplicates().values)
            hazard_combos = set(tuple(x) for x in hazard_table[[EMP_ROLE, EMP_TENURE_BAND]].drop_duplicates().values)
            missing_combos = snapshot_combos - hazard_combos
            if missing_combos:
                logger.warning(f"[HAZARD TABLE] Year {current_sim_year}: Missing hazard table entries for combinations: {missing_combos}")
        else:
            logger.warning(f"[HAZARD TABLE] Year {current_sim_year}: EMP_ROLE or tenure_band not found in current snapshot; cannot check hazard table coverage.")

        # run_one_year now returns (full_event_log_for_year, prev_snapshot)
        # We need to explicitly update our current_snapshot using the events from run_one_year.
        event_log_for_year, _ = run_one_year(
            current_cumulative_event_log,                    # event_log
            current_snapshot.copy(),                         # prev_snapshot
            current_sim_year,                                # year
            global_params,                                   # global parameters (SimpleNamespace)
            plan_rules_config,                               # plan rules (SimpleNamespace)
            hazard_table,                                    # hazard_table DataFrame
            rng,                                             # random number generator
            census_template_path,                            # census file path (str)
            rng_seed_offset=current_sim_year,
            deterministic_term=getattr(global_params, 'deterministic_termination', False)
        )

        if event_log_for_year is None or not isinstance(event_log_for_year, pd.DataFrame):
            logger.error(f"[RUNNER YR={current_sim_year}] run_one_year returned invalid event_log_for_year (type: {type(event_log_for_year)}). Skipping snapshot update for this year.")
            # Optionally, decide how to handle this - e.g., use current_snapshot as is, or raise error
            # For now, we'll just log and the current_snapshot won't change for this year's events.
        else:
            logger.info(f"[RUNNER YR={current_sim_year}] Received {len(event_log_for_year)} events from run_one_year.")
            # Update the main current_snapshot using the events from this year
            current_snapshot, year_end_employee_ids = update_snapshot_with_events(
                current_snapshot, 
                event_log_for_year, 
                pd.Timestamp(f"{current_sim_year}-12-31"), # Apply events as of EOY for next snapshot
                EVENT_PRIORITY
            )
            logger.info(f"[RUNNER YR={current_sim_year}] Snapshot updated. Active after events: {current_snapshot[EMP_ACTIVE].sum()}")

        # Append to cumulative log
        if event_log_for_year is not None and not event_log_for_year.empty:
            # Ensure 'year' column is present and correctly typed in event_log_for_year before concat
            if 'year' not in event_log_for_year.columns:
                event_log_for_year['year'] = current_sim_year
            event_log_for_year['year'] = event_log_for_year['year'].astype(int)
            
            current_cumulative_event_log = pd.concat([current_cumulative_event_log, event_log_for_year], ignore_index=True)
            logger.info(f"[RUNNER YR={current_sim_year}] Cumulative event log now has {len(current_cumulative_event_log)} events.")
        else:
            logger.info(f"[RUNNER YR={current_sim_year}] No events from run_one_year to append to cumulative log.")

        active_employees_df = pd.DataFrame()
        if 'active' in current_snapshot.columns:
            active_employees_df = current_snapshot[current_snapshot[EMP_ACTIVE]].copy()
        active_headcount_eoy = len(active_employees_df)

        eligible_count = active_headcount_eoy
        participant_count = 0
        avg_deferral_rate_participants = 0.0
        if EMP_DEFERRAL_RATE in active_employees_df.columns and not active_employees_df.empty:
            participants_df = active_employees_df[active_employees_df[EMP_DEFERRAL_RATE] > 0]
            participant_count = len(participants_df)
            if participant_count > 0:
                avg_deferral_rate_participants = participants_df[EMP_DEFERRAL_RATE].mean()

        total_employee_gross_compensation = 0.0
        if EMP_GROSS_COMP in active_employees_df.columns and not active_employees_df.empty:
            total_employee_gross_compensation = active_employees_df[EMP_GROSS_COMP].sum()

        total_ee_contribution_this_year = 0.0
        if event_log_for_year is not None and not event_log_for_year.empty and \
           'event_type' in event_log_for_year.columns and \
           'value_num' in event_log_for_year.columns:
            ee_contrib_events_df = event_log_for_year[
                event_log_for_year['event_type'].isin(ee_contrib_event_types)
            ]
            total_ee_contribution_this_year = ee_contrib_events_df['value_num'].sum()

        total_er_contribution_this_year = 0.0
        total_compensation_this_year = total_employee_gross_compensation

        avg_compensation_active = (total_employee_gross_compensation / active_headcount_eoy) if active_headcount_eoy > 0 else 0.0

        avg_age_active = 0.0
        if EMP_BIRTH_DATE in active_employees_df.columns and not active_employees_df.empty:
            eoy_date_for_age_calc = pd.Timestamp(year=current_sim_year, month=12, day=31)
            active_employees_df.loc[:, EMP_BIRTH_DATE] = pd.to_datetime(active_employees_df[EMP_BIRTH_DATE], errors='coerce')
            active_employees_df.loc[:, 'age_years'] = (eoy_date_for_age_calc - active_employees_df[EMP_BIRTH_DATE]).dt.days / 365.25
            avg_age_active = active_employees_df['age_years'].mean()
            
        participation_rate = (participant_count / eligible_count) if eligible_count > 0 else 0.0
        
        # Use refactored employment summary builder
        emp_summary = build_employment_status_summary(
            current_snapshot, event_log_for_year, current_sim_year
        )
        employment_status_summary_data.append(emp_summary)

        # Use refactored core summary builder
        core_summary = build_core_summary({
            'Projection Year': current_sim_year,
            'Active Headcount': active_headcount_eoy,
            'Eligible Count': eligible_count,
            'Participant Count': participant_count,
            'Total Employee Gross Compensation': total_employee_gross_compensation,
            'Total EE Contribution': total_ee_contribution_this_year,
            'Total ER Contribution': total_er_contribution_this_year,
            'Total Compensation': total_compensation_this_year,
            'Avg Compensation (Active)': avg_compensation_active,
            'Avg Age (Active)': avg_age_active,
            'Participation Rate': participation_rate,
            'Avg Deferral Rate (Participants)': avg_deferral_rate_participants,
        })
        summary_results_list.append(core_summary)

        # Use employment snapshot builder and filter prior terminations
        full_snapshot = build_employment_status_snapshot(
            current_snapshot, event_log_for_year, current_sim_year
        )
        # Patch: Guarantee retention of all employees who were active at any point during the year,
        # or who had a compensation or termination event in the current year, for EOY snapshot.
        snap = full_snapshot.copy()
        snap[EMP_TERM_DATE] = pd.to_datetime(snap.get(EMP_TERM_DATE, pd.NaT), errors='coerce')
        snap['term_year'] = snap[EMP_TERM_DATE].dt.year
        # Employees with compensation or termination events this year
        comp_ids = []
        term_ids = []
        if event_log_for_year is not None and 'event_type' in event_log_for_year.columns:
            comp_ids = event_log_for_year.loc[
                event_log_for_year['event_type'] == EVT_COMP,
                EVENT_EMP_ID
            ].unique().tolist()
            term_ids = event_log_for_year.loc[
                event_log_for_year['event_type'] == EVT_TERM,
                EVENT_EMP_ID
            ].unique().tolist()
        # Employees who were active at any point during the year
        active_ids_soy = set(current_snapshot[current_snapshot[EMP_ACTIVE]].index.tolist())
        # Employees who terminated this year
        terminated_this_year = snap['term_year'] == current_sim_year
        # Build mask: active EOY, or terminated this year, or had comp/term event this year, or were active SOY
        mask = (
            snap[EMP_ACTIVE] |
            terminated_this_year |
            snap[EMP_ID].isin(comp_ids) |
            snap[EMP_ID].isin(term_ids) |
            snap[EMP_ID].isin(active_ids_soy)
        )
        yearly_eoy_snapshots[current_sim_year] = snap.loc[mask].drop(columns=['term_year'])

        last_year_active_headcount = active_headcount_eoy

    final_eoy_snapshot = current_snapshot
    final_cumulative_event_log = current_cumulative_event_log
    summary_results_df = pd.DataFrame(summary_results_list)
    employment_status_summary_df = pd.DataFrame(employment_status_summary_data)

    logger.info("Projection engine run completed.")
    if not projection_sim_years:
        logger.info("No projection years were simulated.")
        logger.info(f"Initial Snapshot shape: {final_eoy_snapshot.shape}")
        logger.info(f"Initial Event Log shape: {final_cumulative_event_log.shape}")
    else:
        logger.info(f"Final EOY Snapshot ({projection_sim_years[-1]}) shape: {final_eoy_snapshot.shape}")
        logger.info(f"Final Cumulative Event Log shape: {final_cumulative_event_log.shape}")
    logger.info("Summary Results:")
    logger.info(f"\n{summary_results_df.to_string()}")

    return yearly_eoy_snapshots, final_eoy_snapshot, final_cumulative_event_log, summary_results_df, employment_status_summary_df
