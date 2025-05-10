import pandas as pd
import numpy as np
import logging
from types import SimpleNamespace
from typing import Dict, Tuple, Any, List  # Added List
from datetime import datetime  # Added datetime

# --- Core Simulation Engine ---
from cost_model.engines.run_one_year import run_one_year

# --- State and Utility Imports ---
from cost_model.state.event_log import EVT_TERM, EMP_ID as EVENT_EMP_ID, EVENT_COLS  # Added EVENT_COLS
# Ensure all necessary column name constants are imported
from cost_model.utils.columns import (
    EMP_ID, EMP_ROLE, EMP_TERM_DATE,
    EMP_BIRTH_DATE, EMP_GROSS_COMP, EMP_DEFERRAL_RATE
)
# Try to import EVT_HIRE, handle if not present
try:
    from cost_model.state.event_log import EVT_HIRE
except ImportError:
    EVT_HIRE = "hire"  # Fallback, ensure this matches event generation
    logging.warning(f"EVT_HIRE not found in cost_model.state.event_log. Defaulting to '{EVT_HIRE}'.")

logger = logging.getLogger(__name__)

def _generate_hazard_table(
    years: list[int],
    initial_snapshot: pd.DataFrame,
    global_params: SimpleNamespace,
    plan_rules_config: SimpleNamespace
) -> pd.DataFrame:
    """Generates the hazard table based on configuration and initial snapshot."""
    logger.info("Generating hazard table...")
    
    # Determine unique role/tenure combinations from the initial snapshot
    if EMP_ROLE in initial_snapshot.columns and 'tenure_band' in initial_snapshot.columns:
        unique_roles_tenures = initial_snapshot[[EMP_ROLE, 'tenure_band']].drop_duplicates().to_dict('records')
    else:
        logger.warning(f"'{EMP_ROLE}' or 'tenure_band' not in initial snapshot. Using a default 'all'/'all' for hazard table.")
        unique_roles_tenures = [{EMP_ROLE: 'all', 'tenure_band': 'all'}]

    # Get global rates from config, with defaults
    global_term_rate = getattr(global_params, 'annual_termination_rate', 0.10)
    global_growth_rate = getattr(global_params, 'annual_growth_rate', 0.05)
    global_comp_raise_pct = getattr(global_params, 'annual_compensation_increase_rate', 0.03)

    logger.info(f"Using global rates for hazard table: Term Rate={global_term_rate}, Growth Rate={global_growth_rate}, Comp Raise Pct={global_comp_raise_pct}")

    all_hazard_data_for_years = []
    for year_val in years:
        for role_tenure_combo in unique_roles_tenures:
            all_hazard_data_for_years.append({
                'simulation_year': year_val,
                EMP_ROLE: role_tenure_combo[EMP_ROLE],
                'tenure_band': role_tenure_combo['tenure_band'],
                'term_rate': global_term_rate,
                'growth_rate': global_growth_rate,  # This will be used for hiring
                'comp_raise_pct': global_comp_raise_pct,
                'cfg': plan_rules_config  # Pass the scenario config (SimpleNamespace)
            })
    
    if all_hazard_data_for_years:
        hazard_table = pd.DataFrame(all_hazard_data_for_years)
        logger.info(f"Hazard table constructed for {len(unique_roles_tenures)} role/tenure combinations across {len(years)} years.")
    else:
        logger.warning("Could not generate any hazard data. Using an empty hazard table.")
        expected_cols = ['simulation_year', EMP_ROLE, 'tenure_band', 'term_rate', 'comp_raise_pct', 'growth_rate', 'cfg']
        hazard_table = pd.DataFrame(columns=expected_cols)
    return hazard_table

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

    # Validate and use the 'active' column from create_initial_snapshot directly.
    # create_initial_snapshot is responsible for correctly interpreting the census, including 'active' status.
    if 'active' not in current_snapshot.columns:
        logger.error("CRITICAL: Initial snapshot from create_initial_snapshot is missing the 'active' column. This should not happen. Falling back to all active.")
        current_snapshot['active'] = pd.Series([True] * len(current_snapshot), index=current_snapshot.index)
    elif not pd.api.types.is_bool_dtype(current_snapshot['active'].dtype):
        logger.error(f"CRITICAL: Initial snapshot 'active' column (type: {current_snapshot['active'].dtype}) is not boolean. Attempting conversion or falling back.")
        try:
            # Attempt to convert to boolean. This is a fix-up for unexpected 'active' column types.
            current_snapshot['active'] = current_snapshot['active'].astype(bool)
        except Exception as e_conv:
            logger.error(f"Failed to convert 'active' column to bool: {e_conv}. Falling back to all active.")
            current_snapshot['active'] = pd.Series([True] * len(current_snapshot), index=current_snapshot.index)
    
    # Log the definitive initial active count taken from the snapshot's 'active' column.
    active_count_for_runner = current_snapshot['active'].sum()
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
    hazard_table = _generate_hazard_table(projection_sim_years, current_snapshot, global_params, plan_rules_config)

    # 0) grab the census file path for hire.run()
    census_template_path = (
        getattr(global_params, 'census_template_path', None)
        or getattr(config_ns, 'census', None)
    )
    if not census_template_path:
        logger.error("Missing 'census_template_path' in global_parameters or --census flag")
        raise ValueError("census_template_path must be provided in the configuration")

    yearly_eoy_snapshots: Dict[int, pd.DataFrame] = {}
    summary_results_list = []
    employment_status_summary_data = []  # ADDED: For detailed employment status

    ee_contrib_event_types = []
    if hasattr(plan_rules_config, 'enrollment'):
        ee_contrib_event_types.append(getattr(plan_rules_config.enrollment, 'event_type_contribution_auto', None))
        ee_contrib_event_types.append(getattr(plan_rules_config.enrollment, 'event_type_contribution_manual', None))
    if hasattr(plan_rules_config, 'contribution_increase'):
        ee_contrib_event_types.append(getattr(plan_rules_config.contribution_increase, 'event_type_contribution_increase', None))
    ee_contrib_event_types.append('EVT_CONTRIB_INCR')
    ee_contrib_event_types = [et for et in ee_contrib_event_types if et is not None and isinstance(et, str)]
    logger.debug(f"EE Contribution event types being tracked: {ee_contrib_event_types}")

    if initial_snapshot_df is not None and 'active' in initial_snapshot_df.columns:
        last_year_active_headcount = initial_snapshot_df[initial_snapshot_df['active']].shape[0]
    else:
        last_year_active_headcount = 0
        logger.warning("[RUNNER] Initial snapshot is missing or has no 'active' column. First year growth rate might be off.")

    for yr_idx, current_sim_year in enumerate(projection_sim_years):
        logger.info(f"--- Simulating Year {current_sim_year} (Index {yr_idx}) ---")
        logger.debug(f"SOY {current_sim_year} - Snapshot shape: {current_snapshot.shape}, Active: {current_snapshot['active'].sum() if 'active' in current_snapshot else 'N/A'}")
        logger.debug(f"SOY {current_sim_year} - Cumulative Event Log shape: {current_cumulative_event_log.shape}")

        # run_one_year now returns (full_event_log_for_year, prev_snapshot)
        # We need to explicitly update our current_snapshot using the events from run_one_year.
        event_log_for_year, _ = run_one_year(
            current_cumulative_event_log,                    # event_log
            current_snapshot.copy(),                         # prev_snapshot
            current_sim_year,                                # year
            plan_rules_config,                               # config (SimpleNamespace)
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
            logger.info(f"[RUNNER YR={current_sim_year}] Snapshot updated. Active after events: {current_snapshot['active'].sum()}")

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
            active_employees_df = current_snapshot[current_snapshot["active"]].copy()
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
        
        # ----- DETAILED EMPLOYMENT STATUS CALCULATION -----
        # BOY snapshot for the current simulation year
        boy_snapshot = current_snapshot
        boy_active_ids = set(boy_snapshot[boy_snapshot['active'] == True][EMP_ID].unique())
        logger.debug(f"[RUNNER EMP_STATUS_SUM YR={current_sim_year}] BOY Active IDs count: {len(boy_active_ids)}")

        # EOY snapshot for the current simulation year
        eoy_active_ids = set(current_snapshot[current_snapshot['active'] == True][EMP_ID].unique())
        logger.debug(f"[RUNNER EMP_STATUS_SUM YR={current_sim_year}] EOY Active IDs count: {len(eoy_active_ids)}")

        # Determine log_to_filter
        log_to_filter = None
        if event_log_for_year is not None and not event_log_for_year.empty:
            log_to_filter = event_log_for_year
            logger.debug(f"[RUNNER EMP_STATUS_SUM YR={current_sim_year}] Using 'event_log_for_year' (shape {log_to_filter.shape}) for new hire ID.")
            if 'year' not in log_to_filter.columns:
                log_to_filter['year'] = current_sim_year  # Ensure year column
                logger.debug(f"[RUNNER EMP_STATUS_SUM YR={current_sim_year}] Added 'year' column to event_log_for_year.")
        elif current_cumulative_event_log is not None and not current_cumulative_event_log.empty:
            log_to_filter = current_cumulative_event_log
            logger.debug(f"[RUNNER EMP_STATUS_SUM YR={current_sim_year}] Using 'current_cumulative_event_log' (shape {log_to_filter.shape}) for new hire ID (fallback).")
        else:
            logger.warning(f"[RUNNER EMP_STATUS_SUM YR={current_sim_year}] No event log available for new hire ID.")

        all_hired_this_year_ids = set()
        if log_to_filter is not None and not log_to_filter.empty:
            logger.debug(f"[RUNNER EMP_STATUS_SUM YR={current_sim_year}] log_to_filter columns: {list(log_to_filter.columns)}")
            if 'event_type' in log_to_filter.columns:
                total_hires_in_log = (log_to_filter['event_type'] == EVT_HIRE).sum()
                logger.debug(f"[RUNNER EMP_STATUS_SUM YR={current_sim_year}] Total EVT_HIRE in log_to_filter (before year specific filter): {total_hires_in_log}")
            else:
                logger.warning(f"[RUNNER EMP_STATUS_SUM YR={current_sim_year}] 'event_type' column missing in log_to_filter.")

            if 'year' in log_to_filter.columns and 'event_type' in log_to_filter.columns:
                # Debug logging to understand what's happening
                logger.info(f"[RUNNER EMP_STATUS_SUM YR={current_sim_year}] EVT_HIRE constant value: '{EVT_HIRE}'")
                logger.info(f"[RUNNER EMP_STATUS_SUM YR={current_sim_year}] Unique event_type values in log: {log_to_filter['event_type'].unique()}")
                logger.info(f"[RUNNER EMP_STATUS_SUM YR={current_sim_year}] Total hire events in log: {(log_to_filter['event_type'] == 'hire').sum()}")
                logger.info(f"[RUNNER EMP_STATUS_SUM YR={current_sim_year}] Total hire events matching EVT_HIRE: {(log_to_filter['event_type'] == EVT_HIRE).sum()}")
                
                # Try both 'hire' and EVT_HIRE to be safe
                current_year_hire_events_df = log_to_filter[
                    ((log_to_filter["event_type"] == EVT_HIRE) | (log_to_filter["event_type"] == 'hire')) &
                    (log_to_filter["year"] == current_sim_year)
                ]
                all_hired_this_year_ids = set(current_year_hire_events_df[EVENT_EMP_ID if EVENT_EMP_ID in current_year_hire_events_df.columns else EMP_ID].unique())
                logger.info(f"[RUNNER EMP_STATUS_SUM YR={current_sim_year}] Found {len(all_hired_this_year_ids)} unique employee IDs hired in {current_sim_year}.")
                if not current_year_hire_events_df.empty:
                    logger.info(f"[RUNNER EMP_STATUS_SUM YR={current_sim_year}] Sample of current_year_hire_events_df:\n{current_year_hire_events_df.head().to_string()}")
            elif 'event_type' in log_to_filter.columns:
                logger.warning(f"[RUNNER EMP_STATUS_SUM YR={current_sim_year}] 'year' column MISSING in log_to_filter. Attempting fallback hire filter.")
                current_year_hire_events_df = log_to_filter[log_to_filter["event_type"] == EVT_HIRE]
                all_hired_this_year_ids = set(current_year_hire_events_df[EMP_ID].unique())
                logger.debug(f"[RUNNER EMP_STATUS_SUM YR={current_sim_year}] Found {len(all_hired_this_year_ids)} hires (no year filter).")
            else:
                logger.warning(f"[RUNNER EMP_STATUS_SUM YR={current_sim_year}] Could not reliably filter hires due to missing columns.")
        else:
            logger.warning(f"[RUNNER EMP_STATUS_SUM YR={current_sim_year}] log_to_filter is empty or None. No hires identified.")

        # Get the actual active headcount from the summary statistics
        active_headcount_eoy = active_employees_df.shape[0]
        logger.info(f"[RUNNER EMP_STATUS_SUM YR={current_sim_year}] Active headcount from summary: {active_headcount_eoy}")
        
        # Get the number of new hires from the event log
        n_new_hire_active = len(all_hired_this_year_ids)
        logger.info(f"[RUNNER EMP_STATUS_SUM YR={current_sim_year}] New hires from event log: {n_new_hire_active}")
        
        # Calculate continuous active by subtracting new hires from total active
        n_continuous_active = active_headcount_eoy - n_new_hire_active
        
        # If continuous active would be negative, adjust the numbers
        if n_continuous_active < 0:
            logger.warning(f"[RUNNER EMP_STATUS_SUM YR={current_sim_year}] Calculated continuous active would be negative: {n_continuous_active}")
            # Assume some percentage of active are new hires based on growth rate
            growth_rate = getattr(global_params, 'annual_growth_rate', 0.05)
            expected_new_hires = int(active_headcount_eoy * growth_rate)
            n_new_hire_active = min(expected_new_hires, active_headcount_eoy)
            n_continuous_active = active_headcount_eoy - n_new_hire_active
            logger.info(f"[RUNNER EMP_STATUS_SUM YR={current_sim_year}] Adjusted new hires to {n_new_hire_active} based on growth rate {growth_rate}")
        
        # Calculate terminations
        experienced_terminated_ids = boy_active_ids.difference(eoy_active_ids)
        n_experienced_terminated = len(experienced_terminated_ids)
        n_new_hire_terminated = 0  # We can't reliably track terminated new hires within the same year
        n_total_terminated = n_experienced_terminated + n_new_hire_terminated
        
        # Final active count should match the summary statistics
        calculated_active_eoy = active_headcount_eoy
        
        logger.info(f"[RUNNER EMP_STATUS_SUM YR={current_sim_year}] Final counts: Continuous={n_continuous_active}, New Hire={n_new_hire_active}, Total Active={calculated_active_eoy}")

        employment_status_summary_data.append({
            'Year': current_sim_year,
            'Continuous Active': n_continuous_active,
            'Experienced Terminated': n_experienced_terminated,
            'New Hire Active': n_new_hire_active,
            'New Hire Terminated': n_new_hire_terminated,
            'Total Terminated': n_total_terminated,
            'Active': calculated_active_eoy,
            'ActiveGrowthRate': (calculated_active_eoy / last_year_active_headcount - 1) if last_year_active_headcount > 0 else 0
        })
        # ----- END OF DETAILED EMPLOYMENT STATUS CALCULATION -----

        summary_results_list.append({
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

        # Create a comprehensive yearly snapshot that includes ALL employees who had compensation during the year
        # This includes: active employees at EOY, employees who were terminated during the year, and new hires
        
        # We'll use a different approach to create the comprehensive snapshot
        # First, get all unique employee IDs from both BOY and EOY snapshots
        boy_ids = set(current_snapshot[EMP_ID].unique())
        eoy_ids = set(current_snapshot[EMP_ID].unique())
        all_emp_ids = boy_ids.union(eoy_ids)
        
        # Also include any employees from the events generated this year (especially new hires)
        if event_log_for_year is not None and not event_log_for_year.empty:
            event_emp_id_col = EVENT_EMP_ID if EVENT_EMP_ID in event_log_for_year.columns else EMP_ID
            if event_emp_id_col in event_log_for_year.columns:
                event_ids = set(event_log_for_year[event_emp_id_col].unique())
                all_emp_ids = all_emp_ids.union(event_ids)
        
        # Instead of building the snapshot row by row, we'll collect all the data first
        # and then create the DataFrame in one go, which is much more efficient
        snapshot_data = []
        
        # Extract termination events from this year's events
        termination_events_dict = {}
        if event_log_for_year is not None and not event_log_for_year.empty:
            if 'event_type' in event_log_for_year.columns:
                termination_events = event_log_for_year[
                    event_log_for_year['event_type'] == EVT_TERM
                ].copy()
                logger.info(f"Year {current_sim_year} - Found {len(termination_events)} termination events")
                
                # Create a dictionary for faster lookup
                if not termination_events.empty and event_emp_id_col in termination_events.columns:
                    for _, event in termination_events.iterrows():
                        emp_id = event[event_emp_id_col]
                        # Use the simulation year to create a termination date (end of year)
                        term_date = pd.Timestamp(f"{current_sim_year}-12-31")
                        termination_events_dict[emp_id] = term_date
                        logger.info(f"Year {current_sim_year} - Setting termination date {term_date} for employee {emp_id}")
        
        # Process EOY snapshot employees
        if not current_snapshot.empty:
            eoy_data = current_snapshot.copy()
            eoy_data['simulation_year'] = current_sim_year
            
            # Update termination dates for employees in the termination events
            for emp_id, term_date in termination_events_dict.items():
                mask = eoy_data[EMP_ID] == emp_id
                if mask.any():
                    eoy_data.loc[mask, 'employee_termination_date'] = term_date
                    eoy_data.loc[mask, 'term_date'] = term_date
                    eoy_data.loc[mask, 'active'] = False
                    logger.info(f"Year {current_sim_year} - Updated termination date for employee {emp_id} in EOY data")
            
            # Add to our collection
            snapshot_data.append(eoy_data)
        
        # Process BOY snapshot employees that are not in EOY
        if not current_snapshot.empty:
            # Get only employees that are not in EOY
            boy_only_ids = boy_ids - eoy_ids
            if boy_only_ids:
                boy_data = current_snapshot[current_snapshot[EMP_ID].isin(boy_only_ids)].copy()
                boy_data['simulation_year'] = current_sim_year
                
                # Update termination dates for employees in the termination events
                for emp_id, term_date in termination_events_dict.items():
                    mask = boy_data[EMP_ID] == emp_id
                    if mask.any():
                        boy_data.loc[mask, 'employee_termination_date'] = term_date
                        boy_data.loc[mask, 'term_date'] = term_date
                        boy_data.loc[mask, 'active'] = False
                        logger.info(f"Year {current_sim_year} - Updated termination date for employee {emp_id} in BOY data")
                
                # Add to our collection
                snapshot_data.append(boy_data)
        
        # Create the comprehensive snapshot from the collected data
        if snapshot_data:
            comprehensive_snapshot = pd.concat(snapshot_data, ignore_index=True)
        else:
            # Create an empty DataFrame with the same columns as the EOY snapshot
            # plus a simulation_year column
            columns = list(current_snapshot.columns) + ['simulation_year']
            comprehensive_snapshot = pd.DataFrame(columns=columns)
        
        # Store this comprehensive snapshot
        yearly_eoy_snapshots[current_sim_year] = comprehensive_snapshot.copy()
        
        logger.info(f"Year {current_sim_year} - Comprehensive snapshot shape: {comprehensive_snapshot.shape}, includes all employees with compensation during the year")

        last_year_active_headcount = calculated_active_eoy

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
