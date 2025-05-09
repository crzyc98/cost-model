import pandas as pd
import numpy as np
from pathlib import Path
from types import SimpleNamespace
import logging
from datetime import datetime
import yaml
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# --- Core Simulation Engine ---
try:
    from cost_model.engines.run_one_year import run_one_year
    print("INFO: Successfully imported run_one_year.")
except ImportError as e:
    print(f"ERROR: Could not import run_one_year: {e}. Ensure cost_model is in PYTHONPATH.")
    # Define a dummy function if import fails, so the rest of the script can be parsed
    def run_one_year(year, prev_snapshot, event_log, hazard_table, rng, deterministic_term): # type: ignore
        print(f"DUMMY run_one_year called for year {year}")
        return prev_snapshot, event_log

# --- Rule Modules (ensure imports are correct) ---
from cost_model.plan_rules import eligibility_events

# --- State and Utility Imports ---
from cost_model.state import event_log as event_log_module # To avoid name collision
from cost_model.state import snapshot as snapshot_module
from cost_model.utils.columns import (
    EMP_ID, EMP_HIRE_DATE, EMP_BIRTH_DATE, EMP_ROLE, 
    EMP_GROSS_COMP, EMP_DEFERRAL_RATE, EMP_TERM_DATE
) 

# --- Event Log and Snapshot Schema --- (Importing constants from state module)
try:
    from cost_model.state.event_log import EVENT_COLS, EVENT_PANDAS_DTYPES, EMP_ID, EVT_HIRE, EVT_TERM
    from cost_model.state.snapshot import SNAPSHOT_COLS as SNAPSHOT_COL_NAMES, SNAPSHOT_DTYPES # Renamed to avoid conflict
    print("INFO: Successfully imported schema definitions from cost_model.state.")
except ImportError:
    print("WARNING: Could not import schema definitions from cost_model.state. Using placeholders.")
    EMP_ID = "employee_id"
    EVT_HIRE = "hire" # Fallback
    EVT_TERM = "term" # Fallback
    EVENT_COLS = [
        "event_id", "event_time", EMP_ID,
        "event_type", "value_num", "value_json", "meta"
    ]
    EVENT_PANDAS_DTYPES = {
        "event_id": "string", "event_time": "datetime64[ns]", EMP_ID: "string",
        "event_type": "string", "value_num": "float64", "value_json": "string", "meta": "string"
    }
    SNAPSHOT_COL_NAMES = [] # Fallback
    SNAPSHOT_DTYPES = {} # Fallback

# --- Event Types (Ensure these match your engine's expectations) ---
EVT_ELIGIBLE = "EVT_ELIGIBLE"
EVT_1YR = "EVT_1YR" 
EVT_ENROLL = "EVT_ENROLL"
EVT_AUTO_ENROLL = "EVT_AUTO_ENROLL"
EVT_OPT_OUT = "EVT_OPT_OUT"
EVT_CONTRIB_INCR = "EVT_CONTRIB_INCR"
EVT_PROACTIVE_DECREASE = "EVT_PROACTIVE_DECREASE"
EVT_COMP = "EVT_COMP"
EVT_HIRE = EVT_HIRE
EVT_TERM = EVT_TERM

# --- Setup Logging ---
LOG_DIR = Path("output_dev/projection_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG, # Changed to DEBUG for more detailed logs
    format="%(asctime)s [%(levelname)-8s] [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    handlers=[
        logging.StreamHandler(), # Output to console
        logging.FileHandler(LOG_DIR / "projection_run.log") # Output to file
    ]
)
logger = logging.getLogger(__name__)

# Helper function to recursively convert dicts to SimpleNamespace
def dict_to_simplenamespace(d):
    if isinstance(d, dict):
        # Convert non-string keys to strings for the current level's SimpleNamespace creation
        # and recursively call for nested dicts.
        kwargs = {str(k): dict_to_simplenamespace(v) for k, v in d.items()}
        return SimpleNamespace(**kwargs)
    elif isinstance(d, list):
        return [dict_to_simplenamespace(item) for item in d]
    return d

def create_initial_snapshot(start_year: int, census_path: str) -> pd.DataFrame:
    logger.info(f"Creating initial snapshot for start year: {start_year} from {census_path}")
    # Load census data
    try:
        census_df = pd.read_parquet(census_path)
    except Exception as e:
        logger.error(f"Error reading census Parquet file {census_path}: {e}")
        raise
    
    logger.info(f"Loaded census data with {len(census_df)} records. Columns: {census_df.columns.tolist()}")

    # Data for the snapshot at the beginning of the start_year
    # Assumes census_df columns match the imported constants from cost_model.utils.columns
    initial_data = {
        EMP_ID: census_df[EMP_ID],
        EMP_HIRE_DATE: pd.to_datetime(census_df[EMP_HIRE_DATE]), # Ensure datetime
        EMP_BIRTH_DATE: pd.to_datetime(census_df[EMP_BIRTH_DATE]), # Ensure datetime
        EMP_ROLE: census_df[EMP_ROLE],
        EMP_GROSS_COMP: census_df[EMP_GROSS_COMP],
        EMP_TERM_DATE: pd.NaT, # All are active initially in the snapshot logic
        'active': [True] * len(census_df), # 'active' is a snapshot-specific concept
        EMP_DEFERRAL_RATE: census_df[EMP_DEFERRAL_RATE],
        # Aliases, if engines expect these specific names from the snapshot
        # These might become redundant if snapshot_module.SNAPSHOT_COLS uses the constants
        'employee_birth_date': pd.to_datetime(census_df[EMP_BIRTH_DATE]), 
        'employee_hire_date': pd.to_datetime(census_df[EMP_HIRE_DATE]),
        'deferral_rate': census_df[EMP_DEFERRAL_RATE],
        'tenure_band': "all", # Usually calculated, placeholder for now. This needs to align with SNAPSHOT_COLS
    }
    snapshot_df = pd.DataFrame(initial_data)

    # Fill 'tenure_band' for all rows if it was set as a single string "all"
    if 'tenure_band' in snapshot_df.columns and isinstance(snapshot_df['tenure_band'].iloc[0], str) and len(snapshot_df) > 1:
        snapshot_df['tenure_band'] = [snapshot_df['tenure_band'].iloc[0]] * len(snapshot_df)

    # Add tenure_band (example calculation)
    # This is a simplified version; adjust logic as needed for your definition of tenure bands
    current_date_for_tenure = pd.Timestamp(f"{start_year}-01-01")
    snapshot_df['tenure_years'] = (current_date_for_tenure - snapshot_df[EMP_HIRE_DATE]).dt.days / 365.25
    bins = [0, 1, 3, 5, 10, float('inf')] # Example tenure bins (in years)
    labels = ['0-1yr', '1-3yrs', '3-5yrs', '5-10yrs', '10+yrs']
    snapshot_df['tenure_band'] = pd.cut(snapshot_df['tenure_years'], bins=bins, labels=labels, right=False)
    # snapshot_df = snapshot_df.drop(columns=['tenure_years']) # Optional: drop the intermediate calculation

    # Select and order columns according to SNAPSHOT_COL_NAMES and set EMP_ID as index
    # Ensure all columns in SNAPSHOT_COL_NAMES exist, fill missing with NaT/NaN if appropriate
    for col in SNAPSHOT_COL_NAMES:
        if col not in snapshot_df.columns:
            # Determine appropriate fill value based on dtype or general approach
            # For now, this indicates an issue if a defined column is missing after processing
            logger.warning(f"Snapshot column '{col}' defined in SNAPSHOT_COLS is missing. Will be NaN/NaT.")
            # snapshot_df[col] = pd.NA # Or more specific NaT/np.nan based on expected dtype

    # Filter to only include columns defined in SNAPSHOT_COL_NAMES (now including EMP_ID)
    snapshot_df = snapshot_df[[col for col in SNAPSHOT_COL_NAMES if col in snapshot_df.columns]]
    
    # Set EMP_ID as index but keep it as a column
    if EMP_ID in snapshot_df.columns:
        snapshot_df = snapshot_df.set_index(EMP_ID, drop=False)
    else:
        logger.error(f"{EMP_ID} column is missing before attempting to set it as index.")
        # Handle error appropriately, maybe raise an exception or return an empty/error DataFrame

    # Apply final dtypes
    for col, dtype in SNAPSHOT_DTYPES.items():
        if col in snapshot_df.columns:
            try:
                if str(dtype).lower() == 'boolean':
                    snapshot_df[col] = snapshot_df[col].astype(pd.BooleanDtype())
                elif col == EMP_ID and str(dtype).lower() == 'string': # EMP_ID is index and column
                     snapshot_df[col] = snapshot_df[col].astype(pd.StringDtype())
                elif col != EMP_ID : # Avoid re-casting index if it's also a column
                    snapshot_df[col] = snapshot_df[col].astype(dtype)
            except Exception as e:
                logger.error(f"Failed to cast snapshot column '{col}' to '{dtype}': {e}. Current type: {snapshot_df[col].dtype}")

    logger.info(f"Initial snapshot created with {len(snapshot_df)} records. Columns: {snapshot_df.columns.tolist()}")
    logger.debug(f"Initial snapshot dtypes:\n{snapshot_df.dtypes}")
    return snapshot_df

def create_initial_event_log(start_year: int) -> pd.DataFrame:
    logger.info(f"Creating initial event log for events up to start of year: {start_year}")
    # These are events that occurred *before* the first simulation year (2025)
    # E.g., deferral rates set in 2024 that are active on 2025-01-01
    base_event_data = [
        {"event_id": "prior_evt_B", "event_time": pd.Timestamp(f"{start_year-1}-01-01"), EMP_ID: "B", "event_type": EVT_CONTRIB_INCR, "value_num": 0.05},
        {"event_id": "prior_evt_C", "event_time": pd.Timestamp(f"{start_year-1}-01-01"), EMP_ID: "C", "event_type": EVT_CONTRIB_INCR, "value_num": 0.10},
    ]
    event_log_df = pd.DataFrame(base_event_data)

    # Ensure all EVENT_COLS are present and correctly typed
    for col in EVENT_COLS:
        if col not in event_log_df.columns:
            dtype = EVENT_PANDAS_DTYPES.get(col, 'object')
            logger.debug(f"Adding missing event log column: {col} with presumed dtype: {dtype}")
            if pd.api.types.is_datetime64_any_dtype(dtype): event_log_df[col] = pd.NaT
            elif str(dtype).lower() == 'boolean': event_log_df[col] = pd.NA # pd.BooleanDtype()
            elif pd.api.types.is_numeric_dtype(dtype): event_log_df[col] = np.nan # pd.Float64Dtype()
            else: event_log_df[col] = None # pd.StringDtype()
            
    event_log_df = event_log_df[EVENT_COLS] # Ensure column order
    for col, dtype_str in EVENT_PANDAS_DTYPES.items():
        if col in event_log_df.columns:
            try:
                if str(dtype_str).lower() == 'boolean':
                     event_log_df[col] = event_log_df[col].astype(pd.BooleanDtype())
                else:
                     event_log_df[col] = event_log_df[col].astype(dtype_str)
            except Exception as e:
                logger.error(f"Failed to cast event_log column '{col}' to '{dtype_str}': {e}. Current type: {event_log_df[col].dtype}")
    logger.info(f"Initial event log created with {len(event_log_df)} events.")
    logger.debug(f"Initial event log dtypes:\n{event_log_df.dtypes}")
    return event_log_df

# Main projection function
def run_5_year_projection_test(config_path: str, census_path: str):
    """
    Runs a 5-year projection test using a specified configuration and initial census.

    Args:
        config_path: Path to the YAML configuration file.
        census_path: Path to the Parquet file for the initial census.
    """
    # --- Load Configuration --- 
    logger.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {config_path}: {e}")
        return

    if not full_config or 'global_parameters' not in full_config or 'plan_rules' not in full_config:
        logger.error("Configuration file is missing required sections: 'global_parameters' or 'plan_rules'.")
        return

    global_params_dict = full_config.get('global_parameters', {})
    plan_rules_dict = full_config.get('plan_rules', {})

    # --- Parameters --- 
    # Use SimpleNamespace for global_params for now, can be refined
    global_params = SimpleNamespace(**global_params_dict)

    START_YEAR = getattr(global_params, 'start_year', 2025)
    PROJECTION_YEARS = getattr(global_params, 'projection_years', 5)
    RANDOM_SEED = getattr(global_params, 'random_seed', 42)
    # ... (other global_params can be accessed similarly or passed as a dict/SimpleNamespace)

    output_dir = Path(getattr(global_params, 'output_directory', "output_dev/projection_run_output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running {PROJECTION_YEARS}-year projection starting from {START_YEAR}.")
    logger.info(f"Random seed: {RANDOM_SEED}")
    logger.info(f"Output will be saved to: {output_dir}")

    # Initialize RNG
    rng = np.random.default_rng(RANDOM_SEED)

    # --- Plan Rules Configuration ---
    # This section will be expanded in the next step to use plan_rules_dict
    # For now, keeping the existing structure but it will be fed by YAML data
    if False: # PYDANTIC_MODELS_AVAILABLE
        logger.info("Attempting to use Pydantic models for PlanRules.")
        try:
            eligibility_rules_data = plan_rules_dict.get('eligibility', {})
            eligibility_rules = None # EligibilityRules(**eligibility_rules_data) if eligibility_rules_data else None

            auto_enrollment_data = plan_rules_dict.get('auto_enrollment', {})
            auto_enrollment_rules = None # AutoEnrollmentRules(**auto_enrollment_data) if auto_enrollment_data else None
            
            # Placeholder for other rule components from YAML
            # Example for contribution_increase (assuming it's a direct sub-key in plan_rules_dict)
            ci_config_data = plan_rules_dict.get('contribution_increase', {})
            contribution_increase_config = None # ContributionIncreaseRules(**ci_config_data) if ci_config_data else None

            pd_config_data = plan_rules_dict.get('proactive_decrease', {})
            proactive_decrease_config = None # ProactiveDecreaseRules(**pd_config_data) if pd_config_data else None
            
            plan_rules_config = None # PlanRules(
                # eligibility=eligibility_rules,
                # auto_enrollment=auto_enrollment_rules,
                # contribution_increase=contribution_increase_config, # Or however it's structured
                # proactive_decrease=proactive_decrease_config  # Or however it's structured
                # eligibility_events, etc. would be added here if present in Pydantic model and YAML
            # )
        except Exception as e:
            logger.error(f"Error instantiating Pydantic PlanRules: {e}. Falling back to SimpleNamespace.")
            # Fallback or re-raise as appropriate
            plan_rules_config = SimpleNamespace(**plan_rules_dict) # Basic fallback
    else:
        logger.info("Pydantic models not available, using SimpleNamespace for PlanRules.")
        # Convert the entire plan_rules_dict using the helper function
        plan_rules_config = dict_to_simplenamespace(plan_rules_dict)
        # Ensure nested structures are also SimpleNamespaces if needed by downstream code
        # for key, value in plan_rules_dict.items():
        #     if isinstance(value, dict):
        #         setattr(plan_rules_config, key, SimpleNamespace(**value))

    # TODO: Ensure all necessary sub-attributes (eligibility_rules, etc.) are properly assigned
    # to plan_rules_config if using SimpleNamespace directly for the whole thing.
    # The current Pydantic block attempts this more granularly.

    # Ensure plan_rules_config and global_params have expected attributes
    # This is where you would ensure, e.g. plan_rules_config.eligibility exists and has its own attributes
    # This part might need more robust handling based on your config structure

    # Create initial state
    current_snapshot = create_initial_snapshot(START_YEAR, census_path)
    current_event_log = create_initial_event_log(START_YEAR)

    # 2) Define PlanRules configuration is handled by plan_rules_config from YAML parsing

    # 3) Build hazard table using global rates for all role/tenure combinations
    years = list(range(START_YEAR, START_YEAR + PROJECTION_YEARS))
    
    # Get unique role and tenure combinations from the initial snapshot
    if EMP_ROLE in current_snapshot.columns and 'tenure_band' in current_snapshot.columns:
        unique_roles_tenures = current_snapshot[[EMP_ROLE, 'tenure_band']].drop_duplicates().to_dict('records')
    else:
        logger.warning(f"'{EMP_ROLE}' or 'tenure_band' not in initial snapshot. Using a default 'all'/'all' for hazard table.")
        unique_roles_tenures = [{EMP_ROLE: 'all', 'tenure_band': 'all'}]

    # Get global rates from config
    # Ensure default values if not present, though ideally schema validation would handle this.
    global_term_rate = getattr(global_params, 'annual_termination_rate', 0.10) 
    global_growth_rate = getattr(global_params, 'annual_growth_rate', 0.05)
    global_comp_raise_pct = getattr(global_params, 'annual_compensation_increase_rate', 0.03)

    logger.info(f"Using global rates for hazard table: Term Rate={global_term_rate}, Growth Rate={global_growth_rate}, Comp Raise Pct={global_comp_raise_pct}")

    all_hazard_data_for_years = []
    for year_val in years:
        for role_tenure_combo in unique_roles_tenures:
            all_hazard_data_for_years.append({
                'year': year_val,
                EMP_ROLE: role_tenure_combo[EMP_ROLE],
                'tenure_band': role_tenure_combo['tenure_band'],
                'term_rate': global_term_rate,
                'growth_rate': global_growth_rate,
                'comp_raise_pct': global_comp_raise_pct,
                'cfg': plan_rules_config # Pass the scenario config
            })
    
    if all_hazard_data_for_years:
        hazard_table = pd.DataFrame(all_hazard_data_for_years)
        logger.info(f"Hazard table constructed for {len(unique_roles_tenures)} role/tenure combinations across {PROJECTION_YEARS} years.")
    else:
        logger.warning("Could not generate any hazard data. Using an empty hazard table.")
        expected_cols = ['year', EMP_ROLE, 'tenure_band', 'term_rate', 'comp_raise_pct', 'growth_rate', 'cfg']
        hazard_table = pd.DataFrame(columns=expected_cols)

    # 4) Iterate through each year, running the simulation
    results = []
    yearly_snapshots_dict = {} # Store EOY snapshots

    for yr_idx, current_sim_year in enumerate(years):
        logger.info(f"--- Simulating Year {current_sim_year} (Index {yr_idx}) ---")
        logger.debug(f"SOY {current_sim_year} - Snapshot shape: {current_snapshot.shape}, Active: {current_snapshot['active'].sum()}")
        # logger.debug(f"Start of year {current_sim_year} - Snapshot head:\n{current_snapshot.head().to_string()}")
        # logger.debug(f"Start of year {current_sim_year} - Event log tail before run:\n{current_event_log.tail().to_string()}")

        # Prepare hazard slice for the current year
        hazard_slice_for_year = hazard_table[hazard_table["year"] == current_sim_year]
        if hazard_slice_for_year.empty:
            logger.error(f"Hazard slice for year {current_sim_year} is EMPTY. This will likely cause issues.")
            # Potentially create a default slice or handle error, for now, log and continue
            # This branch should ideally not be hit if hazard_table is populated correctly
        else:
            logger.debug(f"Hazard slice for {current_sim_year}:\n{hazard_slice_for_year.to_string()}")

        # run_one_year returns: next_year_snapshot_at_eoy, all_events_up_to_eoy
        eoy_snapshot, cumulative_event_log = run_one_year(
            year=current_sim_year,
            prev_snapshot=current_snapshot, # Snapshot at start of current_sim_year
            event_log=current_event_log,            # All events up to start of current_sim_year
            hazard_table=hazard_table,      # Full hazard table (run_one_year will slice it)
            rng=rng,
            deterministic_term=False        # Use False for stochastic behavior
        )
        
        logger.info(f"EOY {current_sim_year} - Snapshot shape: {eoy_snapshot.shape}, Active: {eoy_snapshot['active'].sum()}")
        # logger.info(f"End of year {current_sim_year} - EOY Snapshot shape: {eoy_snapshot.shape}")
        # logger.info(f"End of year {current_sim_year} - Cumulative Event log shape: {cumulative_event_log.shape}")

        # Store results for summary
        # Calculate active headcount from the EOY snapshot for *this* year
        active_headcount_eoy_current_year = eoy_snapshot[eoy_snapshot["active"]].shape[0]
        # logger.debug(f"EOY {current_sim_year} - Active headcount based on 'active' column: {active_headcount_eoy_current_year}")
        # active_headcount_eoy_current_year = eoy_snapshot[eoy_snapshot[EMP_TERM_DATE].isna()].shape[0]
        
        # Calculate terminations *during* the current year by filtering the cumulative event log
        # These events were generated *within* the run_one_year call for current_sim_year
        term_events_in_current_year_df = cumulative_event_log[
            (cumulative_event_log['event_type'] == EVT_TERM) &
            (cumulative_event_log['event_time'].dt.year == current_sim_year)
        ]
        terminations_count_this_year = len(term_events_in_current_year_df)

        # Calculate new hires *during* the current year
        new_hires_in_current_year_df = cumulative_event_log[
            (cumulative_event_log['event_type'] == EVT_HIRE) &
            (cumulative_event_log['event_time'].dt.year == current_sim_year)
        ]
        new_hires_count_this_year = new_hires_in_current_year_df[EMP_ID].nunique()


        results.append({
            "year": current_sim_year,
            "eoy_active_headcount": active_headcount_eoy_current_year, # Use EOY active count for this year
            "new_hires_made": new_hires_count_this_year, # Use count from this year's events
            "terminations_during_year": terminations_count_this_year, # Use count from this year's events
        })
        
        yearly_snapshots_dict[current_sim_year] = eoy_snapshot.copy() # Optional: if you need all EOY snapshots later

        # Prepare for next iteration: EOY state becomes SOY state for next year
        current_snapshot = eoy_snapshot.copy()
        current_event_log = cumulative_event_log.copy()

    # --- Save Results and Plot --- #
    proj_summary_df = pd.DataFrame(results)
    logger.info("\n--- Projection Summary ---")
    print(proj_summary_df.to_string())

    # Save summary and final event log
    proj_summary_df.to_csv(output_dir / "projection_summary.csv", index=False)
    cumulative_event_log.to_parquet(output_dir / "final_event_log.parquet", index=False)
    logger.info(f"Projection summary and final event log saved to: {output_dir}")

    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('EOY Active Headcount', color=color)
    ax1.plot(proj_summary_df['year'], proj_summary_df['eoy_active_headcount'], color=color, marker='o', linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, axis='y', linestyle=':')

    # Create a second y-axis for growth rate
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Annual Growth Rate (%)', color=color)
    # Calculate growth rate: (current_active - prev_active) / prev_active
    proj_summary_df['growth_rate_pct'] = proj_summary_df['eoy_active_headcount'].pct_change() * 100
    ax2.plot(proj_summary_df['year'], proj_summary_df['growth_rate_pct'], color=color, marker='x', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0)) # xmax=1.0 if rates are 0.05, xmax=100.0 if 5.0

    fig.tight_layout() # otherwise the right y-label is slightly clipped
    plt.title('Headcount Projection and Growth Rate Over 5 Years')
    plot_path = output_dir / "headcount_projection.png"
    plt.savefig(plot_path)
    logger.info(f"Projection plot saved to: {plot_path}")
    # plt.show() # Uncomment to display plot if running interactively

if __name__ == '__main__':
    # Define paths to your configuration and census files
    config_file_path = "config/dev_tiny.yaml"
    census_file_path = "data/census_preprocessed.parquet"
    
    # Ensure paths are relative to the project root or provide absolute paths
    # Assuming the script is run from the project root directory
    project_root = Path(__file__).resolve().parent.parent # Adjust if script is nested deeper
    
    run_5_year_projection_test(str(project_root / config_file_path), 
                               str(project_root / census_file_path))
