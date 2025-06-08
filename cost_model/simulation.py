# cost_model/simulation.py
"""
Main simulation orchestration module.

This module defines the high-level function to run a single simulation scenario,
calling components for configuration, data I/O, population dynamics, plan rules,
and reporting.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# --- Core Model Components --- #
# Configuration Loading & Access
from cost_model.config.models import MainConfig, GlobalParameters # Needed for type hints
from cost_model.config.accessors import get_scenario_config # Import the actual function
from cost_model.utils.columns import EMP_ID, EMP_TENURE_BAND

# Age calculation imports
from cost_model.state.age import apply_age
from cost_model.state.schema import EMP_BIRTH_DATE, EMP_AGE, EMP_AGE_BAND

# Data I/O
# TODO: Update these imports if readers/writers structure changes
try:
    from cost_model.data.readers import read_census_data # Import the specific function
    from cost_model.data.writers import write_snapshots, write_summary_metrics
except ImportError as e:
    print(f"Error importing data components: {e}")
    # Define placeholders if needed for static analysis, though runtime will fail
    def read_census_data(*args, **kwargs): raise ImportError("read_census_data missing")
    def write_snapshots(*args, **kwargs): raise ImportError("write_snapshots missing")
    def write_summary_metrics(*args, **kwargs): raise ImportError("write_summary_metrics missing")

# Simulation Engines
try:
    from cost_model.dynamics.engine import run_dynamics_for_year
    from cost_model.rules.engine import apply_rules_for_year # Corrected function name
except ImportError as e:
    print(f"Error importing simulation components: {e}")
    def run_dynamics_for_year(*args, **kwargs): raise ImportError("run_dynamics_for_year missing")
    def apply_rules_for_year(*args, **kwargs): raise ImportError("apply_rules_for_year missing")

# Reporting
try:
    from cost_model.reporting.metrics import calculate_summary_metrics
except ImportError as e:
    print(f"Error importing reporting components: {e}")
    # Temporarily define a placeholder if calculate_summary_metrics is missing
    def calculate_summary_metrics(*args, **kwargs):
        print("Warning: calculate_summary_metrics not implemented yet.")
        return pd.DataFrame() # Return empty DataFrame

# Import run_one_year for simulation loop
def _import_run_one_year():
    try:
        from cost_model.engines.run_one_year import run_one_year
        return run_one_year
    except ImportError:
        raise ImportError("run_one_year could not be imported from cost_model.engines.run_one_year")
run_one_year = _import_run_one_year()

# Utilities

logger = logging.getLogger(__name__)

def run_simulation(
    main_config: MainConfig,
    scenario_name: str,
    input_census_path: Path,
    output_dir_base: Path,
    # Optional flags to control output
    save_detailed_snapshots: bool = True,
    save_summary_metrics: bool = True,
    random_seed: Optional[int] = None # Added explicit seed argument
) -> None:
    """
    Runs the full simulation for a single specified scenario.

    Args:
        main_config: The validated MainConfig object containing all configurations.
        scenario_name: The name of the scenario to run.
        input_census_path: Path to the initial census data file (CSV or Parquet).
        output_dir_base: The base directory where scenario-specific output folders
                         will be created.
        save_detailed_snapshots: If True, save the full agent-level DataFrame for each year.
        save_summary_metrics: If True, calculate and save summary metrics for the scenario.
        random_seed: Optional random seed to use, overriding the one in the scenario config if provided.

    Raises:
        KeyError: If the scenario_name is not found in main_config.
        FileNotFoundError: If the input_census_path does not exist.
        Exception: For errors during simulation steps (dynamics, rules, etc.).
    """
    logger.info(f"===== Starting Simulation for Scenario: '{scenario_name}' =====")

    # Clear compensation warning cache for this new simulation run
    try:
        from cost_model.engines.comp import clear_warning_cache
        clear_warning_cache()
        logger.debug("Cleared compensation conflict warning cache for new simulation")
    except ImportError:
        # Gracefully handle if the function doesn't exist in older versions
        pass

    # 1. Get Resolved Scenario Configuration
    try:
        # Use the imported get_scenario_config function
        scenario_cfg: GlobalParameters = get_scenario_config(main_config, scenario_name)
    except (KeyError, ValueError) as e:
        logger.error(f"Error getting or validating scenario config for '{scenario_name}': {e}")
        # Log the exception details if validation failed within get_scenario_config
        if isinstance(e, ValueError):
             logger.exception("Configuration validation failed.")
        return # Stop simulation if config is invalid
    except Exception:
        # Catch any other unexpected errors during config resolution
        logger.exception(f"Unexpected error getting scenario config for '{scenario_name}'.")
        return

    # 1.5. Initialize job levels from main config
    try:
        from cost_model.state.job_levels import init_job_levels
        # Convert main_config to dict to extract job_levels
        if hasattr(main_config, 'model_dump'):
            config_dict = main_config.model_dump()
        elif hasattr(main_config, 'dict'):
            config_dict = main_config.dict()
        else:
            config_dict = main_config.__dict__
        init_job_levels(config_dict=config_dict, reset_warnings=True)
        logger.info("Job levels initialized from main configuration")
    except Exception as e:
        logger.warning(f"Failed to initialize job levels from config, using defaults: {e}")
        # Continue with defaults - this is not a fatal error

    # Extract key parameters and setup RNG after successful config load
    try:
        start_year = scenario_cfg.start_year
        projection_years = scenario_cfg.projection_years
        # Use provided seed if available, otherwise use seed from config
        seed = random_seed if random_seed is not None else scenario_cfg.random_seed
        scenario_output_name = main_config.scenarios[scenario_name].name or scenario_name # Use defined name or key
    except Exception:
        logger.exception(f"Error extracting key parameters from scenario config for '{scenario_name}'.")
        return

    # 2. Setup RNG
    seed = random_seed if random_seed is not None else scenario_cfg.random_seed or 42
    rng = np.random.default_rng(seed)
    logger.info(f"Using RNG seed: {seed}")

    # 3. Bootstrap events/snapshot from census
    # Bootstrap events (empty for now, or could seed with hires)
    events = pd.DataFrame()
    # Build snapshot from census data
    from cost_model.projections.snapshot import create_initial_snapshot
    snap = create_initial_snapshot(start_year, input_census_path)

    # 4. Build dynamic hazard table from global_params
    logger.info("Dynamically building hazard table from global_params...")

    # Import dynamic hazard table builder
    from cost_model.projections.dynamic_hazard import build_dynamic_hazard_table

    # Define standard job levels and tenure bands
    # These match the standard values used across the codebase
    standard_job_levels = [0, 1, 2, 3, 4]  # Numeric job levels
    standard_tenure_bands = ["<1", "1-3", "3-5", "5-10", "10-15", "15+"]  # From tenure.py

    # Generate simulation years for the hazard table
    simulation_years = list(range(start_year, start_year + projection_years))

    try:
        hazard = build_dynamic_hazard_table(
            global_params=scenario_cfg,
            simulation_years=simulation_years,
            job_levels=standard_job_levels,
            tenure_bands=standard_tenure_bands,
            cfg_scenario_name=scenario_output_name
        )
        logger.info(f"Dynamic hazard table built successfully with {len(hazard)} rows for {len(simulation_years)} years")
    except Exception as e:
        logger.error(f"Failed to build dynamic hazard table: {e}")
        logger.exception("Dynamic hazard table generation failed, falling back to static loading")

        # Fallback to static loading if dynamic generation fails
        try:
            from cost_model.projections.hazard import load_and_expand_hazard_table
            hazard = load_and_expand_hazard_table('data/hazard_table.parquet')
            if hazard.empty:
                logger.warning("Hazard table from parquet is empty, falling back to CSV")
                # Fallback to CSV if parquet doesn't exist or is empty
                hazard_path = Path('cost_model/state/hazard_table.csv')
                hazard = pd.read_csv(hazard_path)
                # Apply basic column renaming for CSV
                column_mapping = {
                    'year': 'simulation_year',
                    'employee_level': 'employee_level',
                    'tenure_band': 'employee_tenure_band',
                    'term_rate': 'term_rate',
                    'comp_raise_pct': 'comp_raise_pct',
                    'cola_pct': 'cola_pct',
                    'new_hire_termination_rate': 'new_hire_termination_rate'
                }
                hazard = hazard.rename(columns={k: v for k, v in column_mapping.items() if k in hazard.columns})
        except Exception as fallback_error:
            logger.error(f"Fallback hazard table loading also failed: {fallback_error}")
            raise RuntimeError("Both dynamic and static hazard table loading failed") from fallback_error

    # 5. Prepare output dir
    scenario_output_dir = output_dir_base / scenario_output_name
    scenario_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output will be saved to: {scenario_output_dir}")

    # 6. Run multi-year engine pipeline
    yearly_snapshots = {}
    all_events = events.copy()
    snapshot_df = snap.copy()
    for i in range(projection_years):
        year = start_year + i
        logger.info(f"Simulating year {year}")

        # Store start of year snapshot for enhanced yearly snapshot creation
        start_of_year_snapshot = snapshot_df.copy()

        all_events, snapshot_df = run_one_year(
            event_log=all_events,
            prev_snapshot=snapshot_df,
            year=year,
            global_params=scenario_cfg,
            plan_rules=scenario_cfg.plan_rules.model_dump() if scenario_cfg.plan_rules else {},
            hazard_table=hazard,
            rng=rng,
            deterministic_term=True
        )
        # Ensure simulation_year is set in the snapshot
        snapshot_df['simulation_year'] = year

        # --- BEGIN AGE CALCULATION INTEGRATION ---
        # Define the 'as_of' date for age calculation for the current snapshot
        # Typically, this would be the end of the simulation year or start of the next.
        as_of_date_for_age = pd.Timestamp(f"{year}-12-31")  # Example: End of year

        # Ensure the birth date column is in datetime format if not already
        # (apply_age also does pd.to_datetime, but good to be aware)
        # snapshot_df[EMP_BIRTH_DATE] = pd.to_datetime(snapshot_df[EMP_BIRTH_DATE], errors='coerce')

        logger.debug(f"Applying age and age band calculations for year {year} as of {as_of_date_for_age}")
        snapshot_df = apply_age(
            df=snapshot_df,
            birth_col=EMP_BIRTH_DATE,    # From schema.py
            as_of=as_of_date_for_age,
            out_age_col=EMP_AGE,         # From schema.py
            out_band_col=EMP_AGE_BAND    # From schema.py
        )
        logger.debug(f"Columns after apply_age: {snapshot_df.columns.tolist()}")
        if EMP_AGE in snapshot_df.columns:
            logger.debug(f"Sample EMP_AGE for year {year}: {snapshot_df[EMP_AGE].head().tolist()}")
        if EMP_AGE_BAND in snapshot_df.columns:
            logger.debug(f"Sample EMP_AGE_BAND for year {year}: {snapshot_df[EMP_AGE_BAND].head().tolist()}")
        # --- END AGE CALCULATION INTEGRATION ---

        # Create enhanced yearly snapshot with proper employee_status_eoy
        from cost_model.projections.snapshot import build_enhanced_yearly_snapshot

        # Get events for this year from the cumulative event log
        year_events = all_events[all_events['simulation_year'] == year] if 'simulation_year' in all_events.columns else all_events

        # Build the enhanced yearly snapshot that includes all employees active during the year
        # and properly sets employee_status_eoy
        enhanced_yearly_snapshot = build_enhanced_yearly_snapshot(
            start_of_year_snapshot=start_of_year_snapshot,
            end_of_year_snapshot=snapshot_df,
            year_events=year_events,
            simulation_year=year
        )

        # Use the enhanced snapshot for storage and metrics
        yearly_snapshots[year] = enhanced_yearly_snapshot.copy()

        # Save snapshot and events for this year
        year_dir = scenario_output_dir / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        enhanced_yearly_snapshot.to_parquet(year_dir / "snapshot.parquet")
        all_events.to_parquet(year_dir / "events.parquet")
        logger.info(f"Saved enhanced outputs for year {year} with {len(enhanced_yearly_snapshot)} employees")

    logger.info(f"Simulation complete for scenario '{scenario_name}'.")

    # 7. Calculate Summary Metrics (Optional)
    summary_metrics_df = None
    if save_summary_metrics and yearly_snapshots:
        logger.info("Calculating summary metrics...")
        try:
            all_results_df = pd.concat(yearly_snapshots.values(), ignore_index=True)
            summary_metrics_df = calculate_summary_metrics(all_results_df, scenario_cfg)
            logger.info("Summary metrics calculated.")
        except Exception:
            logger.exception("Error calculating summary metrics.")
            # Continue without metrics if calculation fails

    # 8. Save Outputs
    logger.info("Saving simulation outputs...")
    output_prefix = scenario_output_name # Use scenario name for file prefix

    # Save Summary Metrics & Sanity Checks
    if save_summary_metrics and summary_metrics_df is not None:
        try:
            _validate_summary(summary_metrics_df)
            from cost_model.data.writers import write_summary_metrics
            summary_path = write_summary_metrics(summary_metrics_df, scenario_output_dir, output_prefix)
            logger.info(f"Summary metrics written to {summary_path}")
            logger.info("Summary metrics saved.")
        except Exception:
            logger.exception("Sanity check or write of summary metrics failed.")
            raise

    logger.info(f"===== Simulation Finished for Scenario: '{scenario_name}' =====")


def _validate_summary(df):
    """Fast-fail checks for your summary metrics."""
    import numpy as np
    from cost_model.state.schema import SUMMARY_ACTIVE_HEADCOUNT

    # Use canonical column names from schema
    required_cols = {
        SUMMARY_ACTIVE_HEADCOUNT,  # 'active_headcount'
        'eligible_count',
        'participant_count',
        'participation_rate',
        'total_compensation'
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Summary is missing required columns: {missing}")
    # No zero headcounts
    if (df[SUMMARY_ACTIVE_HEADCOUNT] == 0).any():
        zero_years = df.index[df[SUMMARY_ACTIVE_HEADCOUNT] == 0].tolist()
        raise ValueError(f"Year(s) with zero headcount: {zero_years}")
    # Internal consistency: hires - terms + initial == final
    if 'new_hires' in df.columns and 'terminations' in df.columns:
        diffs = df['new_hires'] - df['terminations'] + df[SUMMARY_ACTIVE_HEADCOUNT].iloc[0]
        if not np.allclose(diffs.cumsum().iloc[-1], df[SUMMARY_ACTIVE_HEADCOUNT].iloc[-1]):
            raise ValueError("Headcount evolution inconsistency detected.")
