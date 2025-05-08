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
from typing import Dict, Optional, List, Tuple, Any
import os

# --- Core Model Components --- #
# Configuration Loading & Access
from cost_model.config.loaders import load_yaml_config, ConfigLoadError
from cost_model.config.models import MainConfig, GlobalParameters # Needed for type hints
from cost_model.config.accessors import get_scenario_config # Import the actual function

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

# Utilities
from cost_model.utils.columns import STATUS_COL, ACTIVE_STATUS, EMP_PLAN_YEAR_COMP # Import constants
from cost_model.utils.labels import label_employment_status # Import the new function

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
    except Exception as e:
        # Catch any other unexpected errors during config resolution
        logger.exception(f"Unexpected error getting scenario config for '{scenario_name}'.")
        return

    # Extract key parameters and setup RNG after successful config load
    try:
        start_year = scenario_cfg.start_year
        projection_years = scenario_cfg.projection_years
        # Use provided seed if available, otherwise use seed from config
        seed = random_seed if random_seed is not None else scenario_cfg.random_seed
        scenario_output_name = main_config.scenarios[scenario_name].name or scenario_name # Use defined name or key
    except Exception as e:
        logger.exception(f"Error extracting key parameters from scenario config for '{scenario_name}'.")
        return

    # 2. Setup RNG for simulation steps (if needed at this level)
    # Individual engines (dynamics, rules) might manage their own RNGs based on this seed
    # Or we could create specific RNGs here and pass them down
    if seed is not None:
        logger.info(f"Using Master Random Seed: {seed}")
        # Example: master_rng = np.random.default_rng(seed)
        #          dynamics_rng, rules_rng = master_rng.spawn(2) # If needed to pass down
    else:
        logger.warning("No random seed provided (neither via argument nor config). Results may not be reproducible.")
        # dynamics_rng, rules_rng = None, None # Or initialize default RNGs

    # 3. Load Initial Census Data
    logger.info(f"Loading initial census data from: {input_census_path}")
    try:
        current_df = read_census_data(input_census_path)
        if current_df is None or current_df.empty:
            logger.error("Initial census data is empty or failed to load.")
            return # Or raise error
        # Ensure the required ID column exists after loading/renaming in reader
        if 'employee_id' not in current_df.columns:
             logger.error(f"Required column 'employee_id' not found in loaded census data. Check reader logic.")
             return # Or raise error

        logger.info(f"Loaded initial population: {len(current_df)} records.")
    except FileNotFoundError:
        logger.error(f"Input census file not found: {input_census_path}")
        raise # Re-raise FileNotFoundError
    except Exception as e:
        logger.exception(f"Error loading initial census data from {input_census_path}.")
        raise

    # 4. Prepare Output Directory
    scenario_output_dir = output_dir_base / scenario_output_name
    try:
        scenario_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output will be saved to: {scenario_output_dir}")
    except OSError as e:
        logger.error(f"Could not create output directory {scenario_output_dir}: {e}")
        raise

    # 5. Run Yearly Simulation Loop
    yearly_snapshots: Dict[int, pd.DataFrame] = {}
    all_results_df: Optional[pd.DataFrame] = None # To store combined results if needed

    for year_idx in range(projection_years):
        sim_year = start_year + year_idx
        logger.info(f"--- Simulating Year {sim_year} (Index {year_idx}) ---")

        # --- Step 5a: Population Dynamics (Phase 1 Logic) ---
        try:
            logger.debug(f"Running population dynamics for year {sim_year}...")
            # Pass the DataFrame from the *end* of the previous year
            # run_dynamics_for_year should handle comp bumps, terms, hires
            # It should return the DataFrame representing the state *before* plan rules
            dynamics_output_df = run_dynamics_for_year(
                current_df=current_df,
                year_config=scenario_cfg, # Pass the resolved config
                sim_year=sim_year,
                # Pass RNG if needed: rng=dynamics_rng
            )
            logger.info(f"Dynamics complete for {sim_year}. Headcount before rules: {len(dynamics_output_df)}")

        except Exception as e:
            logger.exception(f"Error during population dynamics for year {sim_year}.")
            raise # Stop simulation on error

        # --- Step 5b: Apply Plan Rules (Phase 2 Logic) ---
        try:
            logger.debug(f"Applying plan rules for year {sim_year}...")
            # Pass the output of the dynamics step
            # apply_rules_for_year should add eligibility, contributions, etc.
            final_year_df = apply_rules_for_year( # Use corrected function name
                population_df=dynamics_output_df, # Pass the df after dynamics
                year_config=scenario_cfg, # Pass the resolved config
                sim_year=sim_year,
                # Pass RNG if needed: rng=rules_rng
            )
            logger.info(f"Plan rules applied for {sim_year}.")

        except Exception as e:
            logger.exception(f"Error applying plan rules for year {sim_year}.")
            raise # Stop simulation on error

        # --- Step 5c: Store Snapshot & Prepare for Next Year ---
        final_year_df['simulation_year'] = sim_year # Add year column

        # Assign employment_status label using the utility function
        final_year_df = label_employment_status(final_year_df, sim_year)

        yearly_snapshots[sim_year] = final_year_df.copy()
        logger.debug(f"Stored snapshot for year {sim_year}.")

        # Prepare for the next iteration: Carry forward only the active population
        # Active status should be determined correctly within apply_rules_for_year or here
        # Filter for snapshot: only include those who earned salary in the sim year
        # This ensures that people terminated in prior years (and not re-hired)
        # or those who had zero comp for other reasons are excluded from the snapshot.
        salary_mask = final_year_df[EMP_PLAN_YEAR_COMP] > 0
        snapshot_df = final_year_df[salary_mask].copy()
        # Ensure days_worked is present in the output for debugging
        if 'days_worked' not in snapshot_df.columns:
            logger.warning("days_worked column missing from snapshot_df; attempting to merge from final_year_df.")
            if 'days_worked' in final_year_df.columns:
                snapshot_df['days_worked'] = final_year_df.loc[snapshot_df.index, 'days_worked']
            else:
                logger.error("days_worked column missing from both snapshot_df and final_year_df!")
        if 'days_worked' in snapshot_df.columns:
            logger.info(f"days_worked column present in snapshot_df for {sim_year} (min={snapshot_df['days_worked'].min()}, max={snapshot_df['days_worked'].max()})")
        logger.info(f"End of Year {sim_year}: Active Headcount in snapshot = {len(snapshot_df)} (days_worked included)")
        current_df = snapshot_df.copy()
        logger.info(f"End of Year {sim_year}: Active Headcount = {len(current_df)}")
        if current_df.empty and year_idx < projection_years - 1:
             logger.warning(f"Population empty at end of year {sim_year}. Stopping simulation early.")
             break # Stop if population dies out


    # 6. Combine Results (Optional, useful for metrics)
    if yearly_snapshots:
        all_results_df = pd.concat(yearly_snapshots.values(), ignore_index=True)
        logger.info(f"Combined results from {len(yearly_snapshots)} snapshots.")
    else:
        logger.warning("No yearly snapshots were generated.")
        return # Exit if no results

    # 7. Calculate Summary Metrics (Optional)
    summary_metrics_df = None
    if save_summary_metrics and all_results_df is not None:
        logger.info("Calculating summary metrics...")
        try:
            summary_metrics_df = calculate_summary_metrics(all_results_df, scenario_cfg)
            logger.info("Summary metrics calculated.")
        except Exception as e:
            logger.exception("Error calculating summary metrics.")
            # Continue without metrics if calculation fails

    # 8. Save Outputs
    logger.info("Saving simulation outputs...")
    output_prefix = scenario_output_name # Use scenario name for file prefix

    # Save Snapshots
    if save_detailed_snapshots and yearly_snapshots: # Ensure list is not empty
        logger.info(f"Saving {len(yearly_snapshots)} yearly snapshots...")
        try:
            sim_years_to_run = list(yearly_snapshots.keys())
            for current_sim_year, year_df in zip(sim_years_to_run, yearly_snapshots.values()):
                processed_df = year_df.copy() # Start with a copy for this year's snapshot

                # Filter to only employees who worked at least 1 day
                if 'days_worked' in processed_df.columns:
                    orig_count = len(processed_df)
                    processed_df = processed_df[processed_df['days_worked'] > 0]
                    logger.info(
                        f"Year {current_sim_year}: Filtered snapshot to days_worked > 0. "
                        f"{len(processed_df)} of {orig_count} rows retained."
                    )
                else:
                    logger.warning(
                        f"Year {current_sim_year}: 'days_worked' column missing from snapshot_df. No filtering applied."
                    )

                # Write output snapshot for the current year
                output_snapshot_path = os.path.join(
                    scenario_output_dir, f"{output_prefix}_year{current_sim_year}.parquet"
                )
                processed_df.to_parquet(output_snapshot_path, index=False)
                logger.info(f"Wrote snapshot: {output_snapshot_path}")
            logger.info("All yearly snapshots saved successfully.")
        except Exception as e:
            logger.exception("Error saving detailed yearly snapshots.")

    # Save Summary Metrics
    if save_summary_metrics and summary_metrics_df is not None:
         try:
             write_summary_metrics(summary_metrics_df, scenario_output_dir, output_prefix)
             logger.info("Summary metrics saved.")
         except Exception as e:
             logger.exception("Error saving summary metrics.")

    logger.info(f"===== Simulation Finished for Scenario: '{scenario_name}' =====")
