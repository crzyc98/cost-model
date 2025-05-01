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
from typing import Dict, Any, Optional

# Import necessary components from the cost_model package
# Use absolute imports assuming cost_model is on the Python path or installed
try:
    from cost_model.config.models import MainConfig, GlobalParameters
    from cost_model.config.accessors import get_scenario_config
    from cost_model.data.readers import read_census_data
    from cost_model.data.writers import write_snapshots, write_summary_metrics
    # Assume engine modules exist in dynamics and rules subpackages
    from cost_model.dynamics.engine import run_dynamics_for_year
    from cost_model.rules.engine import apply_rules_for_year
    from cost_model.reporting.metrics import calculate_summary_metrics
    # Import column constants if needed for initialization/checks here
    from cost_model.utils.columns import EMP_TERM_DATE, EMP_ID # Example, adjust as needed
    from cost_model.utils.constants import ACTIVE_STATUS # Example

except ImportError as e:
    print(f"Error importing simulation components: {e}")
    print("Ensure all submodules (config, data, dynamics, rules, reporting, utils) exist and are importable.")
    # Define dummy functions/classes if needed for static analysis, though runtime will fail
    class MainConfig: pass
    class GlobalParameters: pass
    def get_scenario_config(*args, **kwargs): raise NotImplementedError("get_scenario_config missing")
    def read_census_data(*args, **kwargs): raise NotImplementedError("read_census_data missing")
    def write_snapshots(*args, **kwargs): raise NotImplementedError("write_snapshots missing")
    def write_summary_metrics(*args, **kwargs): raise NotImplementedError("write_summary_metrics missing")
    def run_dynamics_for_year(*args, **kwargs): raise NotImplementedError("run_dynamics_for_year missing")
    def apply_rules_for_year(*args, **kwargs): raise NotImplementedError("apply_rules_for_year missing")
    def calculate_summary_metrics(*args, **kwargs): raise NotImplementedError("calculate_summary_metrics missing")
    EMP_TERM_DATE, EMP_ID, ACTIVE_STATUS = 'employee_termination_date', 'employee_id', 'Active'


logger = logging.getLogger(__name__)

def run_simulation(
    main_config: MainConfig,
    scenario_name: str,
    input_census_path: Path,
    output_dir_base: Path,
    # Optional flags to control output
    save_detailed_snapshots: bool = True,
    save_summary_metrics: bool = True
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

    Raises:
        KeyError: If the scenario_name is not found in main_config.
        FileNotFoundError: If the input_census_path does not exist.
        Exception: For errors during simulation steps (dynamics, rules, etc.).
    """
    logger.info(f"===== Starting Simulation for Scenario: '{scenario_name}' =====")

    # 1. Get Resolved Scenario Configuration
    try:
        scenario_cfg: GlobalParameters = get_scenario_config(main_config, scenario_name)
        # Extract key parameters for easier access
        start_year = scenario_cfg.start_year
        projection_years = scenario_cfg.projection_years
        seed = scenario_cfg.random_seed
        scenario_output_name = main_config.scenarios[scenario_name].name or scenario_name # Use defined name or key
    except KeyError:
        # Error already logged by get_scenario_config
        raise # Re-raise the KeyError
    except Exception as e:
        logger.exception(f"Unexpected error getting scenario config for '{scenario_name}'.")
        raise

    # 2. Setup RNG for simulation steps (if needed at this level)
    # Individual engines (dynamics, rules) might manage their own RNGs based on this seed
    # Or we could create specific RNGs here and pass them down
    if seed is not None:
        logger.info(f"Using Master Random Seed: {seed}")
        # Example: master_rng = np.random.default_rng(seed)
        #          dynamics_rng, rules_rng = master_rng.spawn(2) # If needed to pass down
    else:
        logger.warning("No random seed provided. Results may not be reproducible.")
        # dynamics_rng, rules_rng = None, None # Or initialize default RNGs

    # 3. Load Initial Census Data
    logger.info(f"Loading initial census data from: {input_census_path}")
    try:
        current_df = read_census_data(input_census_path)
        if current_df is None or current_df.empty:
            logger.error("Initial census data is empty or failed to load.")
            return # Or raise error
        # Ensure the required ID column exists after loading/renaming in reader
        if EMP_ID not in current_df.columns:
             logger.error(f"Required column '{EMP_ID}' not found in loaded census data. Check reader logic.")
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
            final_year_df = apply_rules_for_year(
                year_end_df=dynamics_output_df,
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
        yearly_snapshots[sim_year] = final_year_df.copy()
        logger.debug(f"Stored snapshot for year {sim_year}.")

        # Prepare for the next iteration: Carry forward only the active population
        # Active status should be determined correctly within apply_rules_for_year or here
        # Example: Assuming a STATUS_COL exists and is set correctly
        # active_mask = final_year_df.get(STATUS_COL, pd.Series(True, index=final_year_df.index)) == ACTIVE_STATUS
        # Or based on termination date
        plan_end_date = pd.Timestamp(f"{sim_year}-12-31")
        active_mask = final_year_df[EMP_TERM_DATE].isna() | (final_year_df[EMP_TERM_DATE] > plan_end_date)
        current_df = final_year_df[active_mask].copy()
        # Drop the simulation_year column before passing to next iteration if needed
        # current_df = current_df.drop(columns=['simulation_year'])

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
    if save_detailed_snapshots:
        try:
            write_snapshots(yearly_snapshots, scenario_output_dir, output_prefix)
            logger.info("Detailed yearly snapshots saved.")
        except Exception as e:
            logger.exception("Error saving detailed snapshots.")

    # Save Summary Metrics
    if save_summary_metrics and summary_metrics_df is not None:
         try:
             write_summary_metrics(summary_metrics_df, scenario_output_dir, output_prefix)
             logger.info("Summary metrics saved.")
         except Exception as e:
             logger.exception("Error saving summary metrics.")

    logger.info(f"===== Simulation Finished for Scenario: '{scenario_name}' =====")

# --- Example CLI Runner (can be moved to scripts/run_simulation.py) ---
if __name__ == '__main__':
    # This block allows running a simulation directly using this file
    # In practice, you'd likely call run_simulation from scripts/run_simulation.py

    parser = argparse.ArgumentParser(description="Run Retirement Plan Cost Model Simulation")
    parser.add_argument("--config", "-c", required=True, help="Path to main YAML config file")
    parser.add_argument("--scenario", "-s", required=True, help="Name of the scenario to run")
    parser.add_argument("--census", "-d", required=True, help="Path to initial census file (CSV or Parquet)")
    parser.add_argument("--output", "-o", required=True, help="Base output directory for results")
    parser.add_argument("--no-snapshots", action="store_true", help="Disable saving detailed yearly snapshots")
    parser.add_argument("--no-summary", action="store_true", help="Disable calculating and saving summary metrics")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging level")

    args = parser.parse_args()

    # Configure Logging Level
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Load Main Config
    try:
        from cost_model.config.loaders import load_yaml_config, ConfigLoadError
        raw_config = load_yaml_config(Path(args.config))
        if not raw_config: sys.exit(1) # Exit if loading fails
        main_cfg_obj = MainConfig(**raw_config) # Validate
    except (ConfigLoadError, FileNotFoundError, Exception) as e:
        logger.exception(f"Failed to load or validate main config file {args.config}")
        sys.exit(1)

    # Run the Simulation
    try:
        run_simulation(
            main_config=main_cfg_obj,
            scenario_name=args.scenario,
            input_census_path=Path(args.census),
            output_dir_base=Path(args.output),
            save_detailed_snapshots=(not args.no_snapshots),
            save_summary_metrics=(not args.no_summary)
        )
        logger.info("Simulation run completed successfully.")
    except (KeyError, FileNotFoundError) as e:
        logger.error(f"Simulation setup failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An error occurred during the simulation run for scenario '{args.scenario}'.")
        sys.exit(1)

