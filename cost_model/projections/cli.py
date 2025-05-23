# cost_model/projections/cli.py
# Command-line interface entry point (click/argparse)
import argparse
import logging
from pathlib import Path

import pandas as pd
from datetime import datetime

from cost_model.config.loaders import load_config_to_namespace  # Use robust loader with flattening
from .snapshot import create_initial_snapshot
from .event_log import create_initial_event_log
from .runner import run_one_year
from cost_model.plan_rules import load_plan_rules
from cost_model.projections.hazard import build_hazard_table
import numpy as np
from .reporting import save_detailed_results, plot_projection_results # Assuming plot_projection_results will be implemented
from .snapshot import consolidate_snapshots_to_parquet

# Setup basic logging
# More sophisticated logging configuration can be added later if needed
LOG_DIR_ROOT = Path("output_dev/projection_logs")
LOG_DIR_ROOT.mkdir(parents=True, exist_ok=True)

# Create a unique log file for each run, or a general one
# log_file_name = f"projection_cli_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file_name = "projection_cli_run.log"

# Ensure the log file is truncated (emptied) at the start of every run
with open(LOG_DIR_ROOT / log_file_name, "w") as f:
    pass  # Just truncate the file

def main():
    parser = argparse.ArgumentParser(description="Run multi-year cost model projections.")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--census", 
        type=str, 
        required=True, 
        help="Path to the Parquet census data file."
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None, # If None, will use from config or a default in reporting
        help="Optional: Directory to save output files. Overrides config if provided."
    )
    parser.add_argument(
        "--scenario-name",
        type=str,
        default="projection_cli",
        help="Name for the scenario, used in output file naming."
    )

    args = parser.parse_args()
    
    # Setup logging based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)-8s] [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        handlers=[
            logging.StreamHandler(),  # Output to console
            logging.FileHandler(LOG_DIR_ROOT / log_file_name)  # Output to file
        ]
    )
    logger = logging.getLogger(__name__)  # Logger for the CLI module itself
    logger.setLevel(log_level)

    logger.info(f"Starting projection run with CLI arguments: {args}")

    try:
        # 1. Load Configuration
        logger.info(f"Loading configuration from: {args.config}")
        config_ns = load_config_to_namespace(args.config)
        # Update logging level from config if specified
        log_level_str = getattr(getattr(config_ns, 'global_parameters', {}), 'log_level', 'INFO').upper()
        numeric_level = getattr(logging, log_level_str, logging.INFO)
        logging.getLogger().setLevel(numeric_level) # Set root logger level
        logger.info(f"Logging level set to: {log_level_str}")

        # Determine output directory
        if args.output_dir:
            output_path = Path(args.output_dir)
        elif hasattr(config_ns.global_parameters, 'output_directory'):
            output_path = Path(config_ns.global_parameters.output_directory)
        else:
            output_path = Path(f"output_dev/{args.scenario_name}_results")
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output will be saved to: {output_path}")

        start_year = config_ns.global_parameters.start_year

        # 2. Create Initial Snapshot
        logger.info(f"Creating initial snapshot from census: {args.census} for start year: {start_year}")
        initial_snapshot = create_initial_snapshot(start_year, args.census)

        # 3. Create Initial Event Log
        logger.info(f"Creating initial event log for start year: {start_year}")
        initial_event_log = create_initial_event_log(start_year)

        # 4. Prepare arguments for run_one_year
        year = start_year
        global_params = config_ns.global_parameters
        plan_rules = load_plan_rules(config_ns) if hasattr(config_ns, 'plan_rules') or hasattr(config_ns, 'plan_rules_path') else {}
        # Prepare arguments for build_hazard_table
        projection_years_list = list(range(year, year + global_params.projection_years))

        hazard_table = build_hazard_table(
            years=projection_years_list,
            initial_snapshot=initial_snapshot,
            global_params=global_params,
            plan_rules_config=plan_rules
        )
        seed = getattr(global_params, 'random_seed', 42)
        rng = np.random.default_rng(seed)
        census_template_path = getattr(global_params, 'census_template_path', None)

        # 5. Run Projection Engine
        logger.info("Starting projection engine...")
        final_cumulative_event_log, final_eoy_snapshot = run_one_year(
            event_log=initial_event_log,
            prev_snapshot=initial_snapshot,
            year=year,
            global_params=global_params,
            plan_rules=plan_rules,
            hazard_table=hazard_table,
            rng=rng,
            census_template_path=census_template_path
        )
        logger.info("Projection engine run completed.")

        # 5. Reporting & Saving
        logger.info("Saving detailed results...")
        # Create empty DataFrames for the missing return values
        summary_results_df = pd.DataFrame()
        employment_status_summary_df = pd.DataFrame()
        yearly_eoy_snapshots = {}
        
        save_detailed_results(
            output_path=output_path,
            scenario_name=args.scenario_name,
            final_snapshot=final_eoy_snapshot,
            full_event_log=final_cumulative_event_log,
            summary_statistics=summary_results_df,
            employment_status_summary_df=employment_status_summary_df,
            yearly_snapshots=yearly_eoy_snapshots,
            config_to_save=config_ns
        )
        logger.info(f"Detailed results for '{args.scenario_name}' saved to {output_path}")
        
        # Detailed directory listing to diagnose snapshot location
        logger.info(f"After save_detailed_results, full directory structure under {output_path!r}:")
        for p in output_path.rglob("*"):
            logger.info(f"  {p.relative_to(output_path)}")
        
        # Consolidate all yearly snapshots into a single file
        logger.info("Consolidating yearly snapshots into a single file...")
        consolidate_snapshots_to_parquet(
            snapshots_dir=output_path / "yearly_snapshots",
            output_path=output_path / "consolidated_snapshots.parquet"
        )
        logger.info("Consolidated snapshots created successfully")

        logger.info("Generating and saving plots...")
        plot_projection_results(summary_results_df, output_path) # Assuming plot_projection_results takes summary_df

        logger.info(f"Projection run for scenario '{args.scenario_name}' completed successfully.")
        logger.info(f"All outputs saved in: {output_path}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during the projection run: {e}", exc_info=True)
        # Consider re-raising or exiting with error code for script automation

if __name__ == "__main__":
    main()
