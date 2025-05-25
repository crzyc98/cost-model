# cost_model/projections/cli.py
# Command-line interface entry point (click/argparse)
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import pandas as pd
import numpy as np
from datetime import datetime

from cost_model.config.loaders import load_config_to_namespace
from .snapshot import create_initial_snapshot
from .event_log import create_initial_event_log
from cost_model.engines.run_one_year.orchestrator import run_one_year
from cost_model.plan_rules import load_plan_rules
from cost_model.projections.hazard import build_hazard_table
from .reporting import save_detailed_results, plot_projection_results
from .snapshot import consolidate_snapshots_to_parquet

# Import logging configuration
from logging_config import setup_logging, PROJECTION_LOGGER, PERFORMANCE_LOGGER, ERROR_LOGGER, DEBUG_LOGGER

# Get logger for this module
logger = logging.getLogger(__name__)

# Directory for log files
LOG_DIR = Path("output_dev/projection_logs")

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run multi-year cost model projections.")
    
    # Required arguments
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
    
    # Optional arguments
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(LOG_DIR),
        help=f"Directory to store log files (default: {LOG_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output files. Overrides config if provided."
    )
    parser.add_argument(
        "--scenario-name",
        type=str,
        default="projection_cli",
        help="Name for the scenario, used in output file naming."
    )
    
    return parser.parse_args()


def initialize_logging(debug: bool = False, log_dir: Path = LOG_DIR) -> None:
    """Initialize the logging configuration.
    
    Args:
        debug: Whether to enable debug logging
        log_dir: Directory to store log files
    """
    try:
        # Configure logging
        setup_logging(log_dir=log_dir, debug=debug)
        
        # Log startup information
        logger.info("Starting cost model projection")
        logger.info(f"Command line arguments: {sys.argv}")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Pandas version: {pd.__version__}")
        logger.info(f"NumPy version: {np.__version__}")
        
        if debug:
            logger.debug("Debug logging enabled")
            
    except Exception as e:
        print(f"Error initializing logging: {e}", file=sys.stderr)
        raise


def main():
    """Main entry point for the cost model projection CLI."""
    # Get error logger early in case we need it for initialization errors
    err_logger = logging.getLogger("warnings_errors")
    
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Initialize logging
        log_dir = Path(args.log_dir)
        initialize_logging(debug=args.debug, log_dir=log_dir)
        
        # Get module-specific loggers after logging is initialized
        proj_logger = logging.getLogger("cost_model.projection")
        perf_logger = logging.getLogger("cost_model.performance")
        debug_logger = logging.getLogger("cost_model.debug")
        
        proj_logger.info("Starting cost model projection")
        perf_logger.info("Performance monitoring initialized")
        debug_logger.debug("Debug logging enabled")
        
        # Log startup information
        logger.info(f"Starting projection run with arguments: {vars(args)}")
        
        # 1. Load Configuration
        logger.info(f"Loading configuration from: {args.config}")
        config_ns = load_config_to_namespace(args.config)
        
        # Set log level from config if specified
        log_level_str = getattr(getattr(config_ns, 'global_parameters', {}), 'log_level', 'INFO').upper()
        numeric_level = getattr(logging, log_level_str, logging.INFO)
        logging.getLogger().setLevel(numeric_level)
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
        
        # Run the projection and handle any errors
        try:
            run_projection(args, config_ns, output_path)
            return 0
            
        except FileNotFoundError as e:
            err_logger.error(f"Configuration file not found: {e}", exc_info=True)
        except ValueError as e:
            err_logger.error(f"Invalid configuration: {e}", exc_info=True)
        except Exception as e:
            err_logger.critical("Fatal error during projection", exc_info=True)
            
        return 1
        
    except Exception as e:
        # This is a last resort catch for errors before logging is properly set up
        print(f"FATAL: {str(e)}", file=sys.stderr)
        err_logger.critical("Fatal initialization error", exc_info=True)
        return 1


def run_projection(args: argparse.Namespace, config_ns: Any, output_path: Path) -> None:
    """Run the projection with the given configuration.
    
    Args:
        args: Parsed command line arguments
        config_ns: Configuration namespace
        output_path: Path to save output files
        
    Raises:
        FileNotFoundError: If required input files are missing
        ValueError: If configuration is invalid
        Exception: For any other unexpected errors
    """
    start_year = config_ns.global_parameters.start_year
    
    # 2. Create Initial Snapshot
    logger.info(f"Creating initial snapshot from census: {args.census} for start year: {start_year}")
    initial_snapshot = create_initial_snapshot(start_year, args.census)

    # 3. Create Initial Event Log
    logger.info(f"Creating initial event log for start year: {start_year}")
    initial_event_log = create_initial_event_log(start_year)
    
    # 4. Prepare arguments for run_one_year
    global_params = config_ns.global_parameters
    plan_rules = load_plan_rules(config_ns) if hasattr(config_ns, 'plan_rules') or hasattr(config_ns, 'plan_rules_path') else {}
    projection_years = list(range(start_year, start_year + global_params.projection_years))
    
    logger.info(f"Building hazard table for years {projection_years}")
    hazard_table = build_hazard_table(
        years=projection_years,
        initial_snapshot=initial_snapshot,
        global_params=global_params,
        plan_rules_config=plan_rules
    )
    
    # Initialize random number generator
    seed = getattr(global_params, 'random_seed', 42)
    rng = np.random.default_rng(seed)
    logger.debug(f"Initialized RNG with seed: {seed}")
    
    census_template_path = getattr(global_params, 'census_template_path', None)
    if census_template_path:
        logger.info(f"Using census template: {census_template_path}")

    # === Refactored Multi-Year Projection Loop ===
    current_snapshot = initial_snapshot
    current_event_log = initial_event_log
    all_core_summaries = []
    all_employment_summaries = []
    yearly_eoy_snapshots = {}
    
    # Import summary functions
    from cost_model.projections.summaries.employment import make_yearly_status
    from cost_model.projections.summaries.core import build_core_summary
    
    # Performance metrics
    performance_logger = logging.getLogger(PERFORMANCE_LOGGER)
    
    for year in projection_years:
        logger.info(f"Running projection for year {year}...")
        
        # Time the projection
        import time
        start_time = time.time()
        
        try:
            # Run the projection for one year
            cumulative_event_log, eoy_snapshot = run_one_year(
                event_log=current_event_log,
                prev_snapshot=current_snapshot,
                year=year,
                global_params=global_params,
                plan_rules=plan_rules,
                hazard_table=hazard_table,
                rng=rng,
                census_template_path=census_template_path
            )
            
            # Log performance metrics
            elapsed = time.time() - start_time
            performance_logger.info(
                f"Year {year} projection completed in {elapsed:.2f} seconds. "
                f"Snapshot size: {len(eoy_snapshot)} rows, "
                f"Event log size: {len(cumulative_event_log)} rows"
            )
            
            logger.info(f"Projection engine run completed for year {year} in {elapsed:.2f} seconds")

            # Generate employment status summary
            employment_summary = make_yearly_status(
                current_snapshot,  # Start of year snapshot
                eoy_snapshot,      # End of year snapshot
                cumulative_event_log, 
                year
            )
            
            # Log employment summary
            logger.info(
                f"Year {year} - Active: {employment_summary.get('active', 0)}, "
                f"New Hires: {employment_summary.get('new_hire_actives', 0) + employment_summary.get('new_hire_terms', 0)}, "
                f"Terminations: {employment_summary.get('experienced_terms', 0) + employment_summary.get('new_hire_terms', 0)}"
            )
            
            # Generate core summary with basic metrics
            active_employees = eoy_snapshot[eoy_snapshot['active']] if 'active' in eoy_snapshot.columns else eoy_snapshot
            terminations = cumulative_event_log[cumulative_event_log['event_type'] == 'termination']
            
            core_summary = {
                'Projection Year': year,
                'active_headcount': len(active_employees),
                'total_terminations': len(terminations),
                'avg_compensation': active_employees['gross_comp'].mean() if 'gross_comp' in active_employees.columns else 0,
                'total_compensation': active_employees['gross_comp'].sum() if 'gross_comp' in active_employees.columns else 0,
                    'new_hires': employment_summary.get('new_hire_actives', 0) + employment_summary.get('new_hire_terms', 0)
                }
            
            # Save snapshot for this year
            yearly_eoy_snapshots[year] = eoy_snapshot
            
            # Append to summary lists
            all_core_summaries.append(core_summary)
            all_employment_summaries.append(employment_summary)
            
            # Update for next iteration
            current_snapshot = eoy_snapshot
            current_event_log = cumulative_event_log
            
        except Exception as e:
            logger.error(f"Error during projection for year {year}", exc_info=True)
            raise
                
    # === End of Projection Loop ===
    
    # Save results
    try:
        # Convert summaries to DataFrames
        core_summary_df = pd.DataFrame(all_core_summaries)
        employment_summary_df = pd.DataFrame(all_employment_summaries)
        # Convert lists to DataFrames
        summary_results_df = pd.DataFrame(all_core_summaries)
        employment_status_summary_df = pd.DataFrame(all_employment_summaries)
        # Set final snapshot and event log for saving
        final_eoy_snapshot = current_snapshot
        final_cumulative_event_log = current_event_log
        # END Refactored Multi-Year Projection Loop
        
        # 7. Reporting & Saving
        logger.info("Saving detailed results...")
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
        
        # Only try to consolidate yearly snapshots if the directory exists and is not empty
        yearly_snapshots_dir = output_path / "yearly_snapshots"
        if yearly_snapshots_dir.exists() and any(yearly_snapshots_dir.glob("*.parquet")):
            logger.info("Consolidating yearly snapshots into a single file...")
            try:
                consolidate_snapshots_to_parquet(
                    snapshots_dir=yearly_snapshots_dir,
                    output_path=output_path / "consolidated_snapshots.parquet"
                )
                logger.info("Consolidated snapshots created successfully")
            except Exception as e:
                logger.warning(f"Could not consolidate yearly snapshots: {e}")
        else:
            logger.info("No yearly snapshots found to consolidate")

        logger.info("Generating and saving plots...")
        plot_projection_results(summary_results_df, output_path) # Assuming plot_projection_results takes summary_df

        logger.info(f"Projection run for scenario '{args.scenario_name}' completed successfully.")
        logger.info(f"All outputs saved in: {output_path}")

    except (FileNotFoundError, ValueError) as e:
        warnings_logger = logging.getLogger("warnings_errors")
        warnings_logger.error("Promotion matrix load/validation failed: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during the projection run: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
