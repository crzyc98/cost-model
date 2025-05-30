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
from cost_model.projections.hazard import build_hazard_table, load_and_expand_hazard_table
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

    # First, load and expand the hazard table from the parquet file
    logger.info("Loading hazard table from parquet file and expanding role='*' entries")
    expanded_hazard_table = load_and_expand_hazard_table('data/hazard_table.parquet')

    # Check if we successfully loaded and expanded the hazard table
    if expanded_hazard_table.empty:
        logger.warning("Could not load or expand hazard table from parquet. Falling back to build_hazard_table.")
        logger.info(f"Building hazard table for years {projection_years}")
        hazard_table = build_hazard_table(
            years=projection_years,
            initial_snapshot=initial_snapshot,
            global_params=global_params,
            plan_rules_config=plan_rules
        )
    else:
        # Filter the expanded hazard table to only include the projection years
        if 'simulation_year' in expanded_hazard_table.columns:
            expanded_hazard_table = expanded_hazard_table[expanded_hazard_table['simulation_year'].isin(projection_years)]

        # Ensure the hazard table has all required columns
        from cost_model.state.schema import (
            SIMULATION_YEAR, EMP_LEVEL, EMP_TENURE_BAND, TERM_RATE, COMP_RAISE_PCT,
            NEW_HIRE_TERM_RATE, COLA_PCT, CFG
        )

        # Map column names if needed
        column_mapping = {
            'simulation_year': SIMULATION_YEAR,
            'employee_level': EMP_LEVEL,
            'tenure_band': EMP_TENURE_BAND,
            'term_rate': TERM_RATE,
            'comp_raise_pct': COMP_RAISE_PCT,
            'new_hire_termination_rate': NEW_HIRE_TERM_RATE,
            'cola_pct': COLA_PCT,
            # Add new granular compensation columns
            'merit_raise_pct': 'merit_raise_pct',
            'promotion_raise_pct': 'promotion_raise_pct',
            'promotion_rate': 'promotion_rate',
        }

        for src, dst in column_mapping.items():
            if src in expanded_hazard_table.columns and dst not in expanded_hazard_table.columns:
                expanded_hazard_table[dst] = expanded_hazard_table[src]

        # Add missing columns with default values
        if CFG not in expanded_hazard_table.columns:
            expanded_hazard_table[CFG] = plan_rules

        # Log some statistics about the expanded hazard table
        logger.info(f"Expanded hazard table has {len(expanded_hazard_table)} rows")
        if EMP_LEVEL in expanded_hazard_table.columns and EMP_TENURE_BAND in expanded_hazard_table.columns:
            unique_combos = expanded_hazard_table[[EMP_LEVEL, EMP_TENURE_BAND]].drop_duplicates()
            logger.info(f"Expanded hazard table has {len(unique_combos)} unique (employee_level, tenure_band) combinations")

        # Use the expanded hazard table
        hazard_table = expanded_hazard_table

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
    all_employment_summaries = []
    yearly_eoy_snapshots = {}

    # Import summary functions
    from cost_model.projections.summaries.employment import make_yearly_status

    # Performance metrics
    performance_logger = logging.getLogger(PERFORMANCE_LOGGER)

    for year in projection_years:
        logger.info(f"Running projection for year {year}...")

        # Time the projection
        import time
        start_time = time.time()

        try:
            # Store the start-of-year snapshot before applying any events
            start_of_year_snapshot = current_snapshot.copy()

            # Run the projection for one year with exact targeting support
            cumulative_event_log, eoy_snapshot = run_one_year(
                event_log=current_event_log,
                prev_snapshot=current_snapshot,
                year=year,
                global_params=global_params,
                plan_rules=plan_rules,
                hazard_table=hazard_table,
                rng=rng,
                census_template_path=census_template_path,
                rng_seed_offset=year,  # Add year-specific seed offset
                deterministic_term=getattr(global_params, 'deterministic_termination', True)  # Enable deterministic terminations
            )

            # Log performance metrics
            elapsed = time.time() - start_time
            performance_logger.info(
                f"Year {year} projection completed in {elapsed:.2f} seconds. "
                f"Snapshot size: {len(eoy_snapshot)} rows, "
                f"Event log size: {len(cumulative_event_log)} rows"
            )

            logger.info(f"Projection engine run completed for year {year} in {elapsed:.2f} seconds")

            # Build enhanced yearly snapshot using the approved solution FIRST
            from cost_model.projections.snapshot import build_enhanced_yearly_snapshot

            # Get events for this year from the cumulative event log
            year_events = cumulative_event_log[
                cumulative_event_log['simulation_year'] == year
            ] if 'simulation_year' in cumulative_event_log.columns else cumulative_event_log

            # Build the enhanced yearly snapshot that includes all employees active during the year
            enhanced_yearly_snapshot = build_enhanced_yearly_snapshot(
                start_of_year_snapshot=start_of_year_snapshot,
                end_of_year_snapshot=eoy_snapshot,
                year_events=year_events,
                simulation_year=year
            )

            # Generate employment status summary using the enhanced snapshot (contains terminated employees)
            employment_summary = make_yearly_status(
                current_snapshot,        # Start of year snapshot
                enhanced_yearly_snapshot,  # Enhanced snapshot with terminated employees
                cumulative_event_log,
                year
            )

            # Log employment summary
            logger.info(
                f"Year {year} - Active: {employment_summary.get('active', 0)}, "
                f"New Hires: {employment_summary.get('new_hire_actives', 0) + employment_summary.get('new_hire_terms', 0)}, "
                f"Terminations: {employment_summary.get('experienced_terms', 0) + employment_summary.get('new_hire_terms', 0)}"
            )

            # Save enhanced snapshot for this year
            yearly_eoy_snapshots[year] = enhanced_yearly_snapshot

            # Append employment summary to list (core summaries will be calculated later using corrected function)
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
        # Use the corrected calculate_summary_metrics function instead of broken inline logic
        logger.info("Calculating summary metrics using corrected function...")
        from cost_model.reporting.metrics import calculate_summary_metrics

        # Combine all yearly snapshots for summary calculation
        if yearly_eoy_snapshots:
            all_results_df = pd.concat(yearly_eoy_snapshots.values(), ignore_index=True)
            logger.info(f"Combined {len(yearly_eoy_snapshots)} yearly snapshots for summary calculation")

            # Calculate corrected summary metrics
            summary_results_df = calculate_summary_metrics(all_results_df, config_ns.__dict__ if config_ns else {})
            logger.info("Summary metrics calculated successfully using corrected function")
        else:
            logger.warning("No yearly snapshots available for summary calculation")
            summary_results_df = pd.DataFrame()

        # Convert employment summaries to DataFrame
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

    except Exception as e:
        warnings_logger = logging.getLogger("warnings_errors")

        # Check if the error message contains specific patterns to provide better diagnostics
        error_str = str(e)
        if "Duplicate column names found" in error_str:
            warnings_logger.error(
                "DataFrame error in save_detailed_results: %s\n"
                "This is likely due to duplicate column names in summary_to_save after merging employment status summary. "
                "Check for column name conflicts in summary and employment_status DataFrames.", e
            )
        elif "must specify a string key" in error_str and "Dict" in error_str:
            warnings_logger.error(
                "Dict serialization error in save_detailed_results: %s\n"
                "This may be due to non-string keys in dictionary columns being saved to parquet.", e
            )
        elif "promotion matrix" in error_str.lower() or "markov" in error_str.lower():
            warnings_logger.error("Promotion matrix load/validation failed: %s", e)
        else:
            warnings_logger.error(
                "Error during results saving: %s\n"
                "Check DataFrame operations in save_detailed_results for potential issues.", e,
                exc_info=True
            )
        sys.exit(1)

if __name__ == "__main__":
    main()
