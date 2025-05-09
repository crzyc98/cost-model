# scripts/run_simulation.py

#!/usr/bin/env python3
"""
Main entry point to run cost model simulations for specified scenarios.

This script orchestrates the simulation process by:
1. Parsing command-line arguments (config, scenario, census, output, etc.).
2. Loading and validating the main configuration file.
3. Calling the core simulation logic from the cost_model package.
4. Handling logging and script exit codes.
"""

import sys
import argparse
import logging
import pathlib
import subprocess

# Add project root to the Python path
# Assumes the script is in <project_root>/scripts/
project_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Now import from the cost_model package
try:
    from cost_model.config.loaders import load_yaml_config, ConfigLoadError
    from cost_model.config.models import MainConfig
    # from cost_model.simulation import run_simulation  # Assuming the core logic is here
except ImportError as e:
    print(f"Error importing cost_model components: {e}", file=sys.stderr)
    print(
        f"Ensure the cost_model package is in the Python path: {sys.path}",
        file=sys.stderr,
    )
    sys.exit(1)


def main():
    """Parses arguments, sets up logging, loads config, and runs the simulation."""

    parser = argparse.ArgumentParser(
        description="Run cost model simulations for specified scenarios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the main YAML configuration file.",
    )
    parser.add_argument(
        "--scenario",
        "-s",
        type=str,
        required=True,
        help="Name of the scenario within the config file to execute.",
    )
    parser.add_argument(
        "--census",
        "-d",
        type=str,
        required=True,
        help="Path to the initial input census file (CSV or Parquet).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to the base output directory. A subdirectory named after the scenario will be created here.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,  # Let the simulation logic handle default if None
        help="Integer for the random seed for reproducibility.",
    )
    parser.add_argument(
        "--no-snapshots",
        action="store_true",
        help="Disable saving detailed yearly snapshot files.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Disable calculating and saving summary metrics CSV.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG level logging.",
    )

    args = parser.parse_args()

    # --- Setup Logging ---
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(level=log_level, format=log_format)
    logger = logging.getLogger(__name__)  # Get logger for this script

    logger.info("Starting simulation run...")
    logger.debug("Parsed arguments: %s", args)

    # --- Load and Validate Configuration ---
    config_path = pathlib.Path(args.config)
    input_census_path = pathlib.Path(args.census)
    output_dir_base = pathlib.Path(args.output)

    try:
        logger.info(f"Loading configuration from: {config_path}")
        raw_config = load_yaml_config(config_path)
        logger.info("Validating configuration structure...")
        main_cfg_obj = MainConfig(**raw_config)
        logger.info("Configuration loaded and validated successfully.")
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except ConfigLoadError as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)
    except Exception as e:  # Catch Pydantic validation errors and others
        logger.error(f"Error validating configuration: {e}", exc_info=args.debug)
        sys.exit(1)

    # --- Call Core Simulation Logic ---
    try:
        logger.info(f"Running simulation for scenario: {args.scenario}")
        from cost_model.simulation import run_simulation

        run_simulation(
            main_config=main_cfg_obj,
            scenario_name=args.scenario,
            input_census_path=input_census_path,
            output_dir_base=output_dir_base,
            random_seed=args.seed,
            save_detailed_snapshots=not args.no_snapshots,
            save_summary_metrics=not args.no_summary,
        )
        logger.info(
            f"Simulation for scenario '{args.scenario}' completed successfully."
        )
        logger.info(f"Results saved in a subdirectory under: {output_dir_base}")

        # Run employment status summary script to provide additional insights
        logger.info("Running employment status summary script...")
        try:
            result = subprocess.run(
                [sys.executable, "scripts/employment_status_summary.py"],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("Employment status summary completed successfully.")
            logger.debug("Employment status summary output:\n%s", result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running employment status summary: {e}")
            logger.error(f"Error output:\n{e.stderr}")

    except KeyError as e:
        logger.error(f"Scenario '{args.scenario}' not found in configuration: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"Input census file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=args.debug)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
