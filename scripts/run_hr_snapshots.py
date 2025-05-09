#!/usr/bin/env python3
# scripts/run_hr_snapshots.py
"""
Script to export a single, shared HR snapshot for each projection year.
This is Phase I of the two‑phase split: headcount, turnover, comp bumps & new hires
are all driven once (from the baseline scenario's settings) and then reused
by each plan‑rules scenario.
"""

import sys
from pathlib import Path

# --- Add project root to Python path FIRST ---
try:
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
except Exception as e:
    print(f"Error determining project root or modifying sys.path: {e}")
    sys.exit(1)

import argparse
import pandas as pd
import yaml
import numpy as np
import logging

# --- Now perform imports from cost_model ---
try:
    # from cost_model.config.validators import PlanRules, ValidationError
    from cost_model.dynamics.engine import run_dynamics_for_year
    from cost_model.utils.columns import (
        EMP_HIRE_DATE,
        EMP_TERM_DATE,
        EMP_BIRTH_DATE,
        EMP_GROSS_COMP,
    )
except ImportError as e:
    print(f"Error importing from cost_model: {e}. Ensure the necessary modules exist.")
    sys.exit(1)

# --- Logging Configuration ---
# Configure logging at the module level or within main()
# Basic configuration: logs INFO level and above to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
# ---------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Phase I: run HR dynamics exactly once (baseline) and dump snapshots"
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to YAML config (must include a 'baseline' scenario)",
    )
    parser.add_argument(
        "--census",
        "-d",
        required=True,
        help="Path to census file (CSV or Parquet, with employee_hire_date, employee_termination_date, employee_birth_date)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory for base_run_year{n}.parquet files",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    logger.info("--- Starting Phase I: HR Snapshot Generation ---")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Input census: {args.census}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Random seed: {args.seed if args.seed is not None else 'Not set'}")

    # Load global params + baseline overrides
    try:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from {args.config}")
    except Exception:
        logger.exception(f"Failed to load or parse config file {args.config}")
        sys.exit(1)

    global_params = cfg.get("global_parameters", {})
    # Validate global plan_rules schema - HR snapshots don't need this, comment out or remove if truly unused
    # gp_pr = global_params.get('plan_rules', {})
    # try:
    #     validated_pr = PlanRules(**gp_pr)
    #     global_params['plan_rules'] = validated_pr.dict() # Keep validated Pydantic model?
    #     logger.debug("Validated global plan_rules schema.")
    # except ValidationError as e:
    #     logger.error(f"Invalid global plan_rules configuration:\n{e}")
    #     sys.exit(1)

    baseline_scenario_name = "baseline"  # Or make this configurable if needed
    baseline = cfg.get("scenarios", {}).get(baseline_scenario_name)
    if baseline is None:
        logger.error(
            f"ERROR: '{baseline_scenario_name}' scenario not found in config file {args.config}."
        )
        sys.exit(1)
    logger.info(f"Using '{baseline_scenario_name}' scenario for HR dynamics.")

    # Build the HR config from global_params + baseline (excluding plan_rules)
    hr_cfg = {**global_params}
    # Carefully override with baseline settings, excluding plan rules and names
    baseline_hr_settings = {
        k: v
        for k, v in baseline.items()
        if k not in ("plan_rules", "scenario_name", "name")
    }
    hr_cfg.update(baseline_hr_settings)

    # Log key HR parameters being used (adjust keys as needed based on your actual config)
    log_hr_params = {
        "start_year": hr_cfg.get("start_year"),
        "projection_years": hr_cfg.get("projection_years"),
        "annual_compensation_increase_rate": hr_cfg.get(
            "annual_compensation_increase_rate"
        ),
        "annual_termination_rate": hr_cfg.get("annual_termination_rate"),
        "new_hire_termination_rate": hr_cfg.get("new_hire_termination_rate"),
        "maintain_headcount": hr_cfg.get("maintain_headcount"),
        # Add other important HR config keys here
    }
    logger.info(f"Effective HR configuration parameters for baseline: {log_hr_params}")
    # For full detail during debugging:
    # logger.debug(f"Full HR configuration: {hr_cfg}")

    # Load census data (supports CSV or Parquet)
    date_cols = [EMP_HIRE_DATE, EMP_TERM_DATE, EMP_BIRTH_DATE]
    census_path = Path(args.census)
    try:
        if not census_path.exists():
            raise FileNotFoundError(f"Census file not found: {census_path}")

        if census_path.suffix.lower() == ".parquet":
            start_df = pd.read_parquet(census_path)
            logger.info(
                f"Loaded {len(start_df)} records from Parquet census: {census_path}"
            )
            # Ensure date columns are datetime
            for col in date_cols:
                if col in start_df.columns:
                    original_dtype = start_df[col].dtype
                    start_df[col] = pd.to_datetime(start_df[col], errors="coerce")
                    if start_df[col].isnull().any():
                        logger.warning(
                            f"Column '{col}' (loaded as {original_dtype}) contained values that could not be parsed as dates."
                        )
        elif census_path.suffix.lower() == ".csv":
            # Check which date cols actually exist before trying to parse
            try:
                cols_in_csv = pd.read_csv(census_path, nrows=0).columns
                parse_dates_present = [c for c in date_cols if c in cols_in_csv]
                start_df = pd.read_csv(census_path, parse_dates=parse_dates_present)
                logger.info(
                    f"Loaded {len(start_df)} records from CSV census: {census_path}"
                )
                # Check for parse errors if parse_dates was used (doesn't coerce by default in read_csv)
                for col in parse_dates_present:
                    if pd.api.types.is_object_dtype(start_df[col]):
                        logger.warning(
                            f"Column '{col}' in CSV could not be fully parsed as dates, check format."
                        )
                        # Optionally force coercion here if needed:
                        # start_df[col] = pd.to_datetime(start_df[col], errors='coerce')
            except Exception as read_err:
                logger.error(f"Error reading CSV file {census_path}: {read_err}")
                sys.exit(1)

        else:
            logger.error(
                f"Unsupported census file format: {census_path}. Please use .csv or .parquet."
            )
            sys.exit(1)

        logger.debug(f"Initial census columns: {start_df.columns.tolist()}")
        logger.debug(f"Initial census data types:\n{start_df.dtypes}")

        # Standardize identifier column name
        source_id_column = "employee_ssn"  # column in input census
        target_id_column = "employee_id"  # column expected by HR projection
        if source_id_column in start_df.columns:
            if target_id_column not in start_df.columns:
                logger.info(
                    f"Renaming column '{source_id_column}' to '{target_id_column}' for consistency."
                )
                start_df.rename(
                    columns={source_id_column: target_id_column}, inplace=True
                )
                logger.debug(f"Columns after rename: {start_df.columns.tolist()}")
            elif source_id_column != target_id_column:
                logger.warning(
                    f"Both '{source_id_column}' and '{target_id_column}' found; using '{target_id_column}'."
                )
        elif target_id_column not in start_df.columns:
            logger.error(
                f"Required ID column '{source_id_column}' or '{target_id_column}' not found in census file. Aborting."
            )
            sys.exit(1)

    except Exception:
        logger.exception(f"Failed to load or process census data from {args.census}")
        sys.exit(1)

    # Seed the global numpy RNG once for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        logger.info(f"NumPy random seed set to {args.seed}.")
    else:
        logger.info("NumPy random seed not set (results may vary).")

    # Run the HR projection once (no further reseeding inside)
    output_dir = Path(args.output)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir}")
    except OSError as e:
        logger.error(f"Could not create output directory {output_dir}: {e}")
        sys.exit(1)

    logger.info("Calling HR projection logic (run_dynamics_for_year)...")
    try:
        hr_snapshots = run_dynamics_for_year(
            start_df, hr_cfg, random_seed=args.seed, parent_logger=logger
        )
        logger.info(
            f"HR projection finished. Generated {len(hr_snapshots)} yearly snapshots."
        )
    except Exception:
        logger.exception("Error occurred during HR projection (run_dynamics_for_year).")
        sys.exit(1)

    # Write out one parquet per year
    logger.info("Writing yearly HR snapshots...")
    for year, df in hr_snapshots.items():
        out_path = output_dir / f"base_run_year{year}.parquet"
        try:
            df.to_parquet(out_path)
            # Original print kept as INFO level, could be DEBUG if too verbose
            logger.info(f"Wrote HR snapshot: {out_path} ({len(df)} records)")
        except Exception as e:
            logger.error(f"Failed to write snapshot file {out_path}: {e}")
            # Decide if you want to exit here or try writing others

    # Generate summary CSV for all years
    logger.info("Generating HR summary file...")
    metrics = []
    total_records_written = 0
    for year, df in hr_snapshots.items():
        headcount = len(df)
        total_records_written += headcount
        # Ensure the compensation column exists before summing
        if EMP_GROSS_COMP in df.columns:
            total_compensation = df[EMP_GROSS_COMP].sum()
        else:
            logger.warning(
                f"Column '{EMP_GROSS_COMP}' not found in snapshot for year {year}. Setting total compensation to 0 for summary."
            )
            total_compensation = 0
        metrics.append(
            {
                "year": year,
                "headcount": headcount,
                "total_gross_compensation": total_compensation,
            }
        )

    if metrics:
        metrics_df = pd.DataFrame(metrics).sort_values("year")
        summary_path = output_dir / "hr_summary.csv"
        try:
            metrics_df.to_csv(
                summary_path, index=False, float_format="%.2f"
            )  # Format floats
            logger.info(f"Wrote HR summary: {summary_path}")
            logger.info(
                f"Summary includes {len(metrics_df)} years, covering {total_records_written} total yearly records."
            )
        except Exception as e:
            logger.error(f"Failed to write HR summary file {summary_path}: {e}")
    else:
        logger.warning("No metrics generated, skipping HR summary file.")

    logger.info("--- Phase I: HR Snapshot Generation Finished Successfully ---")


if __name__ == "__main__":
    main()
