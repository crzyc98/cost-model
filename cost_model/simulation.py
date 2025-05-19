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
from cost_model.utils.columns import EMP_ID, EMP_ROLE, EMP_TENURE_BAND

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
        from cost_model.engines.run_one_year_engine import run_one_year
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

    # 3. Load census and bootstrap events/snapshot
    try:
        if str(input_census_path).endswith('.csv'):
            census_df = pd.read_csv(input_census_path)
        else:
            census_df = pd.read_parquet(input_census_path)
    except Exception:
        logger.error(f"Error loading census file: {input_census_path}")
        raise
    # Bootstrap events (empty for now, or could seed with hires)
    events = pd.DataFrame()
    # Build snapshot
    from cost_model.state.snapshot import build_full as snapmod_build_full
    snap = snapmod_build_full(census_df, year)
    # Patch in role and tenure_band if present using standardized names
    col_map = {
        EMP_ROLE: 'role',
        EMP_TENURE_BAND: 'tenure_band'
    }
    for std_col, census_col in col_map.items():
        if census_col in census_df.columns:
            snap[std_col] = snap.index.map(census_df.set_index(EMP_ID)[census_col].to_dict()).astype(str)

    # 4. Load hazard table
    hazard_path = Path('cost_model/state/hazard_table.csv')
    hazard = pd.read_csv(hazard_path)

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
        snapshot_df, all_events = run_one_year(
            year=year,
            prev_snapshot=snapshot_df,
            event_log=all_events,
            hazard_table=hazard,
            rng=rng,
            deterministic_term=True
        )
        # Ensure simulation_year is set in the snapshot
        snapshot_df['simulation_year'] = year
        yearly_snapshots[year] = snapshot_df.copy()
        # Save snapshot and events for this year
        year_dir = scenario_output_dir / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        snapshot_df.to_parquet(year_dir / "snapshot.parquet")
        all_events.to_parquet(year_dir / "events.parquet")
        logger.info(f"Saved outputs for year {year}")

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
    required_cols = {
        'Active Headcount',
        'Eligible Count',
        'Participant Count',
        'Participation Rate',
        'Total Compensation'
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Summary is missing required columns: {missing}")
    # No zero headcounts
    if (df['Active Headcount'] == 0).any():
        zero_years = df.index[df['Active Headcount'] == 0].tolist()
        raise ValueError(f"Year(s) with zero headcount: {zero_years}")
    # Internal consistency: hires - terms + initial == final
    if 'Hires' in df.columns and 'Terms' in df.columns:
        diffs = df['Hires'] - df['Terms'] + df['Active Headcount'].iloc[0]
        if not np.allclose(diffs.cumsum().iloc[-1], df['Active Headcount'].iloc[-1]):
            raise ValueError("Headcount evolution inconsistency detected.")
