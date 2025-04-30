#!/usr/bin/env python3
"""
scripts/run_retirement_plan_abm.py

Load scenario config, ingest census via a common loader, run the ABM, and save outputs.
"""
import sys
import argparse
import logging
from pathlib import Path

import pandas as pd
from decimal import Decimal

# reuse our centralized loader
from data_processing import load_and_clean_census  
from model.retirement_model import RetirementPlanModel


def str_to_decimal_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Convert a numeric column to Decimal safely."""
    if col in df:
        df[col] = df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else Decimal('0.00'))
    return df


def adjust_year(df: pd.DataFrame, step_col: str, start_year: int) -> pd.DataFrame:
    """Rename `step_col` → Year = step + start_year - 1, drop `step_col`."""
    df = df.reset_index() if step_col not in df.columns else df.copy()
    if step_col in df:
        df['Year'] = df[step_col] + start_year - 1
        df = df.drop(columns=[step_col])
    return df


def run_simulation(config_path: Path, census_path: Path, output_prefix: Path, until_year: int = None):
    logger.info("Loading configuration from %s", config_path)
    if not config_path.exists():
        logger.error("Config file not found at %s", config_path)
        sys.exit(1)
    try:
        config = pd.read_yaml(config_path)
    except Exception as e:
        logger.error("Failed to parse config: %s", e)
        sys.exit(1)

    logger.info("Loading census data via common loader")
    census_df = load_and_clean_census(str(census_path), {'required': ['ssn','birth_date','hire_date','gross_compensation']})
    if census_df is None:
        logger.error("Census loader failed for %s", census_path)
        sys.exit(1)

    # Convert key numeric columns to Decimal
    for col in ['gross_compensation', 'deferral_rate']:
        census_df = str_to_decimal_col(census_df, col)

    # Initialize model
    logger.info("Initializing RetirementPlanModel")
    try:
        model = RetirementPlanModel(initial_census_df=census_df, scenario_config=config)
    except Exception as e:
        logger.error("Error initializing model: %s", e, exc_info=True)
        sys.exit(1)

    projection_years = config.get('projection_years', 5)
    start_year = config.get('start_year', pd.Timestamp.now().year)
    logger.info("Running simulation from %d for %d years", start_year, projection_years)

    for i in range(projection_years):
        if until_year and (start_year + i) > until_year:
            logger.info("Stopping early at year %d per --until-year flag", until_year)
            break
        try:
            model.step()
        except Exception as e:
            logger.error("Error at simulation step %d: %s", i+1, e, exc_info=True)
            sys.exit(1)

    # Collect outputs
    model_df = adjust_year(model.datacollector.get_model_vars_dataframe(), 'YearStep', start_year)
    agent_df = adjust_year(model.datacollector.get_agent_vars_dataframe(), 'Step', start_year)

    # Prepare output directory
    out_dir = output_prefix.parent or Path('.')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write CSVs
    model_path = out_dir / f"{output_prefix.name}_model_results.csv"
    agent_path = out_dir / f"{output_prefix.name}_agent_results.csv"
    model_df.to_csv(model_path, index=False)
    agent_df.to_csv(agent_path, index=False)
    logger.info("Saved model results to %s", model_path)
    logger.info("Saved agent results to %s", agent_path)

    # Print small summary and growth metrics via logging
    table = model_df.set_index('Year')[[
        'Continuous Active', 'New Hire Active', 'Experienced Terminated', 'New Hire Terminated'
    ]]
    growth = table.assign(Total=lambda df: df[['Continuous Active','New Hire Active']].sum(axis=1))
    growth['Growth'] = growth['Total'].diff().fillna(0).astype(int)
    growth['PctGrowth'] = growth['Growth'] / growth['Total'].shift()
    logger.info("Yearly growth metrics:\n%s", growth.to_string())

    logger.info("Simulation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ABM retirement plan simulation.")
    parser.add_argument('--config',    type=Path, default=Path('data/config.yaml'),
                        help="Path to scenario YAML config")
    parser.add_argument('--census',    type=Path, default=Path('data/census_data.csv'),
                        help="Path to initial census CSV")
    parser.add_argument('--output',    type=Path, default=Path('output/abm_simulation'),
                        help="Prefix (directory + basename) for output files")
    parser.add_argument('--until-year',type=int,
                        help="Optional: stop simulation after this calendar year")
    parser.add_argument('--debug',     action='store_true',
                        help="Enable debug-level logging")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)

    run_simulation(args.config, args.census, args.output, until_year=args.until_year)
    sys.exit(0)