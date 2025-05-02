#!/usr/bin/env python3
"""
utils/preprocess_census.py - Preprocess utility for census data: adds derived fields (bools, status, eligibility, etc.) for use in simulation output and downstream analysis.
"""
import os
import sys
import logging
from pathlib import Path
from typing import Union, Optional
logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import argparse
from cost_model.utils.data_processing import assign_employment_status
from cost_model.rules.eligibility import apply as apply_eligibility
from cost_model.rules.validators import PlanRules
from cost_model.utils.constants import ACTIVE_STATUS, INACTIVE_STATUS
from cost_model.utils.columns import (
    EMP_SSN, EMP_ROLE, EMP_BIRTH_DATE, EMP_HIRE_DATE, EMP_TERM_DATE,
    EMP_GROSS_COMP, EMP_PLAN_YEAR_COMP, EMP_CAPPED_COMP,
    EMP_DEFERRAL_RATE, EMP_CONTR,
    EMPLOYER_CORE, EMPLOYER_MATCH,
    ELIGIBILITY_ENTRY_DATE, IS_PARTICIPATING, IS_ELIGIBLE,
    RAW_TO_STD_COLS, to_nullable_bool,
    DATE_COLS
)

def _get_plan_rules(cfg: dict) -> dict:
    """Extract plan_rules from either global_parameters or scenario_config."""
    if 'global_parameters' in cfg:
        return cfg['global_parameters'].get('plan_rules', {})
    return cfg.get('plan_rules', {})

def preprocess_census(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    simulation_year: Optional[int] = None,
    write_parquet: bool = False
) -> pd.DataFrame:
    """
    Reads raw census data, applies status and eligibility rules, renames columns,
    formats dates, writes CSV (and optional Parquet), and returns the processed DataFrame.

    Args:
        input_path: Path or str for source census CSV.
        output_path: Path or str for target CSV path.
        config_path: Optional Path or str to plan rules YAML config.
        simulation_year: Optional int for simulation year (defaults to current year).
        write_parquet: Whether to write Parquet alongside CSV.

    Returns:
        Processed pandas DataFrame with standardized columns.
    """
    inp = Path(input_path)
    out = Path(output_path)
    # Load CSV, parsing standardized date columns if present
    tmp_cols = pd.read_csv(inp, nrows=0).columns
    parse_dates = [col for col in DATE_COLS if col in tmp_cols]
    df = pd.read_csv(inp, parse_dates=parse_dates)
    # Ensure SSN is string to avoid numeric coercion issues
    if EMP_SSN in df:
        df[EMP_SSN] = df[EMP_SSN].astype(str)
    # Add employment status columns
    year = simulation_year
    if year is None:
        # Infer from data, fallback to current year
        year = pd.Timestamp.today().year
    df = assign_employment_status(df, year)
    # Add eligibility columns if config provided
    if config_path:
        if not Path(config_path).exists():
            raise RuntimeError(f"Config not found: {config_path}")
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        plan_rules_dict = _get_plan_rules(config)
        plan_rules = PlanRules(**plan_rules_dict)
        sim_year_end = pd.Timestamp(f"{year}-12-31")
        df = apply_eligibility(df, plan_rules.eligibility, sim_year_end)
        # Calculate contributions to retain contribution columns before pruning
        from cost_model.rules.contributions import apply as apply_contributions
        # Apply contributions if enabled in plan rules
        if plan_rules.contributions.enabled:
            year_start = pd.Timestamp(f"{year}-01-01")
            df = apply_contributions(
                df,
                plan_rules.contributions,
                plan_rules.employer_match,
                plan_rules.employer_nec,
                plan_rules.irs_limits,
                year,
                year_start,
                sim_year_end
            )
    # Rename columns using centralized mapping
    df.rename(columns=RAW_TO_STD_COLS, inplace=True, errors='ignore')
    # Keep only standardized columns and essential fields
    # Include contribution and rate columns
    keep = set(RAW_TO_STD_COLS.values()) | {
        EMP_SSN, EMP_ROLE, EMP_GROSS_COMP, EMP_PLAN_YEAR_COMP, EMP_CAPPED_COMP, EMP_DEFERRAL_RATE, EMP_CONTR, EMPLOYER_CORE, EMPLOYER_MATCH, 
        IS_ELIGIBLE, IS_PARTICIPATING, ELIGIBILITY_ENTRY_DATE
    }
    dropped = set(df.columns) - keep
    if dropped:
        logger.debug("Dropping columns: %s", dropped)
    df = df.loc[:, df.columns.intersection(keep)]
    # Validate presence of eligibility entry date if eligibility logic was applied
    if config_path and ELIGIBILITY_ENTRY_DATE not in df.columns:
        raise RuntimeError(f"Missing eligibility column after rename: {ELIGIBILITY_ENTRY_DATE}")
    # Validate presence of required columns
    required = {EMP_SSN, EMP_GROSS_COMP}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns after rename: {missing}")
    # Set is_participating and is_eligible flags
    # Boolean flags for participation
    df[IS_PARTICIPATING] = to_nullable_bool(df.get(EMP_CONTR, 0) > 0)
    # Boolean flag for eligibility ensure series input
    eligible_series = df[IS_ELIGIBLE] if IS_ELIGIBLE in df else pd.Series(False, index=df.index)
    df[IS_ELIGIBLE] = to_nullable_bool(eligible_series)
    # Final date formatting
    for col in [EMP_HIRE_DATE, EMP_TERM_DATE, EMP_BIRTH_DATE, ELIGIBILITY_ENTRY_DATE]:
        if col in df:
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')

    # Save to output (CSV)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"Preprocessed census written to {out} ({len(df)} rows)")
    # Save Parquet only if requested
    if write_parquet:
        parquet_path = out.with_suffix('.parquet')
        try:
            import pyarrow  # noqa: F401
            df.to_parquet(parquet_path, index=False)
            logger.info(f"Preprocessed census also written to {parquet_path}")
        except ImportError:
            logger.info("pyarrow not installed; skipping Parquet output.")
    return df

def main() -> pd.DataFrame:
    # Configure logging for CLI output
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Preprocess census data for simulation.")
    parser.add_argument('--input', default='data/census_data.csv', help='Input census_data.csv path')
    parser.add_argument('--output', default='data/census_preprocessed.csv', help='Output CSV path')
    parser.add_argument('--config', required=False, help='Config YAML for eligibility logic')
    parser.add_argument('--year', type=int, required=False, help='Simulation year (default: infer)')
    parser.add_argument('--parquet', action='store_true', help='Also write output as Parquet')
    parser.add_argument('--verbose', action='store_true', help='Show DataFrame shape and head')
    args = parser.parse_args()
    # Run preprocessing and optionally return df
    df = preprocess_census(
        args.input,
        args.output,
        args.config,
        args.year,
        args.parquet
    )
    if args.verbose:
        print(f"DataFrame shape: {df.shape}")
        print(df.head())
    return df

if __name__ == "__main__":
    main()
