#!/usr/bin/env python3
"""
scripts/run_projection.py
Run retirement plan projection simulations for multiple scenarios.
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from data_processing import load_and_clean_census

from cost_model import load_scenarios, project_census

# Configure logging globally
logger = logging.getLogger(__name__)


def aggregate_scenario_results(yearly_data: dict, scenario_config: dict) -> pd.DataFrame:
    """
    Summarize key metrics per year for a scenario.
    """
    records = []
    for year, df in yearly_data.items():
        active = df[df.get("is_active", False)]
        headcount = len(active)
        pre = active.get("pre_tax_contributions", pd.Series(0))
        match = active.get("employer_match_contribution", pd.Series(0))
        nec = active.get("employer_core_contribution", pd.Series(0))
        total_contrib = (pre + match + nec).sum()
        total_comp = active.get("plan_year_compensation", pd.Series(0)).sum()
        records.append(
            {
                "Scenario": scenario_config.get("scenario_name", ""),
                "Year": int(year),
                "Headcount": headcount,
                "TotalContributions": total_contrib,
                "TotalCompensation": total_comp,
            }
        )
    return pd.DataFrame(records)


def combine_raw(name: str, data: dict, out_dir: Path):
    """Combine per-year dataframes and save a single CSV."""
    dfs = [df.assign(Year=yr) for yr, df in data.items() if not df.empty]
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
    else:
        combined = pd.DataFrame()
    path = out_dir / f"{name}_combined_raw.csv"
    combined.to_csv(path, index=False)
    logger.info("Saved combined raw for %s: %s", name, path)


def main():
    parser = argparse.ArgumentParser(description="Run projection for multiple scenarios")
    parser.add_argument("--config", type=Path, required=True, help="YAML scenarios config")
    parser.add_argument("--census", type=Path, required=True, help="Input census CSV")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"), help="Base output directory"
    )
    parser.add_argument("--raw", action="store_true", help="Save raw per-year outputs")
    args = parser.parse_args()

    # Prepare output directory
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = out_dir / "projection.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger.info("Starting projection pipeline")

    # Load scenarios
    scenarios = load_scenarios(args.config)
    if not scenarios:
        logger.error("No scenarios found in %s", args.config)
        sys.exit(1)

    # Load and clean census
    census_df = load_and_clean_census(
        str(args.census),
        {"required": ["ssn", "birth_date", "hire_date", "gross_compensation"]},
    )
    if census_df is None:
        logger.error("Failed to load census data from %s", args.census)
        sys.exit(1)

    summary_list = []
    raw_outputs = {}

    for sc in scenarios:
        name = sc.get("scenario_name", sc.get("name", ""))
        years = sc.get("projection_years")
        if not isinstance(years, int):
            logger.error("'%s' projection_years must be an integer", name)
            continue
        logger.info("Running scenario %s", name)
        try:
            results = project_census(start_df=census_df.copy(), scenario_config=sc)
            # Save summary
            summary_df = aggregate_scenario_results(results, sc)
            summary_file = out_dir / f"{name}_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            logger.info("Saved summary for %s: %s", name, summary_file)
            summary_list.append(summary_df)

            # Save raw if requested
            if args.raw:
                raw_dir = out_dir / f"{name}_raw"
                raw_dir.mkdir(exist_ok=True)
                for year, df in results.items():
                    f = raw_dir / f"year_{year}.csv"
                    df.to_csv(f, index=False)
                raw_outputs[name] = results
        except Exception as e:
            logger.error("Scenario %s failed: %s", name, e, exc_info=True)
            continue

    # Combine all summaries
    summary_list = [df for df in summary_list if not df.empty]
    if summary_list:
        combined = pd.concat(summary_list, ignore_index=True)
        combined_file = out_dir / "all_scenarios_summary.csv"
        combined.to_csv(combined_file, index=False)
        logger.info("Saved combined summary: %s", combined_file)
    else:
        logger.warning("No scenario summaries generated")

    # Combine raw outputs if any
    if args.raw and raw_outputs:
        for name, data in raw_outputs.items():
            combine_raw(name, data, out_dir)

    logger.info("Projection pipeline complete")


if __name__ == "__main__":
    main()
