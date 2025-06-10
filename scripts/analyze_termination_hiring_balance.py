#!/usr/bin/env python3
"""
Analyze the termination vs hiring balance from recent validation run.

This script investigates why headcount is declining despite positive target growth
by examining the balance between experienced employee terminations and new hires.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_validation_run_data(
    output_dir: str = "output_dev/projection_cli_results",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the most recent validation run data."""
    output_path = Path(output_dir)

    # Load event log
    event_log_path = output_path / "projection_cli_final_cumulative_event_log.parquet"
    events_df = pd.read_parquet(event_log_path) if event_log_path.exists() else pd.DataFrame()

    # Load consolidated snapshots
    snapshots_path = output_path / "consolidated_snapshots.parquet"
    snapshots_df = pd.read_parquet(snapshots_path) if snapshots_path.exists() else pd.DataFrame()

    # Load summary statistics
    summary_path = output_path / "projection_cli_summary_statistics.parquet"
    summary_df = pd.read_parquet(summary_path) if summary_path.exists() else pd.DataFrame()

    logger.info(
        f"Loaded {len(events_df)} events, {len(snapshots_df)} snapshot records, {len(summary_df)} summary records"
    )

    return events_df, snapshots_df, summary_df


def analyze_headcount_progression(snapshots_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze headcount progression across years."""
    if snapshots_df.empty:
        return {}

    progression = {}
    years = sorted(snapshots_df["simulation_year"].unique())

    for year in years:
        year_data = snapshots_df[snapshots_df["simulation_year"] == year]
        total_employees = len(year_data)

        # Count active employees
        if "employee_active" in year_data.columns:
            active_employees = int(year_data["employee_active"].sum())
        else:
            active_employees = total_employees

        progression[year] = {
            "total_employees": total_employees,
            "active_employees": active_employees,
            "terminated_employees": total_employees - active_employees,
        }

    # Calculate growth rates
    if len(years) > 1:
        start_active = progression[years[0]]["active_employees"]
        end_active = progression[years[-1]]["active_employees"]
        total_growth_rate = (end_active - start_active) / start_active
        annual_growth_rate = total_growth_rate / (len(years) - 1)

        progression["summary"] = {
            "start_active": start_active,
            "end_active": end_active,
            "total_growth_rate": total_growth_rate,
            "annual_growth_rate": annual_growth_rate,
            "years_simulated": len(years) - 1,
        }

    return progression


def analyze_terminations_by_tenure(
    events_df: pd.DataFrame, snapshots_df: pd.DataFrame
) -> Dict[str, Any]:
    """Analyze terminations broken down by employee tenure/experience level."""
    if events_df.empty:
        return {}

    # Filter termination events
    term_events = events_df[events_df["event_type"] == "EVT_TERM"].copy()

    if term_events.empty:
        logger.warning("No termination events found")
        return {}

    # Add year column
    term_events["year"] = pd.to_datetime(term_events["event_time"]).dt.year

    # Classify terminations as new hire vs experienced
    # New hire terminations have value_json containing 'new_hire_termination' or meta containing 'New-hire termination'
    nh_by_json = term_events["value_json"].str.contains("new_hire_termination", na=False)
    nh_by_meta = term_events["meta"].str.contains("New-hire termination", na=False)
    nh_terms = term_events[nh_by_json | nh_by_meta]
    exp_terms = term_events[~(nh_by_json | nh_by_meta)]

    analysis = {
        "total_terminations": len(term_events),
        "new_hire_terminations": len(nh_terms),
        "experienced_terminations": len(exp_terms),
        "terminations_by_year": term_events["year"].value_counts().sort_index().to_dict(),
        "nh_terms_by_year": (
            nh_terms["year"].value_counts().sort_index().to_dict() if not nh_terms.empty else {}
        ),
        "exp_terms_by_year": (
            exp_terms["year"].value_counts().sort_index().to_dict() if not exp_terms.empty else {}
        ),
    }

    return analysis


def analyze_hiring_patterns(events_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze hiring patterns across years."""
    if events_df.empty:
        return {}

    # Filter hiring events
    hire_events = events_df[events_df["event_type"] == "EVT_HIRE"].copy()

    if hire_events.empty:
        logger.warning("No hiring events found")
        return {}

    # Add year column
    hire_events["year"] = pd.to_datetime(hire_events["event_time"]).dt.year

    analysis = {
        "total_hires": len(hire_events),
        "hires_by_year": hire_events["year"].value_counts().sort_index().to_dict(),
    }

    return analysis


def calculate_net_flow_analysis(term_analysis: Dict, hire_analysis: Dict) -> Dict[str, Any]:
    """Calculate net employee flow (hires - terminations) by year."""
    if not term_analysis or not hire_analysis:
        return {}

    # Get all years from both analyses
    term_years = set(term_analysis.get("terminations_by_year", {}).keys())
    hire_years = set(hire_analysis.get("hires_by_year", {}).keys())
    all_years = sorted(term_years.union(hire_years))

    net_flow = {}
    total_hires = 0
    total_terms = 0

    for year in all_years:
        year_hires = hire_analysis.get("hires_by_year", {}).get(year, 0)
        year_terms = term_analysis.get("terminations_by_year", {}).get(year, 0)
        year_nh_terms = term_analysis.get("nh_terms_by_year", {}).get(year, 0)
        year_exp_terms = term_analysis.get("exp_terms_by_year", {}).get(year, 0)

        net_change = year_hires - year_terms

        net_flow[year] = {
            "hires": year_hires,
            "total_terminations": year_terms,
            "new_hire_terminations": year_nh_terms,
            "experienced_terminations": year_exp_terms,
            "net_change": net_change,
        }

        total_hires += year_hires
        total_terms += year_terms

    net_flow["summary"] = {
        "total_hires": total_hires,
        "total_terminations": total_terms,
        "net_change": total_hires - total_terms,
        "new_hire_terminations": term_analysis.get("new_hire_terminations", 0),
        "experienced_terminations": term_analysis.get("experienced_terminations", 0),
    }

    return net_flow


def analyze_hiring_algorithm_logic():
    """Analyze the hiring algorithm logic to understand potential issues."""
    print(f"\n🔧 HIRING ALGORITHM ANALYSIS:")
    print(f"  Current Logic (manage_headcount_to_exact_target):")
    print(f"    1. Calculate target EOY actives = SOY_actives * (1 + target_growth)")
    print(f"    2. Determine survivors = SOY_actives - num_markov_exits_existing")
    print(f"    3. Calculate net_actives_needed = target_EOY - survivors")
    print(f"    4. Gross up for NH attrition = net_needed / (1 - nh_term_rate)")
    print(f"")
    print(f"  Key Question: Does 'num_markov_exits_existing' capture ALL expected")
    print(f"  experienced employee terminations for the year, or only those that")
    print(f"  have already occurred by the time hiring decisions are made?")
    print(f"")
    print(f"  If it only captures terminations that have already occurred,")
    print(f"  then the algorithm systematically under-hires because it doesn't")
    print(f"  account for experienced employees who will terminate later in the year.")


def main():
    """Main analysis function."""
    logger.info("Starting termination vs hiring balance analysis...")

    # Load data
    events_df, snapshots_df, summary_df = load_validation_run_data()

    if events_df.empty and snapshots_df.empty:
        logger.error("No data found in output directory")
        return

    # Analyze headcount progression
    logger.info("Analyzing headcount progression...")
    headcount_analysis = analyze_headcount_progression(snapshots_df)

    # Analyze terminations
    logger.info("Analyzing termination patterns...")
    termination_analysis = analyze_terminations_by_tenure(events_df, snapshots_df)

    # Analyze hiring
    logger.info("Analyzing hiring patterns...")
    hiring_analysis = analyze_hiring_patterns(events_df)

    # Calculate net flow
    logger.info("Calculating net employee flow...")
    net_flow_analysis = calculate_net_flow_analysis(termination_analysis, hiring_analysis)

    # Print results
    print("\n" + "=" * 80)
    print("TERMINATION VS HIRING BALANCE ANALYSIS")
    print("=" * 80)

    # Headcount progression
    if headcount_analysis:
        print(f"\n📊 HEADCOUNT PROGRESSION:")
        for year, data in headcount_analysis.items():
            if year != "summary":
                print(
                    f"  Year {year}: {data['active_employees']} active, {data['terminated_employees']} terminated"
                )

        if "summary" in headcount_analysis:
            summary = headcount_analysis["summary"]
            print(f"\n  📈 GROWTH SUMMARY:")
            print(f"    Start: {summary['start_active']} active employees")
            print(f"    End: {summary['end_active']} active employees")
            print(f"    Total Growth: {summary['total_growth_rate']:.2%}")
            print(f"    Annual Growth: {summary['annual_growth_rate']:.2%}")

    # Net flow analysis
    if net_flow_analysis:
        print(f"\n🔄 NET EMPLOYEE FLOW ANALYSIS:")
        for year, data in net_flow_analysis.items():
            if year != "summary":
                print(
                    f"  Year {year}: {data['hires']} hires, {data['total_terminations']} terms "
                    f"({data['new_hire_terminations']} NH, {data['experienced_terminations']} Exp), "
                    f"net: {data['net_change']:+d}"
                )

        if "summary" in net_flow_analysis:
            summary = net_flow_analysis["summary"]
            print(f"\n  📋 FLOW SUMMARY:")
            print(f"    Total Hires: {summary['total_hires']}")
            print(f"    Total Terminations: {summary['total_terminations']}")
            print(f"      - New Hire Terms: {summary['new_hire_terminations']}")
            print(f"      - Experienced Terms: {summary['experienced_terminations']}")
            print(f"    Net Change: {summary['net_change']:+d}")

    # Analysis and hypothesis
    print(f"\n🔍 ANALYSIS & HYPOTHESIS:")

    if net_flow_analysis and "summary" in net_flow_analysis:
        summary = net_flow_analysis["summary"]
        total_hires = summary["total_hires"]
        exp_terms = summary["experienced_terminations"]
        nh_terms = summary["new_hire_terminations"]

        print(f"  1. Experienced Employee Attrition Impact:")
        print(f"     - {exp_terms} experienced employees terminated")
        if total_hires > 0:
            print(f"     - This represents {exp_terms/total_hires:.1%} of total hires")

        print(f"  2. New Hire Termination Engine:")
        print(f"     - {nh_terms} new hire terminations occurred")
        print(f"     - Expected rate: 25% (from config)")
        if total_hires > 0:
            actual_nh_rate = nh_terms / total_hires
            print(f"     - Actual rate: {actual_nh_rate:.1%}")
            print(f"     - Difference: {actual_nh_rate - 0.25:.1%}")

        print(f"  3. Hiring Algorithm Assessment:")
        print(f"     - The hiring algorithm may be underestimating experienced attrition")
        print(f"     - Net result: {summary['net_change']:+d} employees vs target growth")

        if headcount_analysis and "summary" in headcount_analysis:
            target_growth = 0.03  # From config
            actual_growth = headcount_analysis["summary"]["annual_growth_rate"]
            print(f"     - Target growth: {target_growth:.1%}")
            print(f"     - Actual growth: {actual_growth:.1%}")
            print(f"     - Gap: {actual_growth - target_growth:.1%}")

    # Analyze hiring algorithm logic
    analyze_hiring_algorithm_logic()


if __name__ == "__main__":
    main()
