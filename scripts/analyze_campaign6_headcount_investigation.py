#!/usr/bin/env python3
"""
Campaign 6 Headcount Investigation Analysis

This script analyzes the existing Campaign 6 simulation output to investigate
the persistent negative headcount growth issue despite aggressive parameter tuning.

Based on Epic Y investigation plan.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_campaign6_best_config():
    """Load the best configuration from Campaign 6."""
    config_path = project_root / "campaign_6_results" / "best_config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config, str(config_path)


def load_simulation_output(output_dir):
    """Load events and snapshots from simulation output directory."""
    output_path = Path(output_dir)

    # Load all events
    events_files = []
    for year_dir in output_path.glob("year=*"):
        events_file = year_dir / "events.parquet"
        if events_file.exists():
            events_df = pd.read_parquet(events_file)
            events_df["year"] = int(year_dir.name.split("=")[1])
            events_files.append(events_df)

    all_events = pd.concat(events_files, ignore_index=True) if events_files else pd.DataFrame()

    # Load all snapshots
    snapshots = []
    for year_dir in sorted(output_path.glob("year=*")):
        snapshot_file = year_dir / "snapshot.parquet"
        if snapshot_file.exists():
            snapshot_df = pd.read_parquet(snapshot_file)
            snapshot_df["year"] = int(year_dir.name.split("=")[1])
            snapshots.append(snapshot_df)

    return all_events, snapshots


def analyze_termination_events(events_df):
    """Analyze termination events in detail."""
    if events_df.empty:
        return {"error": "No events data available"}

    # Filter termination events
    term_events = events_df[events_df["event_type"] == "EVT_TERM"].copy()

    if term_events.empty:
        return {"error": "No termination events found"}

    analysis = {
        "total_terminations": len(term_events),
        "terminations_by_year": term_events["year"].value_counts().sort_index().to_dict(),
    }

    # Analyze new hire vs experienced terminations
    if "employee_hire_date" in term_events.columns and "event_time" in term_events.columns:
        # Convert to datetime for comparison
        term_events["hire_date"] = pd.to_datetime(term_events["employee_hire_date"])
        term_events["term_date"] = pd.to_datetime(term_events["event_time"])
        term_events["year_hired"] = term_events["hire_date"].dt.year
        term_events["year_termed"] = term_events["term_date"].dt.year

        # Identify new hire terminations (hired and terminated in same year)
        term_events["is_new_hire_term"] = term_events["year_hired"] == term_events["year_termed"]

        new_hire_terms = term_events[term_events["is_new_hire_term"]]
        experienced_terms = term_events[~term_events["is_new_hire_term"]]

        analysis.update(
            {
                "new_hire_terminations": len(new_hire_terms),
                "experienced_terminations": len(experienced_terms),
                "new_hire_term_rate": (
                    len(new_hire_terms) / len(term_events) if len(term_events) > 0 else 0
                ),
                "new_hire_terms_by_year": new_hire_terms["year"]
                .value_counts()
                .sort_index()
                .to_dict(),
                "experienced_terms_by_year": experienced_terms["year"]
                .value_counts()
                .sort_index()
                .to_dict(),
            }
        )

    return analysis


def analyze_hiring_events(events_df):
    """Analyze hiring events in detail."""
    if events_df.empty:
        return {"error": "No events data available"}

    # Filter hiring events
    hire_events = events_df[events_df["event_type"] == "EVT_HIRE"].copy()

    if hire_events.empty:
        return {"error": "No hiring events found"}

    analysis = {
        "total_hires": len(hire_events),
        "hires_by_year": hire_events["year"].value_counts().sort_index().to_dict(),
    }

    return analysis


def analyze_headcount_progression(snapshots):
    """Analyze headcount progression across years."""
    if not snapshots:
        return {"error": "No snapshot data available"}

    progression = {"headcount_by_year": {}, "growth_rates": {}, "active_employees": {}}

    for snapshot in snapshots:
        year = snapshot["year"].iloc[0]

        # Count total employees
        total_count = len(snapshot)
        progression["headcount_by_year"][year] = total_count

        # Count active employees if column exists
        if "employee_active" in snapshot.columns:
            active_count = snapshot["employee_active"].sum()
            progression["active_employees"][year] = int(active_count)
        else:
            progression["active_employees"][year] = total_count

    # Calculate growth rates
    years = sorted(progression["active_employees"].keys())
    for i in range(1, len(years)):
        prev_year = years[i - 1]
        curr_year = years[i]
        prev_count = progression["active_employees"][prev_year]
        curr_count = progression["active_employees"][curr_year]

        if prev_count > 0:
            growth_rate = (curr_count - prev_count) / prev_count
            progression["growth_rates"][curr_year] = growth_rate

    return progression


def calculate_hiring_termination_balance(events_df):
    """Calculate the balance between hiring and terminations."""
    if events_df.empty:
        return {"error": "No events data available"}

    balance = {}

    for year in sorted(events_df["year"].unique()):
        year_events = events_df[events_df["year"] == year]

        hires = len(year_events[year_events["event_type"] == "EVT_HIRE"])
        terms = len(year_events[year_events["event_type"] == "EVT_TERM"])
        net_change = hires - terms

        balance[year] = {"hires": hires, "terminations": terms, "net_change": net_change}

    return balance


def main():
    """Main analysis function."""
    print("=" * 80)
    print("CAMPAIGN 6 HEADCOUNT INVESTIGATION ANALYSIS")
    print("=" * 80)

    try:
        # Load Campaign 6 best config
        config, config_path = load_campaign6_best_config()
        global_params = config.get("global_parameters", {})

        print(f"Campaign 6 Best Config: {config_path}")
        print(f"Target Growth: {global_params.get('target_growth', 'unknown')}")
        print(f"New Hire Rate: {global_params.get('new_hire_rate', 'unknown')}")
        print(
            f"New Hire Termination Rate: {global_params.get('new_hire_termination_rate', 'unknown')}"
        )
        print(f"Maintain Headcount: {global_params.get('maintain_headcount', False)}")
        print()

        # Load simulation output (using the most recent run)
        output_dir = project_root / "tuned" / "output_config_099_20250605_122848" / "Baseline"

        print(f"Loading simulation output from: {output_dir}")
        events_df, snapshots = load_simulation_output(output_dir)

        print(f"Events loaded: {len(events_df)} total events")
        print(f"Snapshots loaded: {len(snapshots)} years")
        print()

        # 1. Analyze termination events
        print("=" * 60)
        print("1. TERMINATION ANALYSIS")
        print("=" * 60)

        term_analysis = analyze_termination_events(events_df)

        if "error" not in term_analysis:
            print(f"Total Terminations: {term_analysis['total_terminations']}")
            print(f"Terminations by Year: {term_analysis['terminations_by_year']}")

            if "new_hire_terminations" in term_analysis:
                print(f"New Hire Terminations: {term_analysis['new_hire_terminations']}")
                print(f"Experienced Terminations: {term_analysis['experienced_terminations']}")
                print(f"New Hire Termination Rate: {term_analysis['new_hire_term_rate']:.1%}")
                print(f"New Hire Terms by Year: {term_analysis['new_hire_terms_by_year']}")
                print(f"Experienced Terms by Year: {term_analysis['experienced_terms_by_year']}")
        else:
            print(f"Error: {term_analysis['error']}")
        print()

        # 2. Analyze hiring events
        print("=" * 60)
        print("2. HIRING ANALYSIS")
        print("=" * 60)

        hire_analysis = analyze_hiring_events(events_df)

        if "error" not in hire_analysis:
            print(f"Total Hires: {hire_analysis['total_hires']}")
            print(f"Hires by Year: {hire_analysis['hires_by_year']}")
        else:
            print(f"Error: {hire_analysis['error']}")
        print()

        # 3. Analyze headcount progression
        print("=" * 60)
        print("3. HEADCOUNT PROGRESSION ANALYSIS")
        print("=" * 60)

        headcount_analysis = analyze_headcount_progression(snapshots)

        if "error" not in headcount_analysis:
            print(f"Headcount by Year: {headcount_analysis['headcount_by_year']}")
            print(f"Active Employees by Year: {headcount_analysis['active_employees']}")
            print(f"Growth Rates by Year: {headcount_analysis['growth_rates']}")

            # Calculate overall growth
            years = sorted(headcount_analysis["active_employees"].keys())
            if len(years) >= 2:
                start_count = headcount_analysis["active_employees"][years[0]]
                end_count = headcount_analysis["active_employees"][years[-1]]
                overall_growth = (end_count - start_count) / start_count
                print(f"Overall Growth Rate: {overall_growth:.1%}")
        else:
            print(f"Error: {headcount_analysis['error']}")
        print()

        # 4. Hiring vs Termination Balance
        print("=" * 60)
        print("4. HIRING VS TERMINATION BALANCE")
        print("=" * 60)

        balance_analysis = calculate_hiring_termination_balance(events_df)

        if "error" not in balance_analysis:
            for year, data in balance_analysis.items():
                print(
                    f"Year {year}: {data['hires']} hires, {data['terminations']} terms, "
                    f"net change: {data['net_change']:+d}"
                )
        else:
            print(f"Error: {balance_analysis['error']}")
        print()

        # 5. Key Findings Summary
        print("=" * 60)
        print("5. KEY FINDINGS SUMMARY")
        print("=" * 60)

        # Check for new hire termination engine functionality
        if "new_hire_terminations" in term_analysis:
            configured_nh_term_rate = global_params.get("new_hire_termination_rate", 0.25)
            actual_nh_terms = term_analysis["new_hire_terminations"]
            total_hires = hire_analysis.get("total_hires", 0)

            if total_hires > 0:
                actual_nh_term_rate = actual_nh_terms / total_hires
                print(f"🔍 NEW HIRE TERMINATION ENGINE STATUS:")
                print(f"   Configured Rate: {configured_nh_term_rate:.1%}")
                print(f"   Actual Rate: {actual_nh_term_rate:.1%}")
                print(f"   Total Hires: {total_hires}")
                print(f"   Actual NH Terms: {actual_nh_terms}")

                if actual_nh_term_rate < configured_nh_term_rate * 0.5:
                    print("   ⚠️  NEW HIRE TERMINATION ENGINE APPEARS TO BE MALFUNCTIONING!")
                    print("   ⚠️  This explains the negative growth despite aggressive parameters!")
                else:
                    print("   ✅ New hire termination engine appears to be working correctly")

        print()
        print("=" * 80)

    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
