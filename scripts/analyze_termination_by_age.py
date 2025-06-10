#!/usr/bin/env python3
"""Analyze termination patterns by age to verify age multipliers are working."""

import json
from pathlib import Path

import pandas as pd


def analyze_terminations_by_age(scenario_dir, scenario_name):
    """Analyze termination patterns by age band for a scenario."""
    print(f"\n{'='*60}")
    print(f"📊 TERMINATION ANALYSIS: {scenario_name}")
    print(f"{'='*60}")

    scenario_path = Path(scenario_dir)
    scenario_dirs = [d for d in scenario_path.iterdir() if d.is_dir()]
    if not scenario_dirs:
        print("❌ No scenario directory found")
        return

    scenario_subdir = scenario_dirs[0]

    # Analyze 2025 data (first year)
    year_2025_dir = scenario_subdir / "year=2025"
    if not year_2025_dir.exists():
        print("❌ No 2025 data found")
        return

    # Load snapshot and events
    snapshot_path = year_2025_dir / "snapshot.parquet"
    events_path = year_2025_dir / "events.parquet"

    if not snapshot_path.exists() or not events_path.exists():
        print("❌ Missing snapshot or events file")
        return

    snapshot = pd.read_parquet(snapshot_path)
    events = pd.read_parquet(events_path)

    # Filter termination events
    term_events = events[events["event_type"] == "EVT_TERM"]

    print(f"📈 Total employees in 2025: {len(snapshot)}")
    print(f"🚪 Total terminations in 2025: {len(term_events)}")
    print(f"📊 Overall termination rate: {len(term_events)/len(snapshot)*100:.1f}%")

    # Analyze by age band
    if "employee_age_band" in snapshot.columns:
        print(f"\n🎯 TERMINATION RATES BY AGE BAND")
        print(f"{'Age Band':<10} {'Total':<8} {'Terminated':<12} {'Rate':<8} {'Expected'}")
        print("-" * 55)

        # Get terminated employee IDs
        terminated_ids = set(term_events["employee_id"].unique())

        # Calculate rates by age band
        age_band_stats = {}
        for age_band in sorted(snapshot["employee_age_band"].unique()):
            band_employees = snapshot[snapshot["employee_age_band"] == age_band]
            band_terminated = len(
                [emp for emp in band_employees["employee_id"] if emp in terminated_ids]
            )
            band_total = len(band_employees)
            band_rate = (band_terminated / band_total * 100) if band_total > 0 else 0

            # Expected multiplier based on scenario
            if scenario_name == "High Early Attrition":
                expected_multipliers = {
                    "<30": 2.5,
                    "30-39": 1.8,
                    "40-49": 1.0,
                    "50-59": 0.7,
                    "60-65": 0.5,
                }
            elif scenario_name == "High Late Attrition":
                expected_multipliers = {
                    "<30": 0.5,
                    "30-39": 0.7,
                    "40-49": 1.0,
                    "50-59": 2.0,
                    "60-65": 3.5,
                }
            else:  # Baseline
                expected_multipliers = {
                    "<30": 1.0,
                    "30-39": 1.0,
                    "40-49": 1.0,
                    "50-59": 1.0,
                    "60-65": 1.0,
                }

            expected = expected_multipliers.get(age_band, 1.0)

            age_band_stats[age_band] = {
                "total": band_total,
                "terminated": band_terminated,
                "rate": band_rate,
                "expected_multiplier": expected,
            }

            print(
                f"{age_band:<10} {band_total:<8} {band_terminated:<12} {band_rate:<7.1f}% {expected}x"
            )

        # Analysis
        print(f"\n🔍 ANALYSIS")
        print("-" * 30)

        if scenario_name == "High Early Attrition":
            young_rate = age_band_stats.get("<30", {}).get("rate", 0)
            old_rate = age_band_stats.get("60-65", {}).get("rate", 0)
            print(f"Young (<30) termination rate: {young_rate:.1f}%")
            print(f"Old (60-65) termination rate: {old_rate:.1f}%")
            if young_rate > old_rate:
                print("✅ Young employees are terminating at higher rates (as expected)")
            else:
                print("❌ Age multipliers may not be working correctly")

        elif scenario_name == "High Late Attrition":
            young_rate = age_band_stats.get("<30", {}).get("rate", 0)
            old_rate = age_band_stats.get("60-65", {}).get("rate", 0)
            print(f"Young (<30) termination rate: {young_rate:.1f}%")
            print(f"Old (60-65) termination rate: {old_rate:.1f}%")
            if old_rate > young_rate:
                print("✅ Old employees are terminating at higher rates (as expected)")
            else:
                print("❌ Age multipliers may not be working correctly")
    else:
        print("❌ No age band data found in snapshot")


def main():
    """Main analysis function."""
    print("🔍 TERMINATION PATTERN ANALYSIS BY AGE")
    print("This analysis checks if age multipliers are actually affecting termination rates")

    scenarios = [
        ("output_age_test_baseline", "Baseline"),
        ("output_age_test_high_early_attrition", "High Early Attrition"),
        ("output_age_test_high_late_attrition", "High Late Attrition"),
    ]

    for scenario_dir, scenario_name in scenarios:
        analyze_terminations_by_age(scenario_dir, scenario_name)

    print(f"\n{'='*60}")
    print("🎯 SUMMARY")
    print(f"{'='*60}")
    print("This analysis helps verify if age multipliers are being applied correctly")
    print("by examining actual termination rates by age band in each scenario.")


if __name__ == "__main__":
    main()
