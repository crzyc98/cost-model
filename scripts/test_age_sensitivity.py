#!/usr/bin/env python3
"""
Test script for verifying age sensitivity in termination engine.

This script runs three test scenarios:
1. Baseline: Neutral age multipliers (all 1.0)
2. High Early Attrition: Young employees have higher termination rates
3. High Late Attrition: Older employees have higher termination rates

Expected outcomes:
- Baseline: Stable age distribution over time
- High Early Attrition: Workforce should skew older over time
- High Late Attrition: Workforce should skew younger over time
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_simulation_with_hazard_config(hazard_config: str, output_suffix: str) -> str:
    """
    Run a simulation with a specific hazard configuration.

    Args:
        hazard_config: Name of the hazard configuration file (e.g., 'hazard_baseline_test.yaml')
        output_suffix: Suffix for the output directory

    Returns:
        Path to the output directory
    """
    output_dir = f"output_age_test_{output_suffix}"

    # Set environment variable to override hazard configuration
    env = os.environ.copy()
    env["HAZARD_CONFIG_FILE"] = hazard_config

    # Run the simulation
    cmd = [
        sys.executable,
        "scripts/run_simulation.py",
        "--config",
        "config/test_age_sensitivity_baseline.yaml",
        "--scenario",
        "baseline",
        "--census",
        "data/census_template.parquet",
        "--output",
        output_dir,
        "--debug",
    ]

    print(f"\n{'='*60}")
    print(f"Running simulation with hazard config: {hazard_config}")
    print(f"Output directory: {output_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            print(f"ERROR: Simulation failed with return code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return None
        else:
            print("✅ Simulation completed successfully")
            return output_dir

    except subprocess.TimeoutExpired:
        print("ERROR: Simulation timed out after 5 minutes")
        return None
    except Exception as e:
        print(f"ERROR: Failed to run simulation: {e}")
        return None


def analyze_age_distribution(output_dir: str, scenario_name: str) -> Dict:
    """
    Analyze age distribution from simulation output.

    Args:
        output_dir: Path to simulation output directory
        scenario_name: Name of the scenario for labeling

    Returns:
        Dictionary with analysis results
    """
    try:
        # Find the scenario subdirectory (it's named after the scenario)
        output_path = Path(output_dir)
        scenario_dirs = [d for d in output_path.iterdir() if d.is_dir()]

        if not scenario_dirs:
            print(f"WARNING: No scenario directories found in {output_dir}")
            return None

        scenario_dir = scenario_dirs[0]  # Take the first (should be only one)
        print(f"📁 Found scenario directory: {scenario_dir.name}")

        # Look for yearly snapshot directories
        year_dirs = [d for d in scenario_dir.iterdir() if d.is_dir() and d.name.startswith("year=")]

        if not year_dirs:
            print(f"WARNING: No yearly directories found in {scenario_dir}")
            return None

        # Load snapshots from each year
        all_snapshots = []
        years = []

        for year_dir in sorted(year_dirs):
            year_str = year_dir.name.replace("year=", "")
            year = int(year_str)
            years.append(year)

            snapshot_path = year_dir / "snapshot.parquet"
            if snapshot_path.exists():
                df_year = pd.read_parquet(snapshot_path)
                df_year["year"] = year  # Add year column
                all_snapshots.append(df_year)
                print(f"📊 Loaded {len(df_year)} employees for year {year}")
            else:
                print(f"WARNING: No snapshot found for year {year}")

        if not all_snapshots:
            print(f"WARNING: No snapshots loaded from {output_dir}")
            return None

        # Combine all snapshots
        df = pd.concat(all_snapshots, ignore_index=True)

        # Ensure we have age data
        if "employee_age" not in df.columns:
            print(f"WARNING: No age data found in snapshots")
            return None

        # Calculate age statistics by year
        age_stats = {}
        for year in sorted(years):
            year_data = df[df["year"] == year]
            age_data = year_data["employee_age"].dropna()

            if len(age_data) > 0:
                age_bands = (
                    year_data["employee_age_band"].value_counts().to_dict()
                    if "employee_age_band" in year_data.columns
                    else {}
                )
                age_stats[year] = {
                    "count": len(age_data),
                    "mean_age": age_data.mean(),
                    "median_age": age_data.median(),
                    "std_age": age_data.std(),
                    "min_age": age_data.min(),
                    "max_age": age_data.max(),
                    "age_bands": age_bands,
                }
            else:
                age_stats[year] = None

        final_year_data = df[df["year"] == max(years)]

        return {
            "scenario": scenario_name,
            "output_dir": output_dir,
            "age_stats": age_stats,
            "total_employees": len(final_year_data),
            "years": sorted(years),
        }

    except Exception as e:
        print(f"ERROR: Failed to analyze {output_dir}: {e}")
        import traceback

        traceback.print_exc()
        return None


def create_comparison_report(results: List[Dict]) -> None:
    """
    Create a comparison report of age sensitivity test results.

    Args:
        results: List of analysis results from different scenarios
    """
    print(f"\n{'='*80}")
    print("AGE SENSITIVITY TEST RESULTS")
    print(f"{'='*80}")

    # Filter out None results
    valid_results = [r for r in results if r is not None]

    if not valid_results:
        print("❌ No valid results to analyze")
        return

    # Create summary table
    print(f"\n{'Scenario':<25} {'Years':<15} {'Final Count':<12} {'Mean Age Change':<15}")
    print("-" * 70)

    for result in valid_results:
        scenario = result["scenario"]
        years = result["years"]
        final_count = result["total_employees"]

        # Calculate age change from first to last year
        age_stats = result["age_stats"]
        first_year = min(years)
        last_year = max(years)

        if age_stats[first_year] and age_stats[last_year]:
            age_change = age_stats[last_year]["mean_age"] - age_stats[first_year]["mean_age"]
            age_change_str = f"{age_change:+.2f} years"
        else:
            age_change_str = "N/A"

        years_str = f"{first_year}-{last_year}"
        print(f"{scenario:<25} {years_str:<15} {final_count:<12} {age_change_str:<15}")

    # Detailed year-by-year analysis
    print(f"\n{'='*80}")
    print("DETAILED YEAR-BY-YEAR ANALYSIS")
    print(f"{'='*80}")

    for result in valid_results:
        print(f"\n📊 {result['scenario'].upper()}")
        print("-" * 50)

        age_stats = result["age_stats"]
        print(f"{'Year':<6} {'Count':<8} {'Mean Age':<10} {'Median Age':<12} {'Age Bands'}")
        print("-" * 70)

        for year in result["years"]:
            if age_stats[year]:
                stats = age_stats[year]
                bands_str = ", ".join([f"{k}:{v}" for k, v in stats["age_bands"].items()])
                if len(bands_str) > 30:
                    bands_str = bands_str[:27] + "..."

                print(
                    f"{year:<6} {stats['count']:<8} {stats['mean_age']:<10.1f} {stats['median_age']:<12.1f} {bands_str}"
                )
            else:
                print(f"{year:<6} {'N/A':<8} {'N/A':<10} {'N/A':<12} N/A")

    # Interpretation
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")

    print("\n🔍 Expected Outcomes:")
    print("• Baseline: Stable age distribution (minimal change)")
    print("• High Early Attrition: Workforce ages over time (positive age change)")
    print("• High Late Attrition: Workforce gets younger over time (negative age change)")

    print("\n📈 Actual Results:")
    for result in valid_results:
        scenario = result["scenario"]
        age_stats = result["age_stats"]
        years = result["years"]

        if len(years) >= 2:
            first_year = min(years)
            last_year = max(years)

            if age_stats[first_year] and age_stats[last_year]:
                age_change = age_stats[last_year]["mean_age"] - age_stats[first_year]["mean_age"]

                if scenario == "baseline":
                    if abs(age_change) < 0.5:
                        status = "✅ PASS"
                    else:
                        status = "❌ FAIL"
                    print(f"• {scenario}: {age_change:+.2f} years change {status}")

                elif scenario == "high_early_attrition":
                    if age_change > 0.5:
                        status = "✅ PASS"
                    else:
                        status = "❌ FAIL"
                    print(f"• {scenario}: {age_change:+.2f} years change {status}")

                elif scenario == "high_late_attrition":
                    if age_change < -0.5:
                        status = "✅ PASS"
                    else:
                        status = "❌ FAIL"
                    print(f"• {scenario}: {age_change:+.2f} years change {status}")


def main():
    """Main function to run age sensitivity tests."""
    print("🧪 Starting Age Sensitivity Verification Tests")
    print("This will run three simulation scenarios to verify age-based termination logic")

    # Test configurations
    test_configs = [
        ("hazard_baseline_test.yaml", "baseline", "Baseline (Neutral Age Multipliers)"),
        ("hazard_high_early_attrition_test.yaml", "high_early_attrition", "High Early Attrition"),
        ("hazard_high_late_attrition_test.yaml", "high_late_attrition", "High Late Attrition"),
    ]

    results = []

    # Run each test scenario
    for hazard_config, output_suffix, description in test_configs:
        print(f"\n🚀 Running test: {description}")

        output_dir = run_simulation_with_hazard_config(hazard_config, output_suffix)

        if output_dir:
            analysis = analyze_age_distribution(output_dir, output_suffix)
            results.append(analysis)
        else:
            print(f"❌ Failed to run {description}")
            results.append(None)

    # Generate comparison report
    create_comparison_report(results)

    print(f"\n{'='*80}")
    print("🎯 Age Sensitivity Verification Complete!")
    print("Check the detailed analysis above to verify that age multipliers")
    print("are working as expected in the termination engine.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
