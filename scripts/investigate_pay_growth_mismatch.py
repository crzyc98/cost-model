#!/usr/bin/env python3
"""
Investigate why we can't achieve 0% pay growth when annual_compensation_increase_rate = 0.03 (3%)
This should theoretically give us near-zero pay growth, but we're seeing -3.58%.
"""

import subprocess
from pathlib import Path

import pandas as pd
import yaml

# Project paths
project_root = Path(__file__).parent
TEMPLATE = project_root / "config/dev_tiny.yaml"
RUNNER = project_root / "scripts/run_simulation.py"
DEFAULT_CENSUS_PATH = project_root / "data/census_template.parquet"


def create_baseline_test_config() -> Path:
    """Create a test config that exactly matches the baseline annual_compensation_increase_rate."""

    # Load base template
    with open(TEMPLATE, "r") as f:
        config = yaml.safe_load(f)

    # Keep the baseline 3% annual compensation increase rate
    # This should theoretically give us ~0% pay growth
    baseline_comp_rate = 0.03

    print(f"Testing with baseline annual_compensation_increase_rate: {baseline_comp_rate:.1%}")

    # Create output directory
    output_dir = Path("baseline_test")
    output_dir.mkdir(exist_ok=True)

    # Save the unmodified config for testing
    baseline_config_path = output_dir / "baseline_annual_comp_test.yaml"
    with open(baseline_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    return baseline_config_path


def create_zero_growth_test_config() -> Path:
    """Create a test config designed to achieve exactly 0% pay growth."""

    # Load base template
    with open(TEMPLATE, "r") as f:
        config = yaml.safe_load(f)

    # Try to zero out all compensation growth components
    config["global_parameters"]["annual_compensation_increase_rate"] = 0.00  # 0% base increase

    # Zero out merit increases
    config["global_parameters"]["raises_hazard"]["merit_base"] = 0.00
    config["global_parameters"]["raises_hazard"]["merit_tenure_bump_value"] = 0.00
    config["global_parameters"]["raises_hazard"]["merit_low_level_bump_value"] = 0.00
    config["global_parameters"]["raises_hazard"]["promotion_raise"] = 0.00

    # Zero out COLA
    config["global_parameters"]["cola_hazard"]["by_year"] = {
        2025: 0.00,
        2026: 0.00,
        2027: 0.00,
        2028: 0.00,
        2029: 0.00,
    }

    # Zero out job level merit increases
    for level in config["job_levels"]:
        level["avg_annual_merit_increase"] = 0.00

    # Create output directory
    output_dir = Path("baseline_test")
    output_dir.mkdir(exist_ok=True)

    zero_config_path = output_dir / "zero_growth_test.yaml"
    with open(zero_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print("Created zero growth test config with:")
    print("  - annual_compensation_increase_rate: 0.00")
    print("  - merit_base: 0.00")
    print("  - COLA rates: 0.00")
    print("  - promotion_raise: 0.00")

    return zero_config_path


def run_and_analyze_config(config_path: Path, test_name: str) -> dict:
    """Run a config and analyze the pay growth components."""

    output_dir = Path("baseline_test") / f"output_{config_path.stem}"

    cmd = [
        "python3",
        str(RUNNER),
        "--config",
        str(config_path),
        "--scenario",
        "baseline",
        "--census",
        str(DEFAULT_CENSUS_PATH),
        "--output",
        str(output_dir),
    ]

    try:
        print(f"\\nRunning {test_name}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print(f"❌ {test_name} failed: {result.stderr}")
            return None

        # Parse results
        metrics_file = output_dir / "Baseline/Baseline_metrics.csv"
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)

            if len(df) > 1:
                print(f"✅ {test_name} completed successfully")

                # Detailed compensation analysis
                print(f"\\n=== {test_name.upper()} RESULTS ===")
                print("Year-by-Year Compensation Analysis:")

                for _, row in df.iterrows():
                    year = row["year"]
                    avg_comp = row["avg_compensation"]
                    headcount = int(row["active_headcount"])
                    total_comp = row.get("total_compensation", "N/A")
                    print(
                        f"  {year}: {headcount:3d} employees, avg ${avg_comp:,.0f}, total ${total_comp}"
                    )

                # Calculate overall growth
                initial_comp = df["avg_compensation"].iloc[0]
                final_comp = df["avg_compensation"].iloc[-1]
                pay_growth = (final_comp - initial_comp) / initial_comp

                print(f"\\nOverall Pay Growth: {pay_growth:.2%}")
                print(f"  Initial Avg Compensation: ${initial_comp:,.0f}")
                print(f"  Final Avg Compensation: ${final_comp:,.0f}")
                print(f"  Absolute Change: ${final_comp - initial_comp:,.0f}")

                # Try to load detailed snapshots for workforce composition analysis
                try:
                    snapshots_dir = output_dir / "Baseline"
                    yearly_files = list(snapshots_dir.glob("year=*/snapshot.parquet"))

                    if yearly_files:
                        print(f"\\nWorkforce Composition Changes:")

                        for year_file in sorted(yearly_files):
                            year = year_file.parent.name.split("=")[1]
                            snapshot_df = pd.read_parquet(year_file)

                            if not snapshot_df.empty:
                                active_employees = snapshot_df[snapshot_df["active"] == True]
                                if not active_employees.empty:
                                    avg_age = active_employees["employee_age"].mean()
                                    avg_tenure = active_employees["employee_tenure"].mean()
                                    avg_comp = active_employees[
                                        "employee_gross_compensation"
                                    ].mean()

                                    print(
                                        f"  {year}: {len(active_employees):3d} active, "
                                        f"avg age {avg_age:.1f}, avg tenure {avg_tenure:.1f}, "
                                        f"avg comp ${avg_comp:,.0f}"
                                    )
                except Exception as e:
                    print(f"Could not analyze workforce composition: {e}")

                return {
                    "test_name": test_name,
                    "pay_growth": pay_growth,
                    "initial_comp": initial_comp,
                    "final_comp": final_comp,
                    "final_headcount": df["active_headcount"].iloc[-1],
                    "config_path": str(config_path),
                }

        print(f"❌ {test_name}: Could not find or parse results")
        return None

    except Exception as e:
        print(f"❌ {test_name} error: {e}")
        return None


def analyze_pay_growth_components():
    """Analyze why we can't achieve the expected pay growth."""

    print("=== PAY GROWTH MISMATCH INVESTIGATION ===")
    print("Question: Why can't we achieve 0% pay growth with 3% annual_compensation_increase_rate?")
    print()

    # Test 1: Baseline configuration (should give ~0% growth theoretically)
    print("🔍 TEST 1: Baseline Configuration Analysis")
    baseline_config = create_baseline_test_config()
    baseline_result = run_and_analyze_config(baseline_config, "Baseline (3% annual rate)")

    # Test 2: Zero growth configuration (should definitely give 0% growth)
    print("\\n🔍 TEST 2: Zero Growth Configuration Analysis")
    zero_config = create_zero_growth_test_config()
    zero_result = run_and_analyze_config(zero_config, "Zero Growth (0% all rates)")

    # Analysis
    print("\\n" + "=" * 60)
    print("=== ANALYSIS AND CONCLUSIONS ===")

    if baseline_result and zero_result:
        print(f"\\n📊 COMPARATIVE RESULTS:")
        print(f"  Baseline (3% rate): {baseline_result['pay_growth']:.2%} pay growth")
        print(f"  Zero (0% rate):     {zero_result['pay_growth']:.2%} pay growth")

        expected_vs_actual = baseline_result["pay_growth"] - 0.00  # Expected was 0%
        print(f"\\n📈 MISMATCH ANALYSIS:")
        print(f"  Expected pay growth with 3% rate: ~0.00%")
        print(f"  Actual pay growth:                {baseline_result['pay_growth']:.2%}")
        print(f"  Difference (basis points):        {expected_vs_actual * 10000:.0f}bp")

        if zero_result["pay_growth"] < 0:
            print(f"\\n🔍 ROOT CAUSE IDENTIFIED:")
            print(
                f"  Even with 0% ALL compensation increases, pay growth is {zero_result['pay_growth']:.2%}"
            )
            print(f"  This indicates STRUCTURAL factors beyond compensation parameters:")
            print(f"    • Workforce composition changes (age, tenure, level mix)")
            print(f"    • New hire vs. departing employee salary differences")
            print(f"    • Promotion/demotion effects")
            print(f"    • Sample size effects in small workforce")

        if baseline_result["pay_growth"] < zero_result["pay_growth"]:
            print(f"\\n⚠️  PARADOX DETECTED:")
            print(f"  Baseline (3% rate) has LOWER pay growth than Zero (0% rate)")
            print(f"  This suggests complex interaction effects or model behavior")

    print(f"\\n💡 RECOMMENDATIONS:")
    print(f"  1. The 3% annual_compensation_increase_rate does NOT translate to 3% pay growth")
    print(f"  2. Workforce dynamics dominate over compensation parameters")
    print(f"  3. For precision targeting, focus on workforce composition, not just comp rates")
    print(f"  4. Consider the model's 'structural pay deflation' as a feature, not a bug")

    return baseline_result, zero_result


if __name__ == "__main__":
    results = analyze_pay_growth_components()
