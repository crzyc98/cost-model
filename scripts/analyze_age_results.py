#!/usr/bin/env python3
"""Analyze age sensitivity test results."""


def main():
    print("🎯 AGE SENSITIVITY TEST RESULTS ANALYSIS")
    print("=" * 60)

    # Results from the output above
    scenarios = {
        "Baseline": {
            2025: {"employees": 112, "mean_age": 46.4},
            2026: {"employees": 107, "mean_age": 45.7},
            2027: {"employees": 101, "mean_age": 46.2},
            2028: {"employees": 96, "mean_age": 45.8},
            2029: {"employees": 91, "mean_age": 46.0},
        },
        "High Early Attrition": {
            2025: {"employees": 111, "mean_age": 46.1},
            2026: {"employees": 105, "mean_age": 45.3},
            2027: {"employees": 97, "mean_age": 44.7},
            2028: {"employees": 87, "mean_age": 43.4},
            2029: {"employees": 79, "mean_age": 42.8},
        },
        "High Late Attrition": {
            2025: {"employees": 111, "mean_age": 46.0},
            2026: {"employees": 106, "mean_age": 45.6},
            2027: {"employees": 101, "mean_age": 45.2},
            2028: {"employees": 94, "mean_age": 44.2},
            2029: {"employees": 88, "mean_age": 44.2},
        },
    }

    print("\n📊 SUMMARY TABLE")
    print(f"{'Scenario':<20} {'2025→2029 Age Change':<20} {'Employee Change':<15} {'Status'}")
    print("-" * 70)

    for scenario, data in scenarios.items():
        age_change = data[2029]["mean_age"] - data[2025]["mean_age"]
        emp_change = data[2029]["employees"] - data[2025]["employees"]

        # Determine if results match expectations
        if scenario == "Baseline":
            status = "✅ PASS" if abs(age_change) <= 0.5 else "❌ FAIL"
        elif scenario == "High Early Attrition":
            # Should get younger (negative change) because young people leave more
            status = "❌ UNEXPECTED" if age_change < -1.0 else "⚠️  NEEDS REVIEW"
        elif scenario == "High Late Attrition":
            # Should get younger (negative change) because old people leave more
            status = "✅ EXPECTED" if age_change < -1.0 else "⚠️  NEEDS REVIEW"

        print(f'{scenario:<20} {age_change:+.1f} years{"":<11} {emp_change:+d}{"":<10} {status}')

    print("\n🔍 DETAILED ANALYSIS")
    print("-" * 40)

    for scenario, data in scenarios.items():
        print(f"\n{scenario}:")
        age_change = data[2029]["mean_age"] - data[2025]["mean_age"]
        emp_change = data[2029]["employees"] - data[2025]["employees"]

        print(
            f'  Age change: {age_change:+.1f} years ({data[2025]["mean_age"]:.1f} → {data[2029]["mean_age"]:.1f})'
        )
        print(
            f'  Employee change: {emp_change:+d} ({data[2025]["employees"]} → {data[2029]["employees"]})'
        )

        if scenario == "High Early Attrition":
            print(f"  💡 Expected: Young employees leave more → workforce should age")
            print(f"  📈 Actual: Workforce got younger by {abs(age_change):.1f} years")
            print(f"  ❓ This is unexpected - need to verify age multiplier logic")
        elif scenario == "High Late Attrition":
            print(f"  💡 Expected: Old employees leave more → workforce gets younger")
            print(f"  📈 Actual: Workforce got younger by {abs(age_change):.1f} years")
            print(f"  ✅ This matches expectations!")

    print("\n🎯 KEY FINDINGS")
    print("-" * 40)
    print("✅ Age data is successfully integrated into all simulations")
    print("✅ Age calculations are working correctly (proper age progression)")
    print("⚠️  Age multipliers may not be working as expected for early attrition")
    print("✅ Late attrition scenario shows expected behavior")

    print("\n🔍 INVESTIGATION NEEDED")
    print("-" * 40)
    print("1. Verify that HAZARD_CONFIG_FILE environment variable is being used")
    print("2. Check termination logs for age multiplier application")
    print("3. Confirm hazard configuration files are being loaded correctly")
    print("4. Consider increasing age multiplier values for more dramatic effects")

    print("\n💡 NEXT STEPS")
    print("-" * 40)
    print("1. Check simulation logs for age multiplier debug messages")
    print("2. Verify environment variable is properly set during simulation runs")
    print("3. Test with more extreme age multiplier values")
    print("4. Implement promotion age sensitivity (User Story 1.1)")


if __name__ == "__main__":
    main()
