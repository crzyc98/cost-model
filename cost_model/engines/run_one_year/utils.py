"""
Utility functions for run_one_year package.

Contains debugging, logging, and other helper functions.
"""
import logging
from typing import Dict, Optional, Any, List, Union
import pandas as pd


def dbg(year: int, label: str, df: pd.DataFrame) -> None:
    """
    Debug helper for logging DataFrame stats during simulation.

    Args:
        year: Current simulation year
        label: Label for the debug message
        df: DataFrame to analyze
    """
    active_ct = 0
    if "active" in df.columns:
        active_ct = df["active"].sum()

    unique_ids = len(df["employee_id"].unique()) if "employee_id" in df.columns else 0

    logging.debug(
        f"[DBG YR={year}] {label:25s} rows={df.shape[0]:5d} "
        f"uniq_ids={unique_ids:5d} act={active_ct}"
    )


import math

def compute_headcount_targets(start_count: int, survivor_count: int, target_growth: float, nh_term_rate: float):
    """
    Compute headcount targets for the year given growth and new-hire termination rates.

    Args:
        start_count: Number of employees at start of year
        survivor_count: Employees remaining after attrition
        target_growth: Target growth rate (e.g., 0.05 for 5%)
        nh_term_rate: New hire termination rate (e.g., 0.2 for 20%)
    Returns:
        (target_eoy, net_needed, gross_needed)
    """
    target_eoy = int(round(start_count * (1 + target_growth)))
    net_needed = max(target_eoy - survivor_count, 0)
    gross_needed = math.ceil(net_needed / max(1 - nh_term_rate, 1e-9)) if net_needed > 0 else 0
    return target_eoy, net_needed, gross_needed


def manage_headcount_to_exact_target(
    soy_actives: int,
    target_growth_rate: float,
    num_markov_exits_existing: int,
    new_hire_termination_rate: float
) -> tuple[int, int]:
    """
    Calculate exact headcount targeting for workforce simulation.

    This function ensures that the number of active employees at the end of each
    simulation year meets an exact target growth rate by calculating either:
    1. The number of gross new hires needed (accounting for new hire attrition), or
    2. The number of forced terminations needed from existing survivors

    Args:
        soy_actives: Number of active employees at start of year
        target_growth_rate: Target growth rate (e.g., 0.03 for 3%)
        num_markov_exits_existing: Number of existing employees who terminated via Markov exits
        new_hire_termination_rate: Proportion of new hires expected to terminate within first year

    Returns:
        Tuple of (calculated_gross_new_hires, calculated_forced_terminations_from_existing)
        - calculated_gross_new_hires: Number of new employees to hire
        - calculated_forced_terminations_from_existing: Number of additional terminations needed

    Raises:
        ValueError: If new_hire_termination_rate is 1.0 or greater (would cause division by zero)
    """
    # Validate inputs
    if new_hire_termination_rate >= 1.0:
        raise ValueError(f"new_hire_termination_rate must be < 1.0, got {new_hire_termination_rate}")

    # Step 1: Calculate target EOY actives
    target_eoy_actives = round(soy_actives * (1 + target_growth_rate))

    # Step 2: Determine survivors from initial Markov exits
    survived_soy_actives = soy_actives - num_markov_exits_existing

    # Step 3: Calculate net actives needed from new hires (or excess survivors)
    net_actives_needed_from_hiring_pool = target_eoy_actives - survived_soy_actives

    # Step 4: Determine gross hires and/or forced terminations
    if net_actives_needed_from_hiring_pool >= 0:
        # Need to add employees (or current survivors exactly match target)
        if net_actives_needed_from_hiring_pool == 0:
            calculated_gross_new_hires = 0
        else:
            # Calculate gross hires accounting for new hire attrition
            # Use round() to match the actual termination logic which uses round()
            calculated_gross_new_hires = round(
                net_actives_needed_from_hiring_pool / (1 - new_hire_termination_rate)
            )
        calculated_forced_terminations_from_existing = 0
    else:
        # Current survivors exceed target, need to reduce via forced terminations
        calculated_gross_new_hires = 0
        calculated_forced_terminations_from_existing = abs(net_actives_needed_from_hiring_pool)

    return calculated_gross_new_hires, calculated_forced_terminations_from_existing


def test_compute_headcount_targets():
    # Test growing from 100 to 103 employees with 15% new hire termination rate
    start = 100
    target_eoy = 103  # 3% growth
    target_growth = 0.03
    nh_term_rate = 0.15  # 15% new hire termination rate

    # Simulate some attrition - let's say we lose 10 people to attrition
    survivors = 90  # 100 - 10

    # Calculate hiring needs
    target_eoy, net_needed, gross_needed = compute_headcount_targets(
        start, survivors, target_growth, nh_term_rate
    )

    # Verify calculations
    assert target_eoy == 103, f"Target EOY should be 103, got {target_eoy}"
    assert net_needed == 13, f"Net needed should be 13 (103-90), got {net_needed}"
    # Expected gross: 13 / (1 - 0.15) = 15.29 → 16
    assert gross_needed == 16, f"Expected gross_needed=16, got {gross_needed}"
    print("test_compute_headcount_targets passed!")


def test_manage_headcount_to_exact_target():
    """
    Test the exact headcount targeting function with various scenarios.
    """
    print("Testing manage_headcount_to_exact_target function...")

    # Test Case 1: Normal growth scenario - need to hire
    print("\n--- Test Case 1: Normal growth scenario ---")
    soy_actives = 100
    target_growth_rate = 0.03  # 3% growth
    num_markov_exits = 10  # 10 people left via natural attrition
    new_hire_term_rate = 0.25  # 25% of new hires terminate

    gross_hires, forced_terms = manage_headcount_to_exact_target(
        soy_actives, target_growth_rate, num_markov_exits, new_hire_term_rate
    )

    # Expected: target = 103, survivors = 90, need 13 net, gross = round(13/0.75) = round(17.33) = 17
    expected_target = round(100 * 1.03)  # 103
    expected_survivors = 100 - 10  # 90
    expected_net_needed = 103 - 90  # 13
    expected_gross = round(13 / 0.75)  # 17

    print(f"SOY: {soy_actives}, Target: {expected_target}, Survivors: {expected_survivors}")
    print(f"Net needed: {expected_net_needed}, Gross hires: {gross_hires}, Forced terms: {forced_terms}")

    assert gross_hires == expected_gross, f"Expected {expected_gross} gross hires, got {gross_hires}"
    assert forced_terms == 0, f"Expected 0 forced terminations, got {forced_terms}"

    # Test Case 2: Excess survivors - need forced terminations
    print("\n--- Test Case 2: Excess survivors scenario ---")
    soy_actives = 100
    target_growth_rate = 0.03  # 3% growth (target = 103)
    num_markov_exits = -2  # Negative exits means we gained people somehow, survivors = 102
    # Actually, let's use a more realistic scenario: low attrition
    num_markov_exits = 1  # Only 1 person left, survivors = 99
    # But we want survivors > target, so let's use negative growth instead
    target_growth_rate = -0.05  # -5% shrinkage (target = 95)
    num_markov_exits = 3  # 3 people left, survivors = 97 > target = 95
    new_hire_term_rate = 0.25

    gross_hires, forced_terms = manage_headcount_to_exact_target(
        soy_actives, target_growth_rate, num_markov_exits, new_hire_term_rate
    )

    # Expected: target = 95, survivors = 97, excess = 2, need 2 forced terminations
    expected_target = round(100 * 0.95)  # 95
    expected_survivors = 100 - 3  # 97
    expected_excess = expected_survivors - expected_target  # 2 (positive means excess)
    expected_forced_terms = 2  # Need to terminate 2 excess

    print(f"SOY: {soy_actives}, Target: {expected_target}, Survivors: {expected_survivors}")
    print(f"Excess survivors: {expected_excess}, Gross hires: {gross_hires}, Forced terms: {forced_terms}")

    assert gross_hires == 0, f"Expected 0 gross hires, got {gross_hires}"
    assert forced_terms == expected_forced_terms, f"Expected {expected_forced_terms} forced terms, got {forced_terms}"

    # Test Case 3: Perfect match - no action needed
    print("\n--- Test Case 3: Perfect match scenario ---")
    soy_actives = 100
    target_growth_rate = 0.03  # target = 103
    num_markov_exits = 3  # survivors = 97, but we'll adjust to make it exact
    # Let's make survivors exactly match target
    num_markov_exits = 100 - 103  # This would be negative, so let's use a different approach

    # Adjust: if we want exactly 103 survivors, and target is 103, then markov exits = 100 - 103 = -3
    # That doesn't make sense. Let's use: target = 97 (negative growth)
    target_growth_rate = -0.03  # -3% shrinkage, target = 97
    num_markov_exits = 3  # survivors = 97, exactly matches target

    gross_hires, forced_terms = manage_headcount_to_exact_target(
        soy_actives, target_growth_rate, num_markov_exits, new_hire_term_rate
    )

    expected_target = round(100 * 0.97)  # 97
    expected_survivors = 97

    print(f"SOY: {soy_actives}, Target: {expected_target}, Survivors: {expected_survivors}")
    print(f"Perfect match - Gross hires: {gross_hires}, Forced terms: {forced_terms}")

    assert gross_hires == 0, f"Expected 0 gross hires, got {gross_hires}"
    assert forced_terms == 0, f"Expected 0 forced terms, got {forced_terms}"

    # Test Case 4: Edge case - zero new hire termination rate
    print("\n--- Test Case 4: Zero new hire termination rate ---")
    soy_actives = 100
    target_growth_rate = 0.05  # 5% growth, target = 105
    num_markov_exits = 10  # survivors = 90
    new_hire_term_rate = 0.0  # No new hire attrition

    gross_hires, forced_terms = manage_headcount_to_exact_target(
        soy_actives, target_growth_rate, num_markov_exits, new_hire_term_rate
    )

    # Expected: need 15 net, with 0% attrition, gross = net = 15
    expected_net = 105 - 90  # 15
    expected_gross = 15  # No attrition adjustment needed

    print(f"Zero attrition - Net needed: {expected_net}, Gross hires: {gross_hires}")

    assert gross_hires == expected_gross, f"Expected {expected_gross} gross hires, got {gross_hires}"
    assert forced_terms == 0, f"Expected 0 forced terms, got {forced_terms}"

    # Test Case 5: Error case - invalid termination rate
    print("\n--- Test Case 5: Invalid termination rate ---")
    try:
        gross_hires, forced_terms = manage_headcount_to_exact_target(
            100, 0.03, 10, 1.0  # 100% termination rate should raise error
        )
        assert False, "Should have raised ValueError for termination rate >= 1.0"
    except ValueError as e:
        print(f"Correctly caught error: {e}")

    print("\n✅ All test cases passed!")


def diagnose_headcount_discrepancy(
    start_count: int,
    end_count: int,
    new_hires: int,
    new_hire_terms: int,
    experienced_terms: int,
    target_growth_rate: float,
    configured_nh_term_rate: float
):
    """
    Diagnose discrepancies between expected and actual headcount changes.
    """
    print("=== Headcount Discrepancy Diagnosis ===")
    print(f"Start count: {start_count}")
    print(f"End count: {end_count}")
    print(f"Actual growth: {(end_count - start_count) / start_count:.1%}")
    print(f"Target growth: {target_growth_rate:.1%}")
    print()

    # Calculate what the math should be
    expected_target = round(start_count * (1 + target_growth_rate))
    survivors_after_exp_terms = start_count - experienced_terms
    net_needed = expected_target - survivors_after_exp_terms

    print(f"Expected target EOY: {expected_target}")
    print(f"Survivors after experienced terms: {survivors_after_exp_terms}")
    print(f"Net hires needed: {net_needed}")
    print()

    # Analyze actual vs expected
    actual_net_hires = new_hires - new_hire_terms
    actual_nh_term_rate = new_hire_terms / new_hires if new_hires > 0 else 0

    print(f"Actual new hires: {new_hires}")
    print(f"Actual new hire terminations: {new_hire_terms}")
    print(f"Actual net new hires: {actual_net_hires}")
    print(f"Actual NH termination rate: {actual_nh_term_rate:.1%}")
    print(f"Configured NH termination rate: {configured_nh_term_rate:.1%}")
    print()

    # Check the math
    calculated_end = survivors_after_exp_terms + actual_net_hires
    discrepancy = end_count - calculated_end

    print(f"Calculated end count: {calculated_end}")
    print(f"Actual end count: {end_count}")
    print(f"Discrepancy: {discrepancy}")
    print()

    # What should the gross hires be for the target?
    expected_gross_for_target = round(net_needed / (1 - configured_nh_term_rate))
    expected_terms_for_target = round(expected_gross_for_target * configured_nh_term_rate)

    print(f"Expected gross hires for target: {expected_gross_for_target}")
    print(f"Expected terminations for target: {expected_terms_for_target}")
    print(f"Actual gross hires: {new_hires} (difference: {new_hires - expected_gross_for_target:+d})")
    print(f"Actual terminations: {new_hire_terms} (difference: {new_hire_terms - expected_terms_for_target:+d})")

    if discrepancy != 0:
        print(f"\n⚠️  There's a {discrepancy}-employee discrepancy that suggests:")
        if discrepancy > 0:
            print("   - Additional employees were added from another source")
            print("   - Or the termination logic isn't working as expected")
        else:
            print("   - More employees were removed than accounted for")
            print("   - Or there's an error in the calculation")


if __name__ == "__main__":
    test_compute_headcount_targets()
    test_manage_headcount_to_exact_target()

    # Diagnose the actual vs expected discrepancy
    print("\n" + "="*60)
    diagnose_headcount_discrepancy(
        start_count=100,
        end_count=124,
        new_hires=38,
        new_hire_terms=5,
        experienced_terms=18,
        target_growth_rate=0.03,
        configured_nh_term_rate=0.25
    )
