#!/usr/bin/env python3
"""
Simple test of the enhanced hiring algorithm logic.
"""

def manage_headcount_to_exact_target(
    soy_actives: int,
    target_growth_rate: float,
    num_markov_exits_existing: int,
    new_hire_termination_rate: float,
    expected_additional_experienced_exits: int = 0
) -> tuple[int, int]:
    """Enhanced version of the function with additional exits parameter."""
    
    # Validate inputs
    if new_hire_termination_rate >= 1.0:
        raise ValueError(f"new_hire_termination_rate must be < 1.0, got {new_hire_termination_rate}")

    # Step 1: Calculate target EOY actives
    target_eoy_actives = round(soy_actives * (1 + target_growth_rate))

    # Step 2: Account for ALL expected experienced exits (key fix)
    total_expected_experienced_exits = num_markov_exits_existing + expected_additional_experienced_exits
    final_expected_survivors = soy_actives - total_expected_experienced_exits

    # Step 3: Calculate net actives needed from new hires
    net_actives_needed_from_hiring_pool = target_eoy_actives - final_expected_survivors

    # Step 4: Determine gross hires and/or forced terminations
    if net_actives_needed_from_hiring_pool >= 0:
        if net_actives_needed_from_hiring_pool == 0:
            calculated_gross_new_hires = 0
        else:
            calculated_gross_new_hires = round(
                net_actives_needed_from_hiring_pool / (1 - new_hire_termination_rate)
            )
        calculated_forced_terminations_from_existing = 0
    else:
        calculated_gross_new_hires = 0
        calculated_forced_terminations_from_existing = abs(net_actives_needed_from_hiring_pool)

    return calculated_gross_new_hires, calculated_forced_terminations_from_existing


def test_enhanced_logic():
    """Test the enhanced hiring algorithm logic."""
    print("🧪 Testing Enhanced Hiring Algorithm Logic")
    print("=" * 50)

    # Test scenario based on our actual data
    soy_actives = 126  # From our analysis
    target_growth = 0.03  # 3% growth target
    experienced_exits_occurred = 20  # Approximate from our data
    nh_term_rate = 0.25  # 25% new hire termination
    
    print(f"Scenario:")
    print(f"  Start of Year: {soy_actives} employees")
    print(f"  Target Growth: {target_growth:.1%} (target = {round(soy_actives * (1 + target_growth))})")
    print(f"  Experienced exits occurred: {experienced_exits_occurred}")
    print(f"  NH termination rate: {nh_term_rate:.1%}")
    print()

    # Test OLD logic (no additional exits)
    print("OLD LOGIC (no additional exits):")
    gross_hires_old, forced_terms_old = manage_headcount_to_exact_target(
        soy_actives=soy_actives,
        target_growth_rate=target_growth,
        num_markov_exits_existing=experienced_exits_occurred,
        new_hire_termination_rate=nh_term_rate,
        expected_additional_experienced_exits=0
    )
    print(f"  Gross hires: {gross_hires_old}")
    print(f"  Forced terminations: {forced_terms_old}")

    # Calculate expected outcome
    survivors_old = soy_actives - experienced_exits_occurred
    net_hires_old = gross_hires_old * (1 - nh_term_rate)
    final_count_old = survivors_old + net_hires_old
    target = round(soy_actives * (1 + target_growth))
    print(f"  Expected final count: {final_count_old:.0f} (vs target {target})")
    print(f"  Gap from target: {final_count_old - target:+.0f}")
    print()

    # Test NEW logic (with additional exits)
    expected_additional_exits = 15  # Estimate based on our analysis
    print(f"NEW LOGIC (with {expected_additional_exits} additional exits):")
    gross_hires_new, forced_terms_new = manage_headcount_to_exact_target(
        soy_actives=soy_actives,
        target_growth_rate=target_growth,
        num_markov_exits_existing=experienced_exits_occurred,
        new_hire_termination_rate=nh_term_rate,
        expected_additional_experienced_exits=expected_additional_exits
    )
    print(f"  Gross hires: {gross_hires_new}")
    print(f"  Forced terminations: {forced_terms_new}")

    # Calculate expected outcome
    total_experienced_exits = experienced_exits_occurred + expected_additional_exits
    survivors_new = soy_actives - total_experienced_exits
    net_hires_new = gross_hires_new * (1 - nh_term_rate)
    final_count_new = survivors_new + net_hires_new
    print(f"  Expected final count: {final_count_new:.0f} (vs target {target})")
    print(f"  Gap from target: {final_count_new - target:+.0f}")
    print()

    print("📊 COMPARISON:")
    print(f"  Old logic: {gross_hires_old} hires → {final_count_old:.0f} final (gap: {final_count_old - target:+.0f})")
    print(f"  New logic: {gross_hires_new} hires → {final_count_new:.0f} final (gap: {final_count_new - target:+.0f})")
    print(f"  Difference: {gross_hires_new - gross_hires_old:+d} more hires with new logic")
    print()

    old_gap = abs(final_count_old - target)
    new_gap = abs(final_count_new - target)
    improvement = old_gap - new_gap
    
    print(f"✅ TARGET ACCURACY:")
    print(f"  Old logic gap: {old_gap:.0f} employees")
    print(f"  New logic gap: {new_gap:.0f} employees")
    print(f"  Improvement: {improvement:+.0f} employees closer to target")
    
    if improvement > 0:
        print(f"\n🎉 SUCCESS! Enhanced logic is {improvement:.0f} employees more accurate!")
    else:
        print(f"\n⚠️  Enhanced logic needs adjustment.")


if __name__ == "__main__":
    test_enhanced_logic()
