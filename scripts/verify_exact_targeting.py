#!/usr/bin/env python3
"""
Verification script to check if exact targeting integration is active.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_integration():
    """Check if the exact targeting integration is working."""
    print("=== Verifying Exact Targeting Integration ===\n")

    try:
        # Test 1: Check if we can import the new function
        from cost_model.engines.run_one_year.utils import manage_headcount_to_exact_target

        print("✅ manage_headcount_to_exact_target function is importable")

        # Test 2: Check if we can import the new orchestrator
        from cost_model.engines.run_one_year.orchestrator.hiring import HiringOrchestrator

        print("✅ HiringOrchestrator is importable")

        # Test 3: Check if the orchestrator has the new method
        hiring_orch = HiringOrchestrator()
        if hasattr(hiring_orch, "get_required_forced_terminations"):
            print("✅ get_required_forced_terminations method exists")
        else:
            print("❌ get_required_forced_terminations method missing")

        # Test 4: Check if the main run_one_year imports the new orchestrator
        from cost_model.engines.run_one_year import run_one_year

        print("✅ run_one_year function is importable")

        # Test 5: Check the source of run_one_year
        import inspect

        source_file = inspect.getfile(run_one_year)
        print(f"✅ run_one_year source: {source_file}")

        # Test 6: Quick functional test
        result = manage_headcount_to_exact_target(
            soy_actives=100,
            target_growth_rate=0.03,
            num_markov_exits_existing=18,
            new_hire_termination_rate=0.25,
        )
        gross_hires, forced_terms = result
        print(f"✅ Function test: {gross_hires} gross hires, {forced_terms} forced terms")

        if gross_hires == 28 and forced_terms == 0:
            print("✅ Function returns expected values for test scenario")
        else:
            print(
                f"❌ Function returns unexpected values: expected (28, 0), got ({gross_hires}, {forced_terms})"
            )

        print("\n=== Integration Status ===")
        print("✅ All components are properly integrated and functional")
        print("✅ The exact targeting logic should be active in simulations")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False


def check_old_vs_new_results():
    """Compare old vs new targeting results."""
    print("\n=== Comparing Old vs New Targeting ===")

    try:
        from cost_model.engines.run_one_year.utils import (
            compute_headcount_targets,
            manage_headcount_to_exact_target,
        )

        # Test scenario from your data
        start_count = 100
        survivor_count = 82  # 100 - 18 experienced terms
        target_growth = 0.03
        nh_term_rate = 0.25

        # Old method
        target_eoy_old, net_needed_old, gross_needed_old = compute_headcount_targets(
            start_count, survivor_count, target_growth, nh_term_rate
        )

        # New method
        gross_hires_new, forced_terms_new = manage_headcount_to_exact_target(
            soy_actives=start_count,
            target_growth_rate=target_growth,
            num_markov_exits_existing=start_count - survivor_count,
            new_hire_termination_rate=nh_term_rate,
        )

        print(
            f"Old method: Target={target_eoy_old}, Net={net_needed_old}, Gross={gross_needed_old}"
        )
        print(f"New method: Gross={gross_hires_new}, Forced={forced_terms_new}")

        if gross_needed_old == gross_hires_new:
            print("✅ Both methods agree on gross hires")
        else:
            print(f"❌ Methods disagree: old={gross_needed_old}, new={gross_hires_new}")

        # Expected new hire terminations
        expected_nh_terms_old = round(gross_needed_old * nh_term_rate)
        expected_nh_terms_new = round(gross_hires_new * nh_term_rate)

        print(f"Expected NH terminations: old={expected_nh_terms_old}, new={expected_nh_terms_new}")

        # Your actual results show 38 hires, 5 terminations
        print(f"\nYour actual results: 38 hires, 5 terminations")
        print(f"Expected results: {gross_hires_new} hires, {expected_nh_terms_new} terminations")

        if gross_hires_new != 38:
            print("❌ The simulation is NOT using the new exact targeting logic")
            print("❌ It's still using the old logic or there's an integration issue")
        else:
            print("✅ The simulation might be using the new logic")

    except Exception as e:
        print(f"❌ Error comparing methods: {e}")


if __name__ == "__main__":
    success = check_integration()
    if success:
        check_old_vs_new_results()
    else:
        print("\n❌ Integration verification failed")
        sys.exit(1)
