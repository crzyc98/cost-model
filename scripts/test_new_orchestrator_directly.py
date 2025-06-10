#!/usr/bin/env python3
"""
Test script to run the new orchestrator directly and verify exact targeting.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from types import SimpleNamespace

import numpy as np
import pandas as pd


def create_test_data():
    """Create test data matching your scenario."""
    # Create initial snapshot with 100 employees
    employee_ids = [f"EMP_{i:04d}" for i in range(100)]

    snapshot_data = {
        "employee_id": employee_ids,
        "active": [True] * 100,
        "employee_hire_date": pd.to_datetime("2024-01-01"),
        "employee_gross_compensation": [65000.0] * 100,
        "employee_termination_date": pd.NaT,
        "employee_birth_date": pd.to_datetime("1990-01-01"),
        "employee_deferral_rate": 0.05,
        "employee_tenure_band": "1-3",
        "employee_tenure": 1.5,
        "employee_level": 1,
        "employee_level_source": "existing",
        "employee_exited": False,
        "simulation_year": 2024,
    }

    snapshot = pd.DataFrame(snapshot_data)
    snapshot = snapshot.set_index("employee_id", drop=False)

    # Create hazard table with correct column names
    hazard_data = {
        "simulation_year": [2025],
        "employee_level": [1],
        "employee_tenure_band": ["1-3"],  # Use correct column name
        "term_rate": [0.15],
        "new_hire_termination_rate": [0.25],
        "cola_pct": [0.02],
        "merit_raise_pct": [0.03],
        "promotion_raise_pct": [0.05],  # Add promotion column
    }
    hazard_table = pd.DataFrame(hazard_data)

    # Create global params
    attrition = SimpleNamespace()
    attrition.new_hire_termination_rate = 0.25

    global_params = SimpleNamespace()
    global_params.target_growth = 0.03
    global_params.attrition = attrition
    global_params.deterministic_termination = True
    global_params.dev_mode = True  # Enable dev mode for default promotion matrix

    # Create plan rules
    plan_rules = {}

    # Create RNG
    rng = np.random.default_rng(42)

    # Create empty event log
    event_log = pd.DataFrame(
        columns=[
            "event_id",
            "event_time",
            "employee_id",
            "event_type",
            "value_num",
            "value_json",
            "meta",
            "simulation_year",
        ]
    )

    return snapshot, hazard_table, global_params, plan_rules, rng, event_log


def test_new_orchestrator():
    """Test the new orchestrator directly."""
    print("=== Testing New Orchestrator Directly ===\n")

    # Create test data
    snapshot, hazard_table, global_params, plan_rules, rng, event_log = create_test_data()

    print(f"Initial setup:")
    print(f"- Initial employees: {len(snapshot)}")
    print(f"- Target growth: {global_params.target_growth:.1%}")
    print(f"- New hire termination rate: {global_params.attrition.new_hire_termination_rate:.1%}")
    print()

    try:
        # Import and run the new orchestrator
        from cost_model.engines.run_one_year.orchestrator import run_one_year

        print("Running new orchestrator...")

        # Run the simulation for 2025
        new_events, final_snapshot = run_one_year(
            event_log=event_log,
            prev_snapshot=snapshot,
            year=2025,
            global_params=global_params,
            plan_rules=plan_rules,
            hazard_table=hazard_table,
            rng=rng,
            census_template_path=None,
            rng_seed_offset=0,
            deterministic_term=True,
        )

        print("✅ Orchestrator ran successfully!")
        print()

        # Analyze results
        initial_count = len(snapshot)
        final_count = len(final_snapshot)
        active_final = (
            final_snapshot["active"].sum()
            if "active" in final_snapshot.columns
            else len(final_snapshot)
        )

        print(f"Results:")
        print(f"- Initial employees: {initial_count}")
        print(f"- Final employees: {final_count}")
        print(f"- Active employees: {active_final}")
        print(f"- Growth: {(active_final - initial_count) / initial_count:.1%}")
        print()

        # Analyze events
        if not new_events.empty:
            event_types = new_events["event_type"].value_counts()
            print(f"Events generated:")
            for event_type, count in event_types.items():
                print(f"- {event_type}: {count}")

            # Count hires and terminations
            hire_events = new_events[new_events["event_type"] == "EVT_HIRE"]
            term_events = new_events[new_events["event_type"] == "EVT_TERM"]

            num_hires = len(hire_events)
            num_terms = len(term_events)

            print()
            print(f"Key metrics:")
            print(f"- New hires: {num_hires}")
            print(f"- Terminations: {num_terms}")

            # Check if this matches expected exact targeting
            expected_target = round(initial_count * (1 + global_params.target_growth))
            print(f"- Expected target EOY: {expected_target}")
            print(f"- Actual final count: {active_final}")

            if active_final == expected_target:
                print("✅ Exact targeting is working!")
            else:
                print(f"❌ Exact targeting failed. Difference: {active_final - expected_target}")

            # Check new hire termination rate
            if num_hires > 0:
                # Count new hire terminations (terminations of employees hired this year)
                hire_ids = set(hire_events["employee_id"])
                nh_term_events = term_events[term_events["employee_id"].isin(hire_ids)]
                num_nh_terms = len(nh_term_events)

                actual_nh_rate = num_nh_terms / num_hires if num_hires > 0 else 0
                expected_nh_rate = global_params.attrition.new_hire_termination_rate

                print(f"- New hire terminations: {num_nh_terms}")
                print(f"- Actual NH termination rate: {actual_nh_rate:.1%}")
                print(f"- Expected NH termination rate: {expected_nh_rate:.1%}")

                if abs(actual_nh_rate - expected_nh_rate) < 0.05:  # Within 5%
                    print("✅ New hire termination rate is correct!")
                else:
                    print("❌ New hire termination rate is incorrect!")
        else:
            print("❌ No events generated!")

        # Compare with your actual results
        print()
        print("=== Comparison with Your Results ===")
        print("Your actual results: 100 → 124 (38 hires, 5 NH terms)")
        print(
            f"New orchestrator: {initial_count} → {active_final} ({num_hires} hires, {num_nh_terms if 'num_nh_terms' in locals() else 'N/A'} NH terms)"
        )

        if num_hires != 38:
            print("✅ New orchestrator is using different logic (exact targeting)")
        else:
            print("❌ New orchestrator matches old results (not using exact targeting)")

    except Exception as e:
        print(f"❌ Error running orchestrator: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_new_orchestrator()
