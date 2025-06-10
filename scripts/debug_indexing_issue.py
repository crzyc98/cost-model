#!/usr/bin/env python3
"""
Debug script to check indexing issues in snapshot_update.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cost_model.config.loaders import load_config_to_namespace
from cost_model.engines.run_one_year.orchestrator import run_one_year
from cost_model.projections.event_log import create_initial_event_log
from cost_model.projections.hazard import load_and_expand_hazard_table
from cost_model.projections.snapshot import create_initial_snapshot
from cost_model.state.schema import EMP_GROSS_COMP, EMP_ID


def debug_indexing_issue():
    """Debug indexing issues in compensation application."""

    print("=== DEBUGGING INDEXING ISSUES IN COMPENSATION APPLICATION ===")

    # Load config and data
    config = load_config_to_namespace(Path("config/dev_tiny.yaml"))
    global_params = config.global_parameters

    initial_snapshot = create_initial_snapshot(2025, "data/census_preprocessed.parquet")
    initial_event_log = create_initial_event_log(2025)
    hazard_table = load_and_expand_hazard_table("data/hazard_table.parquet")

    # Use same random seed
    seed = getattr(global_params, "random_seed", 42)
    rng = np.random.default_rng(seed)

    print("Step 1: Check initial snapshot indexing")
    print(f"  Snapshot index type: {type(initial_snapshot.index)}")
    print(f"  Snapshot index dtype: {initial_snapshot.index.dtype}")
    print(f"  Sample index values: {initial_snapshot.index[:3].tolist()}")

    # Run orchestrator for 2026
    print("\nStep 2: Running orchestrator for 2026...")

    cumulative_event_log, eoy_snapshot = run_one_year(
        event_log=initial_event_log,
        prev_snapshot=initial_snapshot,
        year=2026,
        global_params=global_params,
        plan_rules={},
        hazard_table=hazard_table,
        rng=rng,
        census_template_path=None,
        rng_seed_offset=2026,
        deterministic_term=True,
    )

    print("\nStep 3: Check event log indexing")
    comp_events = cumulative_event_log[cumulative_event_log["event_type"] == "EVT_COMP"]

    if not comp_events.empty:
        print(f"  Number of EVT_COMP events: {len(comp_events)}")
        print(f"  Event employee_id dtype: {comp_events['employee_id'].dtype}")
        print(f"  Sample event employee_ids: {comp_events['employee_id'].head(3).tolist()}")

        print("\nStep 4: Check EOY snapshot indexing")
        print(f"  EOY snapshot index type: {type(eoy_snapshot.index)}")
        print(f"  EOY snapshot index dtype: {eoy_snapshot.index.dtype}")
        print(f"  Sample EOY index values: {eoy_snapshot.index[:3].tolist()}")

        print("\nStep 5: Check matching between events and snapshot")
        event_emp_ids = set(comp_events["employee_id"].unique())
        snapshot_emp_ids = set(eoy_snapshot.index.unique())

        # Find employees with events but not in final snapshot
        missing_from_snapshot = event_emp_ids - snapshot_emp_ids
        # Find employees in snapshot but no events
        missing_events = snapshot_emp_ids - event_emp_ids

        print(f"  Employees with EVT_COMP events: {len(event_emp_ids)}")
        print(f"  Employees in EOY snapshot: {len(snapshot_emp_ids)}")
        print(f"  Events but missing from snapshot: {len(missing_from_snapshot)}")
        print(f"  In snapshot but no events: {len(missing_events)}")

        if missing_from_snapshot:
            print(f"  Sample missing from snapshot: {list(missing_from_snapshot)[:3]}")

        # Check specific employees from user's data
        test_employees = ["NEW_900000003", "NEW_900000006", "NEW_900000017"]
        print("\nStep 6: Check specific problematic employees")

        for emp_id in test_employees:
            has_events = emp_id in event_emp_ids
            in_snapshot = emp_id in snapshot_emp_ids
            print(f"  {emp_id}: Events={has_events}, In_Snapshot={in_snapshot}")

            if has_events and in_snapshot:
                # Check compensation change
                if emp_id in initial_snapshot.index and emp_id in eoy_snapshot.index:
                    initial_comp = initial_snapshot.loc[emp_id, EMP_GROSS_COMP]
                    final_comp = eoy_snapshot.loc[emp_id, EMP_GROSS_COMP]
                    print(f"    Compensation: ${initial_comp:,.2f} → ${final_comp:,.2f}")

                    if abs(final_comp - initial_comp) < 0.01:
                        print(f"    ❌ FLAT compensation despite having events!")
                    else:
                        print(f"    ✅ Compensation changed correctly")

    else:
        print("  No EVT_COMP events found")


if __name__ == "__main__":
    debug_indexing_issue()
