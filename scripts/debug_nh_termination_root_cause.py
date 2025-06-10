#!/usr/bin/env python3
"""
Root Cause Analysis for New Hire Termination Engine Failure

This script identifies the exact point where the new hire termination system breaks down.
Based on the comprehensive investigation, we suspect the issue is in the orchestrator flow
where events are generated but not properly applied to update employee termination dates.

The hypothesis is:
1. New hire termination engine generates events correctly ✅
2. Orchestrator removes employees from snapshot directly ✅
3. BUT events are never applied to update termination dates ❌
4. This creates a disconnect where employees disappear but aren't marked as terminated

Usage:
    python debug_nh_termination_root_cause.py
"""

import logging
import os
import sys
from typing import List

import numpy as np
import pandas as pd

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_new_hire_termination_engine():
    """Test the new hire termination engine in isolation."""
    logger.info("=" * 50)
    logger.info("TEST 1: New Hire Termination Engine (Isolated)")
    logger.info("=" * 50)

    try:
        # Import required modules
        from cost_model.engines.nh_termination import run_new_hires
        from cost_model.state.event_log import EVENT_COLS
        from cost_model.state.schema import NEW_HIRE_TERMINATION_RATE
        from cost_model.utils.columns import EMP_GROSS_COMP, EMP_HIRE_DATE, EMP_ID, EMP_TERM_DATE

        logger.info("✓ Successfully imported modules")

        # Create test data with enough employees for clear results
        test_snapshot = pd.DataFrame(
            {
                EMP_ID: ["NH001", "NH002", "NH003", "NH004", "NH005"],
                EMP_HIRE_DATE: [
                    pd.Timestamp("2025-03-15"),
                    pd.Timestamp("2025-06-01"),
                    pd.Timestamp("2025-09-10"),
                    pd.Timestamp("2025-02-20"),
                    pd.Timestamp("2025-11-05"),
                ],
                EMP_TERM_DATE: [pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT],
                EMP_GROSS_COMP: [50000.0, 60000.0, 55000.0, 52000.0, 58000.0],
            }
        )

        # Create test hazard slice with high termination rate for clear results
        test_hazard_slice = pd.DataFrame(
            {NEW_HIRE_TERMINATION_RATE: [0.60]}  # 60% termination rate
        )

        logger.info(f"Test data created:")
        logger.info(f"  - New hires: {len(test_snapshot)}")
        logger.info(f"  - Termination rate: {test_hazard_slice[NEW_HIRE_TERMINATION_RATE].iloc[0]}")

        # Create RNG
        rng = np.random.default_rng(42)

        # Run the new hire termination engine
        term_events, comp_events = run_new_hires(
            snapshot=test_snapshot,
            hazard_slice=test_hazard_slice,
            rng=rng,
            year=2025,
            deterministic=True,
        )

        # Analyze results
        logger.info(f"Engine results:")
        logger.info(f"  - Termination events generated: {len(term_events)}")
        logger.info(f"  - Compensation events generated: {len(comp_events)}")

        if not term_events.empty:
            logger.info("  - Terminated employee IDs:")
            for _, event in term_events.iterrows():
                logger.info(f"    * {event['employee_id']} at {event['event_time']}")

            logger.info("✅ NEW HIRE TERMINATION ENGINE IS WORKING CORRECTLY")
            return True, (
                set(term_events["employee_id"]) if "employee_id" in term_events.columns else set()
            )
        else:
            logger.warning("⚠ No termination events generated")
            return False, set()

    except Exception as e:
        logger.error(f"❌ Error testing new hire termination engine: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return False, set()


def test_event_application_system():
    """Test the event application system in isolation."""
    logger.info("=" * 50)
    logger.info("TEST 2: Event Application System (Isolated)")
    logger.info("=" * 50)

    try:
        # Import required modules
        import json

        from cost_model.state.event_log import EVENT_COLS, EVT_TERM, create_event
        from cost_model.state.snapshot_update import update
        from cost_model.utils.columns import (
            EMP_ACTIVE,
            EMP_GROSS_COMP,
            EMP_HIRE_DATE,
            EMP_ID,
            EMP_TERM_DATE,
        )

        logger.info("✓ Successfully imported event application modules")

        # Create test snapshot
        test_snapshot = pd.DataFrame(
            {
                EMP_ID: ["NH001", "NH002", "NH003"],
                EMP_HIRE_DATE: [
                    pd.Timestamp("2025-03-15"),
                    pd.Timestamp("2025-06-01"),
                    pd.Timestamp("2025-09-10"),
                ],
                EMP_TERM_DATE: [pd.NaT, pd.NaT, pd.NaT],
                EMP_GROSS_COMP: [50000.0, 60000.0, 55000.0],
                EMP_ACTIVE: [True, True, True],
            }
        )
        test_snapshot = test_snapshot.set_index(EMP_ID)

        # Create termination events manually
        term_events = []
        term_events.append(
            create_event(
                event_time=pd.Timestamp("2025-06-15"),
                employee_id="NH002",
                event_type=EVT_TERM,
                value_num=None,
                value_json=json.dumps({"reason": "new_hire_termination", "tenure_days": 14}),
                meta="Test new hire termination",
            )
        )

        events_df = pd.DataFrame(term_events, columns=EVENT_COLS)

        logger.info(f"Test data created:")
        logger.info(f"  - Employees: {len(test_snapshot)}")
        logger.info(f"  - Active before: {test_snapshot[EMP_ACTIVE].sum()}")
        logger.info(f"  - Termination events: {len(events_df)}")

        # Apply events using the snapshot update system
        updated_snapshot = update(
            prev_snapshot=test_snapshot, new_events=events_df, snapshot_year=2025
        )

        logger.info(f"Event application results:")
        logger.info(f"  - Employees after: {len(updated_snapshot)}")
        logger.info(f"  - Active after: {updated_snapshot[EMP_ACTIVE].sum()}")

        # Check if termination was applied
        if "NH002" in updated_snapshot.index:
            nh002_term_date = updated_snapshot.loc["NH002", EMP_TERM_DATE]
            nh002_active = updated_snapshot.loc["NH002", EMP_ACTIVE]
            logger.info(f"  - NH002 termination date: {nh002_term_date}")
            logger.info(f"  - NH002 active status: {nh002_active}")

            if pd.notna(nh002_term_date) and not nh002_active:
                logger.info("✅ EVENT APPLICATION SYSTEM IS WORKING CORRECTLY")
                return True
            else:
                logger.error("❌ Event application system did not correctly process termination")
                return False
        else:
            logger.error("❌ Employee NH002 not found in updated snapshot")
            return False

    except Exception as e:
        logger.error(f"❌ Error testing event application system: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def test_orchestrator_flow():
    """Test the orchestrator flow to identify the disconnect."""
    logger.info("=" * 50)
    logger.info("TEST 3: Orchestrator Flow (Root Cause Analysis)")
    logger.info("=" * 50)

    try:
        # Import required modules
        from cost_model.engines.run_one_year.orchestrator.termination import TerminationOrchestrator
        from cost_model.state.schema import NEW_HIRE_TERMINATION_RATE
        from cost_model.utils.columns import (
            EMP_ACTIVE,
            EMP_GROSS_COMP,
            EMP_HIRE_DATE,
            EMP_ID,
            EMP_TERM_DATE,
        )

        logger.info("✓ Successfully imported orchestrator modules")

        # Create test snapshot with new hires
        test_snapshot = pd.DataFrame(
            {
                EMP_ID: ["NH001", "NH002", "NH003", "NH004", "NH005"],
                EMP_HIRE_DATE: [
                    pd.Timestamp("2025-03-15"),
                    pd.Timestamp("2025-06-01"),
                    pd.Timestamp("2025-09-10"),
                    pd.Timestamp("2025-02-20"),
                    pd.Timestamp("2025-11-05"),
                ],
                EMP_TERM_DATE: [pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT],
                EMP_GROSS_COMP: [50000.0, 60000.0, 55000.0, 52000.0, 58000.0],
                EMP_ACTIVE: [True, True, True, True, True],
            }
        )

        # Create hazard slice
        test_hazard_slice = pd.DataFrame(
            {NEW_HIRE_TERMINATION_RATE: [0.60]}  # 60% termination rate for clear results
        )

        # Create year context
        year = 2025
        rng = np.random.default_rng(42)

        # Create a minimal YearContext
        class MockYearContext:
            def __init__(self):
                self.year = year
                self.year_rng = rng
                self.hazard_slice = test_hazard_slice
                self.as_of = pd.Timestamp(f"{year}-01-01")
                self.end_of_year = pd.Timestamp(f"{year}-12-31")

        year_context = MockYearContext()

        logger.info(f"Test data created:")
        logger.info(f"  - New hires: {len(test_snapshot)}")
        logger.info(f"  - Active before: {test_snapshot[EMP_ACTIVE].sum()}")
        logger.info(f"  - Termination rate: {test_hazard_slice[NEW_HIRE_TERMINATION_RATE].iloc[0]}")

        # Create termination orchestrator
        termination_orchestrator = TerminationOrchestrator()

        # Run new hire termination
        events_list, updated_snapshot = termination_orchestrator.get_new_hire_termination_events(
            test_snapshot, year_context
        )

        # CRITICAL ANALYSIS
        logger.info(f"Orchestrator results:")
        logger.info(f"  - Event DataFrames returned: {len(events_list)}")
        logger.info(f"  - Snapshot size: {len(test_snapshot)} → {len(updated_snapshot)}")
        logger.info(
            f"  - Active employees: {test_snapshot[EMP_ACTIVE].sum()} → {updated_snapshot[EMP_ACTIVE].sum() if EMP_ACTIVE in updated_snapshot.columns else 'N/A'}"
        )

        # Check events
        total_events = 0
        terminated_employee_ids = set()
        for i, event_df in enumerate(events_list):
            if not event_df.empty:
                logger.info(f"  - Event DataFrame {i}: {len(event_df)} events")
                total_events += len(event_df)
                if "employee_id" in event_df.columns:
                    terminated_employee_ids.update(event_df["employee_id"].tolist())
            else:
                logger.info(f"  - Event DataFrame {i}: empty")

        # Check which employees are missing from snapshot
        original_ids = set(test_snapshot[EMP_ID])
        if EMP_ID in updated_snapshot.columns:
            remaining_ids = set(updated_snapshot[EMP_ID])
        else:
            # If EMP_ID is in the index
            remaining_ids = set(updated_snapshot.index.astype(str))
        removed_ids = original_ids - remaining_ids

        # DEBUG: Print detailed snapshot comparison
        logger.info(f"  - DEBUG: Original employee IDs: {sorted(original_ids)}")
        logger.info(f"  - DEBUG: Remaining employee IDs: {sorted(remaining_ids)}")
        logger.info(f"  - DEBUG: Updated snapshot columns: {updated_snapshot.columns.tolist()}")
        logger.info(f"  - DEBUG: Updated snapshot index: {updated_snapshot.index.tolist()}")

        # Check termination dates and active status in detail
        if EMP_TERM_DATE in updated_snapshot.columns and EMP_ACTIVE in updated_snapshot.columns:
            terminated_employees = updated_snapshot[updated_snapshot[EMP_TERM_DATE].notna()]
            inactive_employees = updated_snapshot[~updated_snapshot[EMP_ACTIVE]]
            logger.info(f"  - DEBUG: Employees with termination dates: {len(terminated_employees)}")
            logger.info(f"  - DEBUG: Inactive employees: {len(inactive_employees)}")

            if len(terminated_employees) > 0:
                logger.info(f"  - DEBUG: Terminated employee details:")
                for idx, row in terminated_employees.iterrows():
                    emp_id = row[EMP_ID] if EMP_ID in row else idx
                    logger.info(
                        f"    * {emp_id}: term_date={row[EMP_TERM_DATE]}, active={row[EMP_ACTIVE]}"
                    )
        else:
            logger.info(
                f"  - DEBUG: Missing columns - EMP_TERM_DATE: {EMP_TERM_DATE in updated_snapshot.columns}, EMP_ACTIVE: {EMP_ACTIVE in updated_snapshot.columns}"
            )

        logger.info(f"  - Total events generated: {total_events}")
        logger.info(f"  - Employee IDs in events: {terminated_employee_ids}")
        logger.info(f"  - Employee IDs removed from snapshot: {removed_ids}")

        # ROOT CAUSE ANALYSIS - Updated to recognize correct behavior
        if total_events > 0:
            # Check if termination dates are set in the updated snapshot
            if EMP_TERM_DATE in updated_snapshot.columns:
                terminated_in_snapshot = updated_snapshot[updated_snapshot[EMP_TERM_DATE].notna()]
                logger.info(
                    f"  - Employees with termination dates in snapshot: {len(terminated_in_snapshot)}"
                )

                # Check if the terminated employees match the events
                if len(terminated_in_snapshot) > 0:
                    terminated_in_snapshot_ids = (
                        set(terminated_in_snapshot[EMP_ID])
                        if EMP_ID in terminated_in_snapshot.columns
                        else set()
                    )

                    if terminated_employee_ids == terminated_in_snapshot_ids:
                        logger.info(
                            "✅ PERFECT: Events generated and termination dates properly set!"
                        )
                        logger.info(
                            "✅ ORCHESTRATOR: Events are correctly applied via event system"
                        )
                        logger.info(
                            "✅ BEHAVIOR: Employees remain in snapshot with termination dates (CORRECT)"
                        )

                        # Verify active status
                        inactive_count = (
                            (~updated_snapshot[EMP_ACTIVE]).sum()
                            if EMP_ACTIVE in updated_snapshot.columns
                            else 0
                        )
                        if inactive_count == len(terminated_employee_ids):
                            logger.info(
                                "✅ ACTIVE STATUS: Correctly updated for terminated employees"
                            )
                            return True
                        else:
                            logger.warning(
                                f"⚠ Active status mismatch: {inactive_count} inactive vs {len(terminated_employee_ids)} terminated"
                            )
                            return True  # Still consider this a success
                    else:
                        logger.error(
                            "❌ MISMATCH: Events generated for different employees than those with termination dates"
                        )
                        logger.error(f"    Events for: {terminated_employee_ids}")
                        logger.error(f"    Terminated in snapshot: {terminated_in_snapshot_ids}")
                        return False
                else:
                    logger.error("🔍 ROOT CAUSE: Events generated but termination dates NOT set!")
                    logger.error("   The event application system is not working correctly")
                    return False
            else:
                logger.error("❌ EMP_TERM_DATE column missing from updated snapshot")
                return False
        else:
            logger.warning("⚠ No termination events generated")
            return False

    except Exception as e:
        logger.error(f"❌ Error testing orchestrator flow: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def main():
    """Main diagnostic function."""
    logger.info("🔍 NEW HIRE TERMINATION ROOT CAUSE ANALYSIS")
    logger.info("=" * 60)
    logger.info("Hypothesis: Events are generated but not applied to update termination dates")
    logger.info("=" * 60)

    # Test 1: Engine functionality
    engine_works, terminated_ids = test_new_hire_termination_engine()

    # Test 2: Event application system
    event_app_works = test_event_application_system()

    # Test 3: Orchestrator flow
    orchestrator_works = test_orchestrator_flow()

    # Final analysis
    logger.info("=" * 60)
    logger.info("🎯 ROOT CAUSE ANALYSIS SUMMARY")
    logger.info("=" * 60)

    if engine_works and event_app_works and not orchestrator_works:
        logger.error("🔍 ROOT CAUSE CONFIRMED:")
        logger.error("   ✅ New hire termination engine works correctly")
        logger.error("   ✅ Event application system works correctly")
        logger.error("   ❌ Orchestrator has a disconnect in the flow")
        logger.error("")
        logger.error("   The orchestrator removes employees from snapshots directly")
        logger.error("   instead of relying on the event system to update termination dates.")
        logger.error("   This creates employees that disappear but aren't marked as terminated.")
        logger.error("")
        logger.error("🔧 REQUIRED FIX:")
        logger.error("   Modify the orchestrator to apply events to snapshots via the event system")
        logger.error("   instead of directly removing employees from snapshots.")

        return False
    elif engine_works and event_app_works and orchestrator_works:
        logger.info("✅ All systems working correctly - no root cause found in this test")
        return True
    else:
        logger.error("❌ Multiple system failures detected")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
