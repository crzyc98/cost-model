#!/usr/bin/env python3
"""
Debug script to trace compensation application in the orchestrator context.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from logging_config import get_logger

from cost_model.config.loaders import load_config_to_namespace
from cost_model.engines.run_one_year.orchestrator import run_one_year
from cost_model.projections.event_log import create_initial_event_log
from cost_model.projections.hazard import load_and_expand_hazard_table
from cost_model.projections.snapshot import create_initial_snapshot
from cost_model.state.schema import EMP_GROSS_COMP, EMP_ID

# Set up targeted logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


def debug_orchestrator_compensation():
    """Debug compensation application in orchestrator for specific employee."""

    print("=== DEBUGGING ORCHESTRATOR COMPENSATION APPLICATION ===")

    # Load config
    config = load_config_to_namespace(Path("config/dev_tiny.yaml"))
    global_params = config.global_parameters

    # Create initial data
    initial_snapshot = create_initial_snapshot(2025, "data/census_preprocessed.parquet")
    initial_event_log = create_initial_event_log(2025)
    hazard_table = load_and_expand_hazard_table("data/hazard_table.parquet")

    # Use same random seed as real simulation
    seed = getattr(global_params, "random_seed", 42)
    rng = np.random.default_rng(seed)

    # Track specific employee
    emp_id = "NEW_900000017"

    print(f"Tracking employee: {emp_id}")
    print(f"Initial compensation: ${initial_snapshot.loc[emp_id, EMP_GROSS_COMP]:,.2f}")

    # Run 2026 simulation
    print("\n=== RUNNING 2026 SIMULATION ===")

    # Test the compensation application step by step
    current_snapshot = initial_snapshot.copy()

    # Check compensation before orchestrator
    comp_before = current_snapshot.loc[emp_id, EMP_GROSS_COMP]
    print(f"Compensation before 2026 orchestrator: ${comp_before:,.2f}")

    # Run orchestrator
    cumulative_event_log, eoy_snapshot = run_one_year(
        event_log=initial_event_log,
        prev_snapshot=current_snapshot,
        year=2026,
        global_params=global_params,
        plan_rules={},
        hazard_table=hazard_table,
        rng=rng,
        census_template_path=None,
        rng_seed_offset=2026,
        deterministic_term=True,
    )

    # Check compensation after orchestrator
    if emp_id in eoy_snapshot.index:
        comp_after = eoy_snapshot.loc[emp_id, EMP_GROSS_COMP]
        print(f"Compensation after 2026 orchestrator: ${comp_after:,.2f}")

        # Check events generated
        events_2026 = cumulative_event_log[
            (cumulative_event_log["employee_id"] == emp_id)
            & (cumulative_event_log["simulation_year"] == 2026)
        ]

        comp_events = events_2026[events_2026["event_type"] == "EVT_COMP"]
        cola_events = events_2026[events_2026["event_type"] == "EVT_COLA"]

        print(f"\nEvents generated for {emp_id} in 2026:")
        if not comp_events.empty:
            event = comp_events.iloc[0]
            print(f"  EVT_COMP: ${event['value_num']:,.2f} (new total)")
            print(f"  Meta: {event['meta']}")

        if not cola_events.empty:
            event = cola_events.iloc[0]
            print(f"  EVT_COLA: ${event['value_num']:,.2f} (amount to add)")
            print(f"  Meta: {event['meta']}")

        # Calculate expected compensation
        if not comp_events.empty and not cola_events.empty:
            expected_comp = comp_events.iloc[0]["value_num"] + cola_events.iloc[0]["value_num"]
            print(f"\nExpected final compensation: ${expected_comp:,.2f}")
            print(f"Actual final compensation: ${comp_after:,.2f}")
            print(f"Difference: ${comp_after - expected_comp:,.2f}")

            if abs(comp_after - expected_comp) > 0.01:
                print("❌ COMPENSATION EVENTS NOT APPLIED TO SNAPSHOT!")
                return False
            else:
                print("✅ Compensation events applied correctly")
                return True
        else:
            print("❌ No compensation events generated")
            return False
    else:
        print(f"❌ Employee {emp_id} not found in EOY snapshot")
        return False


if __name__ == "__main__":
    success = debug_orchestrator_compensation()

    if not success:
        print("\n💡 DIAGNOSIS:")
        print("   The orchestrator is either:")
        print("   1. Not applying compensation events to the internal snapshot")
        print("   2. Applying events but returning wrong snapshot")
        print("   3. Has a bug in the snapshot_update.update() function")
        print("   4. Has indexing issues with employee IDs")
