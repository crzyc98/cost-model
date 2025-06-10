#!/usr/bin/env python3
"""
Final test to identify the exact bug in compensation application.
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

# Set up logging to capture orchestrator details
logging.basicConfig(level=logging.DEBUG)
logger = get_logger(__name__)


def test_compensation_bug():
    """Test the exact scenario from the real simulation."""

    print("=== REPRODUCING THE COMPENSATION BUG ===")

    # Use the EXACT same configuration as the failing simulation
    config = load_config_to_namespace(Path("config/dev_tiny.yaml"))
    global_params = config.global_parameters

    # Create initial snapshot and event log
    initial_snapshot = create_initial_snapshot(2025, "data/census_preprocessed.parquet")
    initial_event_log = create_initial_event_log(2025)

    # Load hazard table
    hazard_table = load_and_expand_hazard_table("data/hazard_table.parquet")

    # Use EXACT same random seed as the real simulation
    # The real simulation uses global_params.random_seed + year offset
    seed = getattr(global_params, "random_seed", 42)
    rng = np.random.default_rng(seed + 2026)  # Year offset for 2026

    emp_id = "NH_2025_0021"

    # Check initial compensation
    if emp_id in initial_snapshot.index:
        initial_comp = initial_snapshot.loc[emp_id, EMP_GROSS_COMP]
        print(f"Initial compensation (2025): ${initial_comp:,.2f}")
    else:
        print(f"Employee {emp_id} not found in initial snapshot")
        return False

    print("\nRunning orchestrator for 2026...")

    # Run orchestrator with same settings as CLI
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
        deterministic_term=getattr(global_params, "deterministic_termination", True),
    )

    # Check if employee survived
    if emp_id in eoy_snapshot.index:
        final_comp = eoy_snapshot.loc[emp_id, EMP_GROSS_COMP]
        print(f"Final compensation (EOY 2026): ${final_comp:,.2f}")

        # Check for compensation events
        comp_events = cumulative_event_log[
            (cumulative_event_log["employee_id"] == emp_id)
            & (cumulative_event_log["simulation_year"] == 2026)
            & (cumulative_event_log["event_type"] == "EVT_COMP")
        ]

        cola_events = cumulative_event_log[
            (cumulative_event_log["employee_id"] == emp_id)
            & (cumulative_event_log["simulation_year"] == 2026)
            & (cumulative_event_log["event_type"] == "EVT_COLA")
        ]

        print(f"\\nGenerated EVT_COMP events: {len(comp_events)}")
        print(f"Generated EVT_COLA events: {len(cola_events)}")

        if len(comp_events) > 0:
            expected_comp = comp_events["value_num"].iloc[0]
            if len(cola_events) > 0:
                expected_comp += cola_events["value_num"].iloc[0]

            print(f"Expected final compensation: ${expected_comp:,.2f}")
            diff = final_comp - expected_comp
            print(f"Compensation difference: ${diff:,.2f}")

            if abs(diff) > 0.01:
                print("❌ BUG CONFIRMED: Events generated but not applied to snapshot!")
                return False
            else:
                print("✅ Compensation applied correctly")
                return True
        else:
            print("❌ No compensation events generated for this employee")
            return False
    else:
        print(f"Employee {emp_id} was removed from snapshot during orchestrator")
        # Check if they were terminated
        term_events = cumulative_event_log[
            (cumulative_event_log["employee_id"] == emp_id)
            & (cumulative_event_log["simulation_year"] == 2026)
            & (cumulative_event_log["event_type"].isin(["EVT_TERM", "EVT_NEW_HIRE_TERM"]))
        ]

        if len(term_events) > 0:
            print(f"Employee was terminated: {term_events['meta'].iloc[0]}")
        else:
            print("Employee removed without termination event - potential bug!")

        return False


if __name__ == "__main__":
    success = test_compensation_bug()

    if not success:
        print("\\n💡 The bug is confirmed. The orchestrator is either:")
        print("   1. Not applying compensation events to the snapshot")
        print("   2. Overriding the updated snapshot with an old version")
        print("   3. Removing employees who should receive compensation")
