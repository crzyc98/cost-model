#!/usr/bin/env python3
"""
Debug script to compare the eoy_snapshot from orchestrator vs enhanced_yearly_snapshot.
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
from cost_model.projections.snapshot import build_enhanced_yearly_snapshot, create_initial_snapshot
from cost_model.state.schema import EMP_GROSS_COMP, EMP_ID

logger = get_logger(__name__)


def test_snapshot_comparison():
    """Compare eoy_snapshot from orchestrator vs enhanced_yearly_snapshot."""

    print("=== COMPARING EOY_SNAPSHOT VS ENHANCED_YEARLY_SNAPSHOT ===")

    # Load minimal config for testing
    config = load_config_to_namespace(Path("config/dev_tiny.yaml"))
    global_params = config.global_parameters

    # Create initial snapshot and event log for 2026
    initial_snapshot = create_initial_snapshot(2025, "data/census_preprocessed.parquet")
    initial_event_log = create_initial_event_log(2025)

    # Load hazard table
    hazard_table = load_and_expand_hazard_table("data/hazard_table.parquet")

    # Initialize RNG
    rng = np.random.default_rng(42)

    # Store 2025 snapshot (start of 2026)
    snapshot_2025 = initial_snapshot.copy()

    print("Step 1: Run orchestrator for 2026")

    # Run the orchestrator for 2026 (this applies compensation events)
    cumulative_event_log, eoy_snapshot_2026 = run_one_year(
        event_log=initial_event_log,
        prev_snapshot=snapshot_2025,
        year=2026,
        global_params=global_params,
        plan_rules={},
        hazard_table=hazard_table,
        rng=rng,
        census_template_path=None,
        rng_seed_offset=2026,
        deterministic_term=True,
    )

    # Check NH_2025_0021 compensation in eoy_snapshot
    emp_id = "NH_2025_0021"
    if emp_id in eoy_snapshot_2026.index:
        eoy_comp = eoy_snapshot_2026.loc[emp_id, EMP_GROSS_COMP]
        print(f"EOY Snapshot compensation for {emp_id}: ${eoy_comp:,.2f}")
    else:
        print(f"Employee {emp_id} not found in EOY snapshot")
        return False

    print("\\nStep 2: Build enhanced yearly snapshot (like CLI does)")

    # Get 2026 events
    year_events_2026 = (
        cumulative_event_log[cumulative_event_log["simulation_year"] == 2026]
        if "simulation_year" in cumulative_event_log.columns
        else cumulative_event_log
    )

    # Build enhanced snapshot (this is what CLI saves)
    enhanced_snapshot_2026 = build_enhanced_yearly_snapshot(
        start_of_year_snapshot=snapshot_2025,
        end_of_year_snapshot=eoy_snapshot_2026,
        year_events=year_events_2026,
        simulation_year=2026,
    )

    # Check NH_2025_0021 compensation in enhanced_snapshot
    enhanced_emp = enhanced_snapshot_2026[enhanced_snapshot_2026[EMP_ID] == emp_id]
    if not enhanced_emp.empty:
        enhanced_comp = enhanced_emp[EMP_GROSS_COMP].iloc[0]
        print(f"Enhanced Snapshot compensation for {emp_id}: ${enhanced_comp:,.2f}")
    else:
        print(f"Employee {emp_id} not found in enhanced snapshot")
        return False

    print("\\nStep 3: Compare results")

    diff = eoy_comp - enhanced_comp
    print(f"Difference (EOY - Enhanced): ${diff:,.2f}")

    if abs(diff) < 0.01:
        print("✅ SNAPSHOTS MATCH: The enhanced snapshot preserves compensation correctly")
        return True
    else:
        print("❌ BUG FOUND: Enhanced snapshot has different compensation than EOY snapshot!")
        print(f"   This explains why the CLI saves incorrect compensation values.")

        # Show expected compensation
        print("\\nExpected compensation based on events:")
        print("  Starting: $69,108.49")
        print("  Merit raise: $71,872.83 (total)")
        print("  COLA: +$1,243.95")
        print("  Expected final: $73,116.78")

        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = test_snapshot_comparison()

    if not success:
        print(
            "\\n💡 SOLUTION: The CLI should save eoy_snapshot instead of enhanced_yearly_snapshot"
        )
        print("   OR the build_enhanced_yearly_snapshot function needs to be fixed.")
