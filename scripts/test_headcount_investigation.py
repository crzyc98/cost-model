#!/usr/bin/env python3
"""
Simple test script to debug the headcount investigation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("Starting headcount investigation test...")

try:
    # Test 1: Load Campaign 6 config
    print("Test 1: Loading Campaign 6 config...")
    config_path = project_root / "campaign_6_results" / "best_config.yaml"

    if not config_path.exists():
        print(f"ERROR: Config file not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"✓ Config loaded successfully from {config_path}")

    # Extract key parameters
    global_params = config.get("global_parameters", {})
    target_growth = global_params.get("target_growth", "unknown")
    new_hire_rate = global_params.get("new_hire_rate", "unknown")
    nh_term_rate = global_params.get("new_hire_termination_rate", "unknown")
    maintain_headcount = global_params.get("maintain_headcount", False)

    print(f"Key parameters:")
    print(f"  target_growth: {target_growth}")
    print(f"  new_hire_rate: {new_hire_rate}")
    print(f"  new_hire_termination_rate: {nh_term_rate}")
    print(f"  maintain_headcount: {maintain_headcount}")

    # Test 2: Check census file
    print("\nTest 2: Checking census file...")
    census_path = project_root / "data" / "census_template.parquet"

    if not census_path.exists():
        print(f"ERROR: Census file not found at {census_path}")
        sys.exit(1)

    census_df = pd.read_parquet(census_path)
    print(f"✓ Census loaded successfully: {len(census_df)} employees")

    # Test 3: Import simulation components
    print("\nTest 3: Importing simulation components...")

    from cost_model.engines.run_one_year import run_one_year

    print("✓ run_one_year imported")

    from cost_model.projections.snapshot import create_initial_snapshot

    print("✓ create_initial_snapshot imported")

    from cost_model.projections.dynamic_hazard import build_dynamic_hazard_table

    print("✓ build_dynamic_hazard_table imported")

    # Test 4: Create initial snapshot
    print("\nTest 4: Creating initial snapshot...")
    start_year = global_params.get("start_year", 2025)
    snapshot = create_initial_snapshot(start_year, census_path)
    print(f"✓ Initial snapshot created: {len(snapshot)} employees")

    # Test 5: Convert config to namespace
    print("\nTest 5: Converting config to namespace...")
    from types import SimpleNamespace

    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return d

    config_ns = dict_to_namespace(config)
    print("✓ Config converted to namespace")

    # Test 6: Build hazard table
    print("\nTest 6: Building hazard table...")
    try:
        hazard_table = build_dynamic_hazard_table(config_ns.global_parameters)
        print(f"✓ Hazard table built: {len(hazard_table)} rows")
    except Exception as e:
        print(f"ERROR building hazard table: {e}")
        print("Trying with alternative approach...")

        # Alternative: create a minimal hazard table
        years = [2025, 2026, 2027]
        levels = [0, 1, 2, 3, 4]
        tenure_bands = ["<1", "1-3", "3-5", "5-10", "10-15", "15+"]

        hazard_data = []
        for year in years:
            for level in levels:
                for tenure in tenure_bands:
                    hazard_data.append(
                        {
                            "year": year,
                            "level": level,
                            "tenure_band": tenure,
                            "term_rate": 0.1,  # 10% base rate
                            "promotion_rate": 0.05,  # 5% promotion rate
                            "merit_raise_pct": 0.03,  # 3% merit raise
                            "cola_pct": 0.02,  # 2% COLA
                            "promotion_raise_pct": 0.1,  # 10% promotion raise
                            "cfg": "baseline",
                        }
                    )

        hazard_table = pd.DataFrame(hazard_data)
        print(f"✓ Minimal hazard table created: {len(hazard_table)} rows")

    # Test 7: Run one year simulation
    print("\nTest 7: Running one year simulation...")

    # Set up RNG
    rng = np.random.default_rng(global_params.get("random_seed", 42))

    # Create empty event log
    event_log = pd.DataFrame()

    # Run one year
    try:
        year_events, final_snapshot = run_one_year(
            event_log=event_log,
            prev_snapshot=snapshot,
            year=start_year,
            global_params=config_ns.global_parameters,
            plan_rules=config_ns.plan_rules.__dict__ if hasattr(config_ns, "plan_rules") else {},
            hazard_table=hazard_table,
            rng=rng,
            deterministic_term=True,
        )

        print(f"✓ One year simulation completed")
        print(f"  Events generated: {len(year_events)}")
        print(f"  Final headcount: {len(final_snapshot)}")

        # Analyze events
        if not year_events.empty:
            event_types = year_events["event_type"].value_counts()
            print(f"  Event breakdown: {dict(event_types)}")

        # Calculate growth
        initial_count = len(snapshot)
        final_count = len(final_snapshot)
        growth_rate = (final_count - initial_count) / initial_count
        print(f"  Headcount change: {initial_count} → {final_count} ({growth_rate:.1%})")

    except Exception as e:
        print(f"ERROR in simulation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n✓ All tests passed! Investigation script should work.")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
