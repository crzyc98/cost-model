#!/usr/bin/env python3
"""
Test script to verify that the promotion fix works correctly.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from cost_model.config.loaders import load_config_to_namespace
from cost_model.projections.snapshot import create_initial_snapshot
from cost_model.engines.run_one_year.orchestrator import run_one_year
from cost_model.projections.event_log import create_initial_event_log
from cost_model.projections.hazard import load_and_expand_hazard_table

def test_promotion_fix():
    """Test that promotions now work correctly instead of just demotions."""
    
    print("=== TESTING PROMOTION FIX ===")
    
    # Load fixed config
    config = load_config_to_namespace(Path("config/dev_tiny.yaml"))
    global_params = config.global_parameters
    
    # Check that the config is using the correct promotion matrix
    promotion_path = getattr(global_params, 'promotion_matrix_path', None)
    print(f"Promotion matrix path: {promotion_path}")
    
    if promotion_path is None or promotion_path == "~":
        print("❌ ERROR: Config still using null promotion matrix path!")
        return False
        
    # Create test data
    initial_snapshot = create_initial_snapshot(2025, "data/census_preprocessed.parquet")
    initial_event_log = create_initial_event_log(2025)
    hazard_table = load_and_expand_hazard_table('data/hazard_table.parquet')
    
    # Use same random seed for reproducibility
    seed = getattr(global_params, 'random_seed', 42)
    rng = np.random.default_rng(seed)
    
    print(f"Running simulation with promotion matrix: {promotion_path}")
    print(f"Initial snapshot size: {len(initial_snapshot)} employees")
    
    # Run 2026 simulation
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
        deterministic_term=True
    )
    
    # Check promotion events
    promotion_events = cumulative_event_log[
        cumulative_event_log['event_type'] == 'EVT_PROMOTION'
    ]
    
    if promotion_events.empty:
        print("❌ No promotion events found")
        return False
    
    print(f"\nFound {len(promotion_events)} promotion events")
    
    # Analyze promotion directions
    import json
    promotions_up = 0
    promotions_down = 0
    
    for _, event in promotion_events.head(10).iterrows():  # Check first 10
        try:
            data = json.loads(event['value_json'])
            from_level = data.get('from_level')
            to_level = data.get('to_level')
            
            if from_level is not None and to_level is not None:
                print(f"  {event['employee_id']}: Level {from_level} → Level {to_level}")
                
                if to_level > from_level:
                    promotions_up += 1
                elif to_level < from_level:
                    promotions_down += 1
                    
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Error parsing event: {e}")
    
    print(f"\nPromotion Analysis:")
    print(f"  Actual promotions (up): {promotions_up}")
    print(f"  Demotions (down): {promotions_down}")
    
    if promotions_up > 0:
        print("✅ SUCCESS: Found actual promotions (to_level > from_level)!")
        return True
    else:
        print("❌ FAILURE: Still only seeing demotions")
        return False

if __name__ == "__main__":
    success = test_promotion_fix()
    
    if success:
        print("\n🎉 PROMOTION FIX VERIFIED!")
        print("   Run a full simulation to see proper career progression.")
    else:
        print("\n❌ PROMOTION FIX FAILED!")
        print("   Additional investigation needed.")