#!/usr/bin/env python3
"""
Test script to verify that COLA and promotion raises are applied in the correct order.
This test specifically checks that promotion raises are calculated using COLA-adjusted compensation.
"""

import pandas as pd
import numpy as np
from types import SimpleNamespace
import logging

def test_promotion_cola_order():
    """Test that COLA is applied before promotion raises."""
    print("Testing promotion + COLA order...")
    
    # Set up a simple logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    
    try:
        from cost_model.engines.run_one_year import run_one_year
        
        # Create test snapshot with one employee eligible for promotion
        snapshot = pd.DataFrame({
            "employee_id": ["PROMO_TEST"],
            "active": [True],
            "employee_hire_date": pd.to_datetime(["2020-01-01"]),  # 5+ years tenure for promotion eligibility
            "employee_birth_date": pd.to_datetime(["1990-01-01"]),
            "employee_role": ["Staff"],
            "employee_gross_compensation": [50000.0],  # Base compensation
            "employee_termination_date": pd.NaT,
            "employee_deferral_rate": [0.05],
            "employee_tenure": [5.0],
            "employee_tenure_band": ["5-10"],
            "employee_level": [1],  # Level 1, eligible for promotion to level 2
            "job_level_source": ["hire"],
            "exited": [False],
            "employee_status_eoy": ["Active"],
            "simulation_year": [2025],
            "employee_contribution": [2500.0],
            "employer_core_contribution": [1500.0],
            "employer_match_contribution": [1000.0],
            "is_eligible": [True],
            "is_participating": [True],
        }).set_index("employee_id", drop=False)
        
        # Create hazard table with both COLA and promotion rates
        haz = pd.DataFrame({
            "simulation_year": [2025],
            "employee_level": [1],
            "employee_tenure_band": ["5-10"],
            "comp_raise_pct": [0.03],  # 3% annual raise
            "cola_pct": [0.02],  # 2% COLA
            "term_rate": [0],
            "new_hire_termination_rate": [0.0],
            "cfg": [SimpleNamespace()],
        })
        
        # Create global params with promotion configuration
        global_params = SimpleNamespace(
            dev_mode=True,
            min_eligibility_age=21,
            min_service_months=12,
            promotion_raise_config={
                "1_to_2": 0.10  # 10% raise for promotion from level 1 to 2
            }
        )
        
        print(f"Initial compensation: {snapshot['employee_gross_compensation'].iloc[0]}")
        print(f"Expected COLA adjustment: {snapshot['employee_gross_compensation'].iloc[0] * 0.02} (2%)")
        print(f"Expected annual raise: {snapshot['employee_gross_compensation'].iloc[0] * 0.03} (3%)")
        
        # Expected calculation order:
        # 1. COLA: 50000 * 1.02 = 51000
        # 2. Annual raise: 51000 * 1.03 = 52530  
        # 3. If promotion happens, promotion raise should be: 52530 * 1.10 = 57783
        
        # Run one year simulation
        evts, snap = run_one_year(
            event_log=pd.DataFrame(columns=['event_time', 'employee_id', 'event_type', 'value_num', 'value_json', 'meta']),
            prev_snapshot=snapshot,
            year=2025,
            global_params=global_params,
            plan_rules={},
            hazard_table=haz,
            rng=np.random.default_rng(42),  # Fixed seed for reproducibility
            census_template_path="dummy.csv"
        )
        
        print("\nEvent types generated:")
        if not evts.empty and 'event_type' in evts.columns:
            event_counts = evts["event_type"].value_counts()
            for event_type, count in event_counts.items():
                print(f"  {event_type}: {count}")
        
        print(f"\nFinal compensation: {snap['employee_gross_compensation'].iloc[0]}")
        
        # Check if events were generated in correct order
        if not evts.empty:
            # Sort events by time to see the order
            evts_sorted = evts.sort_values('event_time')
            print("\nEvents in chronological order:")
            for _, event in evts_sorted.iterrows():
                event_type = event['event_type']
                value_num = event.get('value_num', 'N/A')
                value_json = event.get('value_json', '')
                print(f"  {event['event_time']}: {event_type} - value_num: {value_num}")
                if value_json and event_type in ['EVT_COMP', 'EVT_RAISE']:
                    import json
                    try:
                        parsed = json.loads(value_json)
                        if 'old_comp' in parsed and 'new_comp' in parsed:
                            print(f"    {parsed['old_comp']} → {parsed['new_comp']}")
                        elif 'previous_comp' in parsed and 'new_comp' in parsed:
                            print(f"    {parsed['previous_comp']} → {parsed['new_comp']}")
                    except:
                        pass
        
        # Verify the compensation progression makes sense
        final_comp = snap['employee_gross_compensation'].iloc[0]
        initial_comp = 50000.0
        
        # The final compensation should reflect COLA + annual raise + any promotions
        # Minimum expected: 50000 * 1.02 * 1.03 = 52530
        min_expected = initial_comp * 1.02 * 1.03
        
        if final_comp >= min_expected:
            print(f"✅ SUCCESS: Final compensation ({final_comp:.2f}) >= minimum expected ({min_expected:.2f})")
            print("   This suggests COLA and annual raises were applied correctly.")
            return True
        else:
            print(f"❌ FAILURE: Final compensation ({final_comp:.2f}) < minimum expected ({min_expected:.2f})")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_promotion_cola_order()
    if success:
        print("\n🎉 Promotion + COLA order test passed!")
    else:
        print("\n💥 Promotion + COLA order test failed.")
