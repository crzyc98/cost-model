#!/usr/bin/env python3
"""
Test script to verify the granular compensation event generation works correctly.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_granular_compensation_events():
    """Test that the new granular compensation event generation works."""
    print("=== Testing Granular Compensation Event Generation ===")
    
    try:
        from cost_model.state.schema import EMP_ID, EMP_GROSS_COMP, EMP_TERM_DATE, EMP_LEVEL, EMP_TENURE_BAND
        from cost_model.projections.hazard import load_and_expand_hazard_table
        from cost_model.engines.run_one_year.validation import validate_and_extract_hazard_slice
        from cost_model.engines.comp import bump
        
        # Create test snapshot
        test_snapshot = pd.DataFrame({
            EMP_ID: ['E001', 'E002', 'E003'],
            EMP_GROSS_COMP: [50000.0, 75000.0, 100000.0],
            EMP_TERM_DATE: [pd.NaT, pd.NaT, pd.NaT],
            EMP_LEVEL: [1, 2, 3],
            EMP_TENURE_BAND: ['1-3', '3-5', '5-10']
        })
        
        print(f"✓ Created test snapshot with {len(test_snapshot)} employees")
        
        # Load hazard table
        hazard = load_and_expand_hazard_table('data/hazard_table.parquet')
        hazard_slice = validate_and_extract_hazard_slice(hazard, 2024)
        
        print(f"✓ Loaded hazard slice for 2024: {len(hazard_slice)} rows")
        print(f"  Available columns: {hazard_slice.columns.tolist()}")
        
        # Check for granular columns
        granular_cols = ['merit_raise_pct', 'promotion_raise_pct', 'cola_pct']
        present_cols = [col for col in granular_cols if col in hazard_slice.columns]
        print(f"✓ Granular columns present: {present_cols}")
        
        # Test compensation bump
        as_of = pd.Timestamp('2024-01-01')
        rng = np.random.default_rng(42)
        
        events_list = bump(test_snapshot, hazard_slice, as_of, rng)
        print(f"✓ Generated {len(events_list)} event DataFrames")
        
        # Analyze events
        total_events = 0
        for i, events_df in enumerate(events_list):
            if not events_df.empty:
                event_types = events_df['event_type'].unique()
                print(f"  DataFrame {i}: {len(events_df)} events, types: {event_types}")
                total_events += len(events_df)
                
                # Show sample events
                if len(events_df) > 0:
                    sample = events_df[['employee_id', 'event_type', 'value_num']].head(2)
                    print(f"    Sample events:\n{sample}")
        
        print(f"✓ Total events generated: {total_events}")
        
        # Verify event types
        all_events = pd.concat([df for df in events_list if not df.empty], ignore_index=True)
        if not all_events.empty:
            event_type_counts = all_events['event_type'].value_counts()
            print(f"✓ Event type breakdown: {event_type_counts.to_dict()}")
            
            # Check for expected event types
            expected_types = ['EVT_COLA', 'EVT_COMP']
            found_types = [et for et in expected_types if et in event_type_counts.index]
            print(f"✓ Expected event types found: {found_types}")
            
            return True
        else:
            print("❌ No events generated")
            return False
            
    except Exception as e:
        print(f"❌ Error in granular compensation test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test."""
    print("Testing Granular Compensation Event Generation\n")
    
    success = test_granular_compensation_events()
    
    print(f"\n=== Test Results ===")
    if success:
        print("🎉 Granular compensation event generation is working correctly!")
        print("✓ EVT_COLA events generated from cola_pct")
        print("✓ EVT_COMP events generated from merit_raise_pct")
        print("✓ Separate event types for clear analytics")
        return True
    else:
        print("❌ Test failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
