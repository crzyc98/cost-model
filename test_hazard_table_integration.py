#!/usr/bin/env python3
"""
Test script to verify the hazard table integration works correctly.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_hazard_table_conversion():
    """Test that the conversion script works correctly."""
    print("=== Testing Hazard Table Conversion ===")
    
    # Check input file exists
    input_file = Path("data/generated_hazard_table_yaml_template.csv")
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return False
    
    print(f"✓ Input file exists: {input_file}")
    
    # Check output file exists
    output_file = Path("data/hazard_table.parquet")
    if not output_file.exists():
        print(f"❌ Output file not found: {output_file}")
        return False
    
    print(f"✓ Output file exists: {output_file}")
    
    # Load and check the parquet file
    try:
        df = pd.read_parquet(output_file)
        print(f"✓ Successfully loaded parquet file: {len(df)} rows")
        print(f"  Columns: {df.columns.tolist()}")
        
        # Check for required columns
        required_cols = ['simulation_year', 'employee_level', 'tenure_band', 'term_rate', 
                        'merit_raise_pct', 'promotion_raise_pct', 'cola_pct']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
        
        print("✓ All required columns present")
        return True
        
    except Exception as e:
        print(f"❌ Error loading parquet file: {e}")
        return False

def test_hazard_table_loading():
    """Test that the hazard table loading function works."""
    print("\n=== Testing Hazard Table Loading ===")
    
    try:
        from cost_model.projections.hazard import load_and_expand_hazard_table
        
        hazard = load_and_expand_hazard_table('data/hazard_table.parquet')
        print(f"✓ Successfully loaded hazard table: {len(hazard)} rows")
        
        # Check for backward compatibility column
        if 'comp_raise_pct' in hazard.columns:
            print("✓ Backward compatibility column 'comp_raise_pct' present")
        else:
            print("❌ Missing backward compatibility column 'comp_raise_pct'")
            return False
        
        # Check for new granular columns
        granular_cols = ['merit_raise_pct', 'promotion_raise_pct', 'promotion_rate']
        present_granular = [col for col in granular_cols if col in hazard.columns]
        print(f"✓ Granular columns present: {present_granular}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in hazard table loading: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_year_slice_extraction():
    """Test that year slice extraction works."""
    print("\n=== Testing Year Slice Extraction ===")
    
    try:
        from cost_model.projections.hazard import load_and_expand_hazard_table
        from cost_model.engines.run_one_year.validation import validate_and_extract_hazard_slice
        
        hazard = load_and_expand_hazard_table('data/hazard_table.parquet')
        slice_2024 = validate_and_extract_hazard_slice(hazard, 2024)
        
        print(f"✓ Successfully extracted 2024 slice: {len(slice_2024)} rows")
        
        # Check that it has the expected columns
        expected_cols = ['comp_raise_pct', 'merit_raise_pct', 'cola_pct']
        present_cols = [col for col in expected_cols if col in slice_2024.columns]
        print(f"✓ Expected columns in slice: {present_cols}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in year slice extraction: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Testing Hazard Table Integration\n")
    
    tests = [
        test_hazard_table_conversion,
        test_hazard_table_loading,
        test_year_slice_extraction,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! Hazard table integration is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
