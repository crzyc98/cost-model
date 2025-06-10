"""
Test script to verify the unified schema system is working correctly.

This script tests the schema migration and unified column handling.
"""

import pandas as pd
import numpy as np
from cost_model.schema import (
    SnapshotColumns, 
    migrate_legacy_columns, 
    detect_schema_version,
    get_migration_recommendations,
    validate_snapshot_schema
)
from cost_model.projections.snapshot.transformers import SnapshotTransformer
from cost_model.projections.snapshot.models import SnapshotConfig

def test_schema_detection():
    """Test schema version detection."""
    print("=== Testing Schema Detection ===")
    
    # Test new schema (which uses descriptive names)
    new_schema_df = pd.DataFrame({
        SnapshotColumns.EMP_ID: ['EMP001', 'EMP002'],
        SnapshotColumns.EMP_HIRE_DATE: ['2020-01-01', '2021-01-01'],
        SnapshotColumns.EMP_GROSS_COMP: [75000, 85000],
        SnapshotColumns.SIMULATION_YEAR: [2024, 2024]  # Add a distinguishing column
    })
    
    version = detect_schema_version(new_schema_df)
    print(f"New schema detected as: {version}")
    assert version == "new", f"Expected 'new', got '{version}'"
    
    # Test legacy EMP_ prefix schema
    legacy_emp_df = pd.DataFrame({
        'EMP_ID': ['EMP001', 'EMP002'],
        'EMP_HIRE_DATE': ['2020-01-01', '2021-01-01'],
        'EMP_GROSS_COMP': [75000, 85000]
    })
    
    version = detect_schema_version(legacy_emp_df)
    print(f"Legacy EMP_ schema detected as: {version}")
    assert version == "legacy_emp_prefix", f"Expected 'legacy_emp_prefix', got '{version}'"
    
    # Test minimal descriptive schema (could be legacy or new)
    minimal_df = pd.DataFrame({
        'employee_id': ['EMP001', 'EMP002'],
        'employee_hire_date': ['2020-01-01', '2021-01-01'],
        'employee_gross_compensation': [75000, 85000]
    })
    
    version = detect_schema_version(minimal_df)
    print(f"Minimal descriptive schema detected as: {version}")
    # Should be detected as 'new' since we default to new for descriptive names
    assert version == "new", f"Expected 'new', got '{version}'"
    
    print("✅ Schema detection tests passed!\n")


def test_schema_migration():
    """Test schema migration functionality."""
    print("=== Testing Schema Migration ===")
    
    # Test migrating legacy EMP_ prefix schema
    legacy_df = pd.DataFrame({
        'EMP_ID': ['EMP001', 'EMP002', 'EMP003'],
        'EMP_HIRE_DATE': ['2020-01-01', '2021-01-01', '2019-06-15'],
        'EMP_GROSS_COMP': [75000, 85000, 65000],
        'EMP_ACTIVE': [True, True, False],
        'EMP_DEFERRAL_RATE': [0.05, 0.08, 0.03]
    })
    
    print(f"Original columns: {list(legacy_df.columns)}")
    
    migrated_df, result = migrate_legacy_columns(legacy_df, schema_type="snapshot")
    
    print(f"Migration successful: {result.success}")
    print(f"Migrated columns: {result.migrated_columns}")
    print(f"New columns: {list(migrated_df.columns)}")
    
    # Verify key columns were migrated
    assert SnapshotColumns.EMP_ID in migrated_df.columns
    assert SnapshotColumns.EMP_HIRE_DATE in migrated_df.columns
    assert SnapshotColumns.EMP_GROSS_COMP in migrated_df.columns
    
    print("✅ Schema migration tests passed!\n")


def test_schema_validation():
    """Test schema validation functionality."""
    print("=== Testing Schema Validation ===")
    
    # Create a valid snapshot
    valid_df = pd.DataFrame({
        SnapshotColumns.EMP_ID: ['EMP001', 'EMP002', 'EMP003'],
        SnapshotColumns.EMP_HIRE_DATE: ['2020-01-01', '2021-01-01', '2019-06-15'],
        SnapshotColumns.EMP_GROSS_COMP: [75000, 85000, 65000],
        SnapshotColumns.EMP_ACTIVE: [True, True, False],
        SnapshotColumns.SIMULATION_YEAR: [2024, 2024, 2024]
    })
    
    validation_result = validate_snapshot_schema(valid_df)
    print(f"Valid snapshot validation: {validation_result.is_valid}")
    print(f"Errors: {validation_result.errors}")
    print(f"Warnings: {validation_result.warnings}")
    
    # Create an invalid snapshot (missing required columns)
    invalid_df = pd.DataFrame({
        SnapshotColumns.EMP_ID: ['EMP001', 'EMP002'],
        # Missing required columns
    })
    
    validation_result = validate_snapshot_schema(invalid_df)
    print(f"Invalid snapshot validation: {validation_result.is_valid}")
    print(f"Errors: {validation_result.errors}")
    
    assert not validation_result.is_valid, "Invalid snapshot should fail validation"
    
    print("✅ Schema validation tests passed!\n")


def test_transformer_with_unified_schema():
    """Test that transformers work with the unified schema."""
    print("=== Testing Transformers with Unified Schema ===")
    
    # Create test data with legacy column names
    legacy_df = pd.DataFrame({
        'EMP_ID': ['EMP001', 'EMP002', 'EMP003'],
        'EMP_HIRE_DATE': ['2020-01-01', '2021-01-01', '2019-06-15'],
        'EMP_BIRTH_DATE': ['1990-05-15', '1985-12-20', '1992-03-10'],
        'EMP_GROSS_COMP': [75000, 85000, 65000],
        'EMP_ACTIVE': [True, True, False],
        'EMP_DEFERRAL_RATE': [0.05, 0.08, 0.03]
    })
    
    print(f"Original legacy columns: {list(legacy_df.columns)}")
    
    # Initialize transformer
    config = SnapshotConfig(start_year=2024)
    transformer = SnapshotTransformer(config)
    
    # Test tenure calculation
    df_with_tenure = transformer.apply_tenure_calculations(legacy_df.copy())
    print(f"Columns after tenure calculation: {list(df_with_tenure.columns)}")
    assert SnapshotColumns.EMP_TENURE in df_with_tenure.columns
    assert SnapshotColumns.EMP_TENURE_BAND in df_with_tenure.columns
    
    # Test age calculation
    df_with_age = transformer.apply_age_calculations(legacy_df.copy())
    print(f"Columns after age calculation: {list(df_with_age.columns)}")
    assert SnapshotColumns.EMP_AGE in df_with_age.columns
    assert SnapshotColumns.EMP_AGE_BAND in df_with_age.columns
    
    # Test contribution calculation
    df_with_contrib = transformer.apply_contribution_calculations(legacy_df.copy())
    print(f"Columns after contribution calculation: {list(df_with_contrib.columns)}")
    assert SnapshotColumns.EMP_CONTRIBUTION in df_with_contrib.columns
    assert SnapshotColumns.EMPLOYER_MATCH_CONTRIBUTION in df_with_contrib.columns
    
    # Test job level inference
    df_with_levels = transformer.infer_job_levels(legacy_df.copy())
    print(f"Columns after job level inference: {list(df_with_levels.columns)}")
    assert SnapshotColumns.EMP_LEVEL in df_with_levels.columns
    assert SnapshotColumns.EMP_LEVEL_SOURCE in df_with_levels.columns
    
    print("✅ Transformer tests with unified schema passed!\n")


def main():
    """Run all tests."""
    print("🚀 Testing Unified Schema System\n")
    
    try:
        test_schema_detection()
        test_schema_migration()
        test_schema_validation()
        test_transformer_with_unified_schema()
        
        print("🎉 All tests passed! Unified schema system is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()