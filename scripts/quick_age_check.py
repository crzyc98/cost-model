#!/usr/bin/env python3
"""Quick script to check age data in test results."""

import pandas as pd
from pathlib import Path

def check_snapshot(snapshot_path):
    """Check a single snapshot for age data."""
    try:
        df = pd.read_parquet(snapshot_path)
        print(f"\n📊 Snapshot: {snapshot_path}")
        print(f"   Employees: {len(df)}")
        print(f"   Columns: {len(df.columns)}")
        
        # Check for age columns
        age_cols = [col for col in df.columns if 'age' in col.lower()]
        print(f"   Age columns: {age_cols}")
        
        if age_cols:
            for col in age_cols:
                non_null = df[col].dropna()
                print(f"   {col}: {len(non_null)} non-null values")
                if len(non_null) > 0:
                    if col.endswith('_age'):
                        print(f"      Range: {non_null.min():.1f} - {non_null.max():.1f}")
                        print(f"      Mean: {non_null.mean():.1f}")
                    else:
                        print(f"      Unique values: {sorted(non_null.unique())}")
        else:
            print("   ❌ No age columns found!")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

def main():
    """Check age data in all test scenarios."""
    print("🔍 Checking Age Data in Test Results")
    
    # Test scenarios
    scenarios = [
        ("output_age_test_baseline", "Baseline"),
        ("output_age_test_high_early_attrition", "High Early Attrition"),
        ("output_age_test_high_late_attrition", "High Late Attrition")
    ]
    
    for output_dir, scenario_name in scenarios:
        print(f"\n{'='*60}")
        print(f"📁 {scenario_name}")
        print(f"{'='*60}")
        
        # Find scenario directory
        output_path = Path(output_dir)
        if not output_path.exists():
            print(f"❌ Directory not found: {output_dir}")
            continue
            
        scenario_dirs = [d for d in output_path.iterdir() if d.is_dir()]
        if not scenario_dirs:
            print(f"❌ No scenario directories found")
            continue
            
        scenario_dir = scenario_dirs[0]
        print(f"📂 Scenario directory: {scenario_dir.name}")
        
        # Check yearly snapshots
        year_dirs = [d for d in scenario_dir.iterdir() if d.is_dir() and d.name.startswith('year=')]
        
        if not year_dirs:
            print(f"❌ No yearly directories found")
            continue
            
        for year_dir in sorted(year_dirs):
            snapshot_path = year_dir / "snapshot.parquet"
            if snapshot_path.exists():
                check_snapshot(snapshot_path)
            else:
                print(f"❌ No snapshot found in {year_dir}")

if __name__ == "__main__":
    main()
