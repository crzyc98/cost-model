#!/usr/bin/env python3
"""
Test script to verify that baseline age distribution loading works correctly.
"""

import sys
from pathlib import Path

# Add project root to Python path (same as in tune_configs.py)
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the load_baseline_distributions function
sys.path.append(str(Path(__file__).parent))
from tune_configs import HAS_AGE_UTILS, HAS_PANDAS, load_baseline_distributions


def main():
    print("Testing baseline distribution loading...")
    print(f"HAS_PANDAS: {HAS_PANDAS}")
    print(f"HAS_AGE_UTILS: {HAS_AGE_UTILS}")
    print()

    # Test the baseline loading function
    try:
        age_dist, tenure_dist = load_baseline_distributions()

        print("=== RESULTS ===")
        print("Age distribution:")
        for band, pct in age_dist.items():
            print(f"  {band}: {pct:.3f}")

        print("\nTenure distribution:")
        for band, pct in tenure_dist.items():
            print(f"  {band}: {pct:.3f}")

        # Check if we're using actual census data or fallback
        if HAS_AGE_UTILS and HAS_PANDAS:
            census_path = project_root / "data/census_template.parquet"
            if census_path.exists():
                print(f"\n✅ SUCCESS: Should be using actual census data from {census_path}")
            else:
                print(f"\n⚠️  WARNING: Census file not found at {census_path}")
        else:
            print(f"\n⚠️  WARNING: Using fallback distribution due to missing dependencies")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
