#!/usr/bin/env python3
"""
Script to fix import paths in test files to match the current codebase structure.
"""

import os
import re
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent

# Mapping of old import paths to new import paths
IMPORT_MAPPINGS = [
    # Utils rules to plan_rules
    (r'from (?:cost_model\.)?utils\.rules\.auto_enrollment import', 'from cost_model.plan_rules.auto_enrollment import'),
    (r'from (?:cost_model\.)?utils\.rules\.auto_increase import', 'from cost_model.plan_rules.auto_increase import'),
    (r'from (?:cost_model\.)?utils\.rules\.eligibility import', 'from cost_model.plan_rules.eligibility import'),
    (r'from (?:cost_model\.)?utils\.rules\.formula_parsers import', 'from cost_model.plan_rules.formula_parsers import'),
    (r'from (?:cost_model\.)?utils\.rules\.contributions import', 'from cost_model.rules.contributions import'),
    (r'from (?:cost_model\.)?utils\.rules\.validators import', 'from cost_model.plan_rules.validators import'),
    
    # Utils to cost_model.utils
    (r'from utils\.', 'from cost_model.utils.'),
    (r'import utils\.', 'import cost_model.utils.'),
    (r'from utils import', 'from cost_model.utils import'),
    
    # Specific module moves
    (r'from (?:cost_model\.)?utils\.projection_utils import', 'from cost_model.projections.utils import'),
    (r'from (?:cost_model\.)?utils\.constants import', 'from cost_model.utils.constants import'),
    (r'from (?:cost_model\.)?utils\.sampling\.salary import', 'from cost_model.dynamics.compensation import'),
]

def update_imports_in_file(file_path):
    """Update import paths in a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        for pattern, replacement in IMPORT_MAPPINGS:
            content = re.sub(pattern, replacement, content)
        
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def find_test_files(directory=None):
    """Find all test files in the given directory."""
    if directory is None:
        directory = PROJECT_ROOT / "tests"
    
    test_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                test_files.append(os.path.join(root, file))
    return test_files

def main():
    """Main function to update import paths in test files."""
    print("Fixing import paths in test files...")
    
    # Find all test files
    test_files = find_test_files()
    
    # Update import paths in each test file
    updated_files = 0
    for file_path in test_files:
        rel_path = os.path.relpath(file_path, PROJECT_ROOT)
        if update_imports_in_file(file_path):
            print(f"✓ Updated {rel_path}")
            updated_files += 1
        else:
            print(f"- No changes needed in {rel_path}")
    
    # Print summary
    print("\nSummary:")
    print(f"- Found {len(test_files)} test files")
    print(f"- Updated {updated_files} files")
    
    print("\nNote: Some tests may still fail if they're testing functionality that has been")
    print("significantly refactored or removed. You may need to update or remove these tests manually.")

if __name__ == "__main__":
    main()
