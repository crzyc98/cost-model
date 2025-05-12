#!/usr/bin/env python3
"""
Script to update import paths in test files from old 'utils.' to 'cost_model.utils.'
"""

import os
import re
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent

# Patterns to replace
REPLACEMENTS = [
    (r'from utils\.', 'from cost_model.utils.'),
    (r'import utils\.', 'import cost_model.utils.'),
    (r'from utils import', 'from cost_model.utils import'),
]

def update_imports_in_file(file_path):
    """Update import paths in a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        for pattern, replacement in REPLACEMENTS:
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
    print("Updating import paths in test files...")
    
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

if __name__ == "__main__":
    main()
