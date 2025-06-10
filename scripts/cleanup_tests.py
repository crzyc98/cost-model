#!/usr/bin/env python3
"""
Test cleanup script to identify and remove outdated tests.

This script:
1. Checks import paths in each test file
2. Verifies that the modules being tested still exist
3. Lists tests that should be removed or updated
"""

import os
import re
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Patterns to identify outdated imports
OUTDATED_IMPORT_PATTERNS = [
    r"from utils\.rules\.",  # Old utils.rules pattern
    r"from utils\.",  # Old utils pattern
    r"import utils\.",  # Old utils import
]

# Known outdated test files that should be removed
KNOWN_OUTDATED_FILES = [
    "tests/debug_elig.py",
    "tests/test_contributions.py",
    "tests/test_eligibility.py",
    "tests/test_contributions_mixin.py",
]


def check_file_for_outdated_imports(file_path):
    """Check if a file contains outdated import patterns."""
    try:
        with open(file_path, "r") as f:
            content = f.read()

        for pattern in OUTDATED_IMPORT_PATTERNS:
            if re.search(pattern, content):
                return True
        return False
    except Exception as e:
        print(f"Error checking {file_path}: {e}")
        return False


def find_test_files(directory=None):
    """Find all test files in the given directory."""
    if directory is None:
        directory = PROJECT_ROOT / "tests"

    test_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                test_files.append(os.path.join(root, file))
    return test_files


def main():
    """Main function to identify outdated tests."""
    print("Checking for outdated test files...")

    # Check if known outdated files exist and should be removed
    for file_path in KNOWN_OUTDATED_FILES:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            print(f"✗ {file_path} - Known outdated file, should be removed")

    # Find all test files
    test_files = find_test_files()

    # Check each test file for outdated imports
    outdated_files = []
    for file_path in test_files:
        rel_path = os.path.relpath(file_path, PROJECT_ROOT)
        if check_file_for_outdated_imports(file_path):
            print(f"✗ {rel_path} - Contains outdated imports, should be updated")
            outdated_files.append(rel_path)

    # Print summary
    print("\nSummary:")
    print(f"- Found {len(test_files)} test files")
    print(f"- {len(outdated_files)} files contain outdated imports")

    # Print instructions
    print("\nTo remove known outdated files, run:")
    for file_path in KNOWN_OUTDATED_FILES:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            print(f"rm {file_path}")

    print("\nTo update files with outdated imports, you need to:")
    print("1. Open each file")
    print("2. Replace 'from utils.' with 'from cost_model.utils.'")
    print("3. Replace other outdated imports with their current equivalents")


if __name__ == "__main__":
    main()
