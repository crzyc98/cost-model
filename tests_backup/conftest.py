import os
import sys
import pytest

# Ensure project root is on sys.path before imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Define pytest markers for test categories
def pytest_configure(config):
    """
    Register custom markers to avoid pytest warnings.
    """
    config.addinivalue_line("markers", "unit: mark a test as a unit test")
    config.addinivalue_line("markers", "integration: mark a test as an integration test")
    config.addinivalue_line("markers", "abm: mark a test as an agent-based model test")
    config.addinivalue_line("markers", "quick: mark a test as a quick test for CI")
    config.addinivalue_line("markers", "slow: mark a test as a slow test")
    config.addinivalue_line("markers", "config: mark a test as a config test")
    config.addinivalue_line("markers", "plan_rules: mark a test as a plan rules test")
    config.addinivalue_line("markers", "engines: mark a test as an engines test")
    config.addinivalue_line("markers", "dynamics: mark a test as a dynamics test")
    config.addinivalue_line("markers", "state: mark a test as a state test")
    config.addinivalue_line("markers", "utils: mark a test as a utils test")
