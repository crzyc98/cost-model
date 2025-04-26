import pandas as pd
from types import SimpleNamespace
from agents.eligibility import EligibilityMixin

def test_custom_checker_overrides_default():
    class A:
        birth_date = pd.Timestamp('2000-01-01')
        hire_date = pd.Timestamp('2020-01-01')
    a = EligibilityMixin()
    a.model = SimpleNamespace(year=2025, scenario_config={'plan_rules': {'eligibility': {'custom_checker': lambda ag, date: True}}})
    a.is_eligible = False
    a.__class__ = type('Agent', (EligibilityMixin, A), {})  # Compose EligibilityMixin and A
    a._update_eligibility()
    assert a.is_eligible is True
