import pandas as pd
import types
from utils.rules.eligibility import agent_is_eligible
from agents.eligibility import EligibilityMixin


class DummyModel:
    def __init__(self, year, plan_rules):
        self.year = year
        self.scenario_config = {"plan_rules": plan_rules}


class DummyAgent(EligibilityMixin, types.SimpleNamespace):
    def __init__(self, birth_date, hire_date, status, hours_worked, model):
        super().__init__(
            birth_date=pd.Timestamp(birth_date),
            hire_date=pd.Timestamp(hire_date),
            status=status,
            hours_worked=hours_worked,
            model=model,
            is_eligible=False,
        )

    def _update_eligibility(self):
        EligibilityMixin._update_eligibility(self)


def test_agent_eligibility_matches_helper():
    sim_end = pd.Timestamp("2025-12-31")
    eligibility_config = {
        "min_age": 21,
        "min_service_months": 12,
        "min_hours_worked": 40,
    }
    plan_rules = {"eligibility": eligibility_config}
    model = DummyModel(2025, plan_rules)

    cases = [
        # All good
        ("2000-01-01", "2024-11-30", "Active", 40, True),
        # Low hours
        ("2000-01-01", "2024-01-01", "Active", 39, False),
        # Wrong status
        ("2000-01-01", "2024-01-01", "Terminated", 40, False),
        # Underage
        ("2006-01-01", "2024-01-01", "Active", 40, False),
    ]
    for b, h, s, hrs, expected in cases:
        agent = DummyAgent(b, h, s, hrs, model)
        agent._update_eligibility()
        # Compare to helper
        helper_result = agent_is_eligible(
            pd.Timestamp(b), pd.Timestamp(h), s, hrs, eligibility_config, sim_end
        )
        assert agent.is_eligible == helper_result == expected
