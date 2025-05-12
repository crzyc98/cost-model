import pandas as pd
from decimal import Decimal

from agents.employee_agent import EmployeeAgent


class DummyModel:
    def __init__(self):
        self.start_year = 2025
        self.year = 2025
        self.scenario_config = {"plan_rules": {}}
        import random

        self.random = random.Random(0)


def test_employee_agent_smoke():
    initial_state = {
        "birth_date": pd.Timestamp("1980-06-01"),
        "hire_date": pd.Timestamp("2022-03-15"),
        "gross_compensation": 75000,
    }
    agent = EmployeeAgent(1, DummyModel(), initial_state)
    # Verify core attributes
    assert isinstance(agent.birth_date, pd.Timestamp)
    assert agent.gross_compensation == Decimal("75000.00")
    # Ensure step runs without errors
    agent.step()
