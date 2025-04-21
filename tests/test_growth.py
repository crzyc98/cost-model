import pandas as pd
import os
from decimal import Decimal
import statistics
import pytest

from model.retirement_model import RetirementPlanModel


def simulate_growth(N0, runs=200, growth_rate=0.02, h_ex=0.13, h_nh=0.2, use_expected=False):
    # Create initial census
    df = pd.DataFrame({
        'ssn': [str(i) for i in range(N0)],
        'birth_date': pd.NaT,
        'hire_date': pd.NaT,
        'termination_date': pd.NaT,
        'gross_compensation': [Decimal('50000')] * N0,
        'deferral_rate': [Decimal('0.0')] * N0
    })
    config = {
        'start_year': 2025,
        'projection_years': 1,
        'annual_growth_rate': growth_rate,
        'annual_termination_rate': h_ex,
        'new_hire_termination_rate': h_nh,
        'use_expected_attrition': use_expected
    }
    growths = []
    for _ in range(runs):
        model = RetirementPlanModel(initial_census_df=df.copy(), scenario_config=config)
        model.step()
        N1 = len(model.schedule.agents)
        growths.append((N1 - N0) / N0)
    return statistics.mean(growths)


def test_average_growth_close_to_target_realized():
    avg = simulate_growth(N0=1000, runs=200, use_expected=False)
    assert abs(avg - 0.02) < 0.005, f"Realized attrition avg growth {avg:.4f} not within 0.5% of target"


def test_average_growth_close_to_target_expected():
    avg = simulate_growth(N0=1000, runs=200, use_expected=True)
    assert abs(avg - 0.02) < 0.001, f"Expected attrition avg growth {avg:.4f} not within 0.1% of target"
