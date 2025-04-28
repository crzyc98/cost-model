import pandas as pd
import pytest

from utils.projection_utils import project_census

# Dummy generator and sampler for new hires
def fake_generate_new_hires(num_hires, **kwargs):
    # Create num_hires rows with known comp of 100
    df = pd.DataFrame({
        'hire_date': [pd.Timestamp(f"{kwargs.get('hire_year')}-01-01")] * num_hires,
        'gross_compensation': [100.0] * num_hires
    })
    df['termination_date'] = pd.NaT
    return df

def identity_sample_new_hire(df, comp_col, arr, rng):
    # Return df unchanged
    return df


def test_onboarding_flat_rate(monkeypatch):
    # Initial census with one employee
    start_df = pd.DataFrame({
        'hire_date': [pd.Timestamp('2024-01-01')],
        'gross_compensation': [500.0],
        'termination_date': [pd.NaT]
    })
    # Config: hire_rate=1 to generate one new hire, no turnover
    config = {
        'scenario_name': 'onboarding_test',
        'start_year': 2025,
        'projection_years': 1,
        'comp_increase_rate': 0.1,
        'hire_rate': 1.0,
        'termination_rate': 0.0,
        'maintain_headcount': False,
        'plan_rules': {
            'onboarding_bump': {
                'enabled': True,
                'method': 'flat_rate',
                'flat_rate': 0.10
            }
        }
    }
    # Monkeypatch new-hire generation and sampling
    monkeypatch.setattr('utils.projection_utils.generate_new_hires', fake_generate_new_hires)
    monkeypatch.setattr('utils.projection_utils.sample_new_hire_compensation', identity_sample_new_hire)

    # Run projection
    projected = project_census(start_df, config, random_seed=42)[1]

    # New hires have tenure=0 and should be bumped by flat_rate
    nh = projected[projected['tenure'] == 0]
    assert len(nh) == 1
    assert nh['gross_compensation'].iloc[0] == pytest.approx(100.0 * 1.10)
