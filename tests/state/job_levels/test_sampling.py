import numpy as np
import pandas as pd
import pytest
from cost_model.state.job_levels.sampling import (
    sample_new_hire_compensation,
    sample_new_hires_vectorized,
    sample_mixed_new_hires
)
from cost_model.state.job_levels.models import JobLevel

@pytest.fixture
def test_level():
    return JobLevel(
        level_id=1, name="Stub", description="",
        min_compensation=50, max_compensation=150,
        comp_base_salary=100, comp_age_factor=0.01, comp_stochastic_std_dev=0.0
    )

def test_sample_new_hire_reproducible(test_level):
    rng = np.random.RandomState(42)
    c1 = sample_new_hire_compensation(test_level, age=30, random_state=rng)
    rng2 = np.random.RandomState(42)
    c2 = sample_new_hire_compensation(test_level, age=30, random_state=rng2)
    assert c1 == pytest.approx(c2)

def test_vectorized_sampling_matches_single(test_level):
    rng = np.random.RandomState(1)
    ages = np.array([25, 35, 45])
    vec = sample_new_hires_vectorized(test_level, ages, random_state=rng)
    singles = [sample_new_hire_compensation(test_level, age=a, random_state=rng) for a in ages]
    assert np.allclose(vec, singles)

def test_sample_mixed_new_hires(tmp_path, monkeypatch):
    # simple two-level sampler
    counts = {1: 3, 2: 2}
    df = sample_mixed_new_hires(counts, age_range=(20,30), random_state=np.random.RandomState(0))
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"level_id", "age", "compensation"}
    assert len(df) == 5