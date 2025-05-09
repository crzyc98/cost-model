import pandas as pd
import pytest

from utils.sampling.salary import DefaultSalarySampler


def test_default_second_year_bump():
    df = pd.DataFrame({"tenure": [1, 2], "gross_compensation": [100, 200]})
    sampler = DefaultSalarySampler()
    out = sampler.sample_second_year(df, "gross_compensation", {}, rate=0.1, seed=42)
    assert out.iloc[0] == pytest.approx(110)
    assert out.iloc[1] == pytest.approx(200)


def test_default_termination_sampling():
    prev = pd.Series([100, 200, 300])
    sampler = DefaultSalarySampler()
    draws = sampler.sample_terminations(prev, size=5, seed=42)
    assert isinstance(draws, pd.Series)
    assert len(draws) == 5
    for v in draws:
        assert v in prev.values


def test_sample_terminations_size_zero():
    prev = pd.Series([], dtype=float)
    sampler = DefaultSalarySampler()
    draws = sampler.sample_terminations(prev, size=0, seed=42)
    assert draws.empty
