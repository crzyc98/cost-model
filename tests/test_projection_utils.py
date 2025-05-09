import pandas as pd
import numpy as np
import pytest
from utils.projection_utils import _apply_comp_bump, _apply_turnover


class DummySampler:
    def sample_second_year(self, df, comp_col, dist, rate, rng):
        # double the compensation for testing
        return df[comp_col] * 2

    def sample_terminations(self, prev_salaries, size, rng):
        # return constant salary draws
        return np.full(size, 1000)


# Helper to stub sample_terminations behavior
def fake_sample_terminations(df, hire_col, probs_or_rate, end_date, rng):
    # Copy df and set one termination_date within window for index 0
    df2 = df.copy()
    df2["termination_date"] = pd.Timestamp(end_date)
    return df2


@pytest.fixture
def dummy_df():
    return pd.DataFrame(
        {
            "gross_compensation": [100, 200, 300],
            "tenure": [1, 0, 2],
        }
    )


def test_apply_comp_bump_sampler_and_standard(dummy_df):
    # tenure==1 rows doubled, others increased by 10%
    rate = 0.10
    sampler = DummySampler()
    rng = np.random.default_rng(42)
    result = _apply_comp_bump(dummy_df, "gross_compensation", {}, rate, rng, sampler)

    # index 0 and 2, tenure==1? only index 0 has tenure 1
    # index 0: 100 * 2 = 200
    # index 1 (tenure 0) unchanged; index 2: standard bump 300*1.1=330
    expected = [200, 200, 330]
    assert np.allclose(result["gross_compensation"], expected)


def test_apply_comp_bump_missing_column(dummy_df):
    sampler = DummySampler()
    rng = np.random.default_rng(0)
    # use non-existent column
    result = _apply_comp_bump(dummy_df, "salary", {}, 0.05, rng, sampler)
    # original df should be returned unchanged
    assert "salary" not in result.columns
    assert result.equals(dummy_df)


def test_apply_turnover_no_terminations(monkeypatch):
    # Build df with no initial termination_date
    df = pd.DataFrame(
        {
            "gross_compensation": [100, 200],
            "hire_date": [pd.Timestamp("2020-01-01")] * 2,
            "termination_date": [pd.NaT, pd.NaT],
        }
    )

    # stub sample_terminations to not change termination_date
    def no_term(df_, hire_col, probs_or_rate, end_date, rng):
        return df_.copy()

    monkeypatch.setattr("utils.projection_utils.sample_terminations", no_term)
    sampler = DummySampler()
    rng = np.random.default_rng(0)
    prev_salaries = np.array([100, 200])
    out = _apply_turnover(
        df,
        "hire_date",
        0.5,
        pd.Timestamp("2021-01-01"),
        pd.Timestamp("2021-12-31"),
        rng,
        sampler,
        prev_salaries,
    )
    # no changes
    assert out["gross_compensation"].tolist() == [100, 200]


def test_apply_turnover_with_terminations(monkeypatch):
    df = pd.DataFrame(
        {
            "gross_compensation": [100, 200],
            "hire_date": [pd.Timestamp("2020-01-01")] * 2,
            "termination_date": [pd.Timestamp("2021-06-01"), pd.NaT],
        }
    )

    # sample_terminations should pass through df unchanged
    def pass_through(df_, hire_col, probs_or_rate, end_date, rng):
        return df_.copy()

    monkeypatch.setattr("utils.projection_utils.sample_terminations", pass_through)
    sampler = DummySampler()
    rng = np.random.default_rng(1)
    prev_salaries = np.array([500])
    out = _apply_turnover(
        df,
        "hire_date",
        0.5,
        pd.Timestamp("2021-01-01"),
        pd.Timestamp("2021-12-31"),
        rng,
        sampler,
        prev_salaries,
    )
    # index 0 terminated, compensation replaced by 1000 from DummySampler
    assert out.loc[0, "gross_compensation"] == 1000
    # index 1 unchanged
    assert out.loc[1, "gross_compensation"] == 200


def test_comp_bump_groups():
    import pandas as pd
    import pytest
    from utils.sampling.salary import DefaultSalarySampler
    from utils.projection_utils import _apply_comp_bump

    df = pd.DataFrame({"tenure": [0, 1, 2], "gross": [100, 100, 100]})
    rng = np.random.default_rng(0)

    class StubSampler(DefaultSalarySampler):
        def sample_second_year(self, df, comp_col, **kwargs):
            return pd.Series(df[comp_col] + 5, index=df.index)

    out = _apply_comp_bump(
        df.copy(), "gross", {}, rate=0.1, rng=rng, sampler=StubSampler()
    )
    # tenure==0 unchanged
    assert out.loc[0, "gross"] == 100
    # tenure==1 stub bump +5
    assert out.loc[1, "gross"] == 105
    # tenure>=2 flat *1.1 → 110
    assert out.loc[2, "gross"] == pytest.approx(110)
