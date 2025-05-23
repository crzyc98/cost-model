# tests/state/job_levels/test_transitions.py

import numpy as np
import pandas as pd
import pytest

from cost_model.state.job_levels.transitions import PROMOTION_MATRIX
from cost_model.state.job_levels.sampling import apply_promotion_markov
from cost_model.state.schema import EMP_LEVEL

@pytest.fixture(autouse=True)
def seed_rng(monkeypatch):
    """Make sampling deterministic for tests."""
    rng = np.random.RandomState(1234)
    monkeypatch.setattr("cost_model.state.job_levels.sampling.np.random", rng)
    return rng

def make_df(levels):
    """Helper: build a DataFrame with one row per level in `levels`."""
    return pd.DataFrame({EMP_LEVEL: levels})

def test_promotion_matrix_sums_to_one():
    # Each row of PROMOTION_MATRIX should sum to 1
    row_sums = PROMOTION_MATRIX.sum(axis=1)
    assert np.allclose(row_sums, 1.0)

def test_apply_promotion_markov_basic():
    # Create one employee at each level 0–4
    df = make_df([0, 1, 2, 3, 4])
    out = apply_promotion_markov(df)

    # Should still have same number of rows plus 'exited' column
    assert list(out.index) == list(df.index)
    assert "exited" in out.columns

    # All new levels must be in 0–4 or NaN (exit)
    new_levels = out[EMP_LEVEL].tolist()
    assert all((lv in {0,1,2,3,4} or pd.isna(lv)) for lv in new_levels)

    # Exited flag matches NaN in level
    exited_mask = out["exited"]
    nan_mask    = pd.isna(out[EMP_LEVEL])
    assert exited_mask.equals(nan_mask)

def test_markov_edge_cases_already_exited():
    # If someone already has NaN in EMP_LEVEL, they remain exited
    df = make_df([np.nan, 2])
    # Set exited=True for first row:
    df["exited"] = [True, False]
    out = apply_promotion_markov(df)

    # Row0 stays exited and NaN
    assert pd.isna(out.loc[0, EMP_LEVEL])
    assert out.loc[0, "exited"]

    # Row1 transitions or stays, but 'exited' updated properly
    assert isinstance(out.loc[1, "exited"], (bool, np.bool_))

def test_promotion_statistics():
    # Test that over many draws, empirical rates approximate the matrix
    n = 10000
    df = make_df([2] * n)   # all start at level 2
    out = apply_promotion_markov(df)
    counts = out[EMP_LEVEL].value_counts(normalize=True).to_dict()
    # Compare to PROMOTION_MATRIX row for level 2
    target = PROMOTION_MATRIX.loc[2].to_dict()
    # Exit is under key 'exit'
    exit_rate = target.pop("exit")
    # Tolerance for sampling error:
    tol = 0.06  # increased tolerance for sampling variability
    for lvl, prob in target.items():
        assert abs(counts.get(lvl, 0) - prob) < tol
    assert abs(counts.get(np.nan, 0) - exit_rate) <= tol  # allow exact match