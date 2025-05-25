import pandas as pd
from cost_model.state.job_levels.sampling import load_markov_matrix

def test_load_markov_matrix_dev_mode_identity_exit():
    STATES = [0, 1, 2, 3, 4, 'exit']
    matrix = load_markov_matrix(None, allow_default=True)
    assert isinstance(matrix, pd.DataFrame)
    assert matrix.shape == (6, 6)
    assert list(matrix.index) == STATES
    assert list(matrix.columns) == STATES
    # Check identity+exit: diagonal 1.0 except last col is exit
    for i, state in enumerate(STATES[:-1]):
        # Each row except exit should have 1.0 on the diagonal or in the exit col
        assert (
            matrix.loc[state, state] == 1.0 or matrix.loc[state, 'exit'] == 1.0
        )
    # The exit row should be absorbing
    assert all(matrix.loc['exit'] == [0, 0, 0, 0, 0, 1])
# /tests/state/test_markov_matrix.py
import pytest
import pandas as pd
import numpy as np
import tempfile
import yaml
from cost_model.state.job_levels.sampling import load_markov_matrix

def test_default_matrix_valid():
    m = load_markov_matrix(None, allow_default=True)
    assert (m.values >= 0).all()
    assert m.shape[0] == m.shape[1]
    assert np.allclose(m.sum(axis=1), 1.0, atol=1e-6)

def test_default_matrix_usable_in_dev():
    from cost_model.state.job_levels.sampling import load_markov_matrix
    try:
        matrix = load_markov_matrix(None, allow_default=True)
    except Exception as e:
        assert False, f"Should not raise when allow_default=True: {e}"
    assert matrix is not None
    # Check identity property
    import numpy as np
    assert np.allclose(matrix, np.eye(matrix.shape[0]))

def test_missing_matrix_fails_in_prod():
    from cost_model.state.job_levels.sampling import load_markov_matrix
    import pytest
    with pytest.raises((FileNotFoundError, ValueError)):
        load_markov_matrix(None, allow_default=False)

def test_missing_matrix_raises():
    with pytest.raises(FileNotFoundError):
        load_markov_matrix("nonexistent.yaml")

def test_malformed_matrix_raises():
    # Create a non-square YAML matrix
    bad_matrix = [[1.0, 0.0], [0.5, 0.5], [0.2, 0.8]]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(bad_matrix, tmp)
        tmp_path = tmp.name
    try:
        with pytest.raises(ValueError):
            load_markov_matrix(tmp_path)
    finally:
        import os
        os.remove(tmp_path)
