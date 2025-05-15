import pytest
import pandas as pd
from cost_model.state.job_levels.intervals import build_intervals, check_for_overlapping_bands
from cost_model.state.job_levels.models import JobLevel, ConfigError

@pytest.fixture
def simple_levels():
    return {
        0: JobLevel(0, "A", "", 0, 100, 0,0,0),
        1: JobLevel(1, "B", "", 101, 200, 0,0,0),
    }

def test_build_intervals(simple_levels):
    idx = build_intervals(simple_levels)
    assert isinstance(idx, pd.IntervalIndex)
    assert len(idx) == 2
    assert idx[0].left == 0 and idx[0].right == 100

def test_overlapping_detect_strict(simple_levels):
    # create overlap
    overlapping = {
        0: JobLevel(0, "A", "", 0, 150, 0,0,0),
        1: JobLevel(1, "B", "", 100, 200, 0,0,0),
    }
    with pytest.raises(ConfigError):
        check_for_overlapping_bands(overlapping, strict=True)

def test_overlapping_non_strict(simple_levels):
    overlapping = {
        0: JobLevel(0, "A", "", 0, 150, 0,0,0),
        1: JobLevel(1, "B", "", 100, 200, 0,0,0),
    }
    overlaps = check_for_overlapping_bands(overlapping, strict=False)
    assert len(overlaps) == 1