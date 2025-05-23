import pandas as pd
import numpy as np
import pytest

from cost_model.engines.compensation import update_salary
from cost_model.state.schema import EMP_GROSS_COMP, EMP_LEVEL
from cost_model.state.event_log import EVT_RAISE

@pytest.fixture
def basic_snapshot():
    return pd.DataFrame({
        EMP_LEVEL: [0, 1],
        EMP_GROSS_COMP: [100.0, 200.0],
        "job_level_source": [None, None]
    })


def test_cola_only(basic_snapshot):
    params = {
        "COLA_rate": 0.10,
        "promo_raise_pct": {},
        "merit_dist": {}
    }
    rng = np.random.default_rng(42)
    events = update_salary(basic_snapshot, params, rng)
    # Both salaries should rise by 10%
    assert basic_snapshot[EMP_GROSS_COMP].tolist() == [110.0, 220.0]
    # Two EVT_RAISE events, with value_num matching increment
    assert len(events) == 2
    assert set(events.value_num.tolist()) == {10.0, 20.0}


def test_promo_and_merit(basic_snapshot):
    basic_snapshot.loc[0, "job_level_source"] = "markov-promo"
    basic_snapshot.loc[1, "job_level_source"] = "salary-band"
    params = {
        "COLA_rate": 0.0,
        "promo_raise_pct": {"0_to_1": 0.05},
        "merit_dist": {
            0: {"mu": 0.10, "sigma": 0.0},
            1: {"mu": 0.02, "sigma": 0.0}
        }
    }
    rng = np.random.default_rng(0)
    events = update_salary(basic_snapshot, params, rng)
    # First row: promo 5% then merit 10% = +15 → 115
    assert pytest.approx(basic_snapshot.loc[0, EMP_GROSS_COMP]) == 115.0
    # Second row: only merit 2% → 200 * 1.02 = 204
    assert pytest.approx(basic_snapshot.loc[1, EMP_GROSS_COMP]) == 204.0
    # Check event types include EVT_RAISE
    assert EVT_RAISE in events.event_type.unique().tolist()
