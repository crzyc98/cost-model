import pytest
import pandas as pd
import numpy as np
import math
from types import SimpleNamespace
from pathlib import Path

from cost_model.engines.run_one_year import run_one_year
from cost_model.utils.columns import (
    EMP_ID, EMP_LEVEL, EMP_GROSS_COMP,
    EMP_HIRE_DATE, EMP_BIRTH_DATE, EMP_ROLE,
    EMP_TERM_DATE, EMP_DEFERRAL_RATE, EMP_TENURE
)
from cost_model.state.event_log import EVENT_COLS
from cost_model.state.job_levels.init import get_level_by_id

@pytest.fixture
def snapshot_df():
    return pd.DataFrame({
        EMP_ID: ["1", "2", "3"],
        EMP_HIRE_DATE: [pd.Timestamp("2024-01-01")] * 3,
        EMP_BIRTH_DATE: [pd.Timestamp("1990-01-01")] * 3,
        EMP_ROLE: ["all"] * 3,
        EMP_LEVEL: [0, 1, 2],
        EMP_GROSS_COMP: [50_000.0, 80_000.0, 120_000.0],
        EMP_TERM_DATE: [pd.NaT] * 3,
        "active": [True] * 3,
        EMP_DEFERRAL_RATE: [0.0] * 3,
        "tenure_band": ["0-1"] * 3,
        EMP_TENURE: [0.0] * 3,
        "job_level_source": ["markov-promo"] * 3,
        "exited": [False] * 3,
    })

@pytest.fixture
def global_params():
    return SimpleNamespace(
        compensation=SimpleNamespace(
            COLA_rate=0.0,
            promo_raise_pct={},
            merit_dist={}
        ),
        days_into_year_for_cola=0,
        days_into_year_for_promotion=0,
        new_hires=SimpleNamespace(new_hire_rate=1/3),
        attrition=SimpleNamespace(new_hire_termination_rate=0.0),
        promotion_rules={},
    )

@pytest.fixture
def hazard_df():
    return pd.DataFrame([{  # minimal hazard
        "simulation_year": 2025,
        "role": "all",
        "tenure_band": "0-1",
        "term_rate": 0.0,
        "comp_raise_pct": 0.0,
        "new_hire_termination_rate": 0.0,
        "cola_pct": 0.0,
        "cfg": SimpleNamespace()
    }])

@pytest.fixture
def census_template():
    return pd.DataFrame({
        EMP_ID: ["template"],
        EMP_HIRE_DATE: [pd.Timestamp("2024-01-01")],
        EMP_BIRTH_DATE: [pd.Timestamp("1990-01-01")],
        EMP_ROLE: ["all"],
        EMP_GROSS_COMP: [50_000.0],
        EMP_TERM_DATE: [pd.NaT],
        "active": [True],
        EMP_DEFERRAL_RATE: [0.0],
        "exited": [False]
    })

@pytest.fixture
def tmp_census_template(tmp_path, census_template):
    path = tmp_path / "census_template.parquet"
    census_template.to_parquet(path)
    return str(path)


def test_run_one_year_hiring(snapshot_df, global_params, hazard_df, tmp_census_template):
    n0 = len(snapshot_df)
    events, snap = run_one_year(
        event_log=pd.DataFrame([], columns=EVENT_COLS),
        prev_snapshot=snapshot_df,
        year=2025,
        global_params=global_params,
        plan_rules=SimpleNamespace(),
        hazard_table=hazard_df,
        rng=np.random.default_rng(0),
        census_template_path=tmp_census_template
    )
    # expected new hires
    expected = int(math.ceil(n0 * global_params.new_hires.new_hire_rate))
    n1 = len(snap)
    assert n1 == n0 + expected

    # identify new hires
    original_ids = set(snapshot_df[EMP_ID])
    new_ids = set(snap[EMP_ID]) - original_ids
    assert len(new_ids) == expected

    hires = snap[snap[EMP_ID].isin(new_ids)]
    # all new hires tagged correctly
    assert all(hires["job_level_source"] == "hire")

    # levels are in valid range
    assert all(lvl in range(0, 5) for lvl in hires[EMP_LEVEL])

    # compensation within bounds per level
    for _, r in hires.iterrows():
        lvl = r[EMP_LEVEL]
        comp = r[EMP_GROSS_COMP]
        level = get_level_by_id(lvl)
        assert level.min_compensation <= comp <= level.max_compensation
