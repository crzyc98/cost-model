import pandas as pd
import numpy as np
import yaml
import types
import pytest
from pathlib import Path

from cost_model.engines.run_one_year import run_one_year
from cost_model.utils.columns import (
    EMP_ID,
    EMP_LEVEL,
    EMP_GROSS_COMP,
    EMP_ROLE,
    EMP_HIRE_DATE,
    EMP_TERM_DATE,
    EMP_BIRTH_DATE,
    EMP_DEFERRAL_RATE,
    EMP_TENURE,
)
from cost_model.state.event_log import EVENT_COLS

@pytest.fixture
def global_params():
    cfg = yaml.safe_load(Path("config/compensation.yaml").read_text())
    return types.SimpleNamespace(
        compensation=types.SimpleNamespace(**cfg),
        days_into_year_for_promotion=0,
        days_into_year_for_cola=0,
        new_hires=types.SimpleNamespace(new_hire_rate=0.0),
        attrition=types.SimpleNamespace(new_hire_termination_rate=0.0),
        promotion_rules={}
    )

@pytest.fixture
def snapshot_df():
    return pd.DataFrame({
        EMP_ID: ["1", "2", "3"],
        EMP_HIRE_DATE: [pd.Timestamp("2024-01-01")]*3,
        EMP_BIRTH_DATE: [pd.Timestamp("1990-01-01")]*3,
        EMP_ROLE: ["all"]*3,
        EMP_LEVEL: [0, 1, 2],
        EMP_GROSS_COMP: [50_000.0, 80_000.0, 120_000.0],
        EMP_TERM_DATE: [pd.NaT]*3,
        "active": [True]*3,
        EMP_DEFERRAL_RATE: [0.0]*3,
        "tenure_band": ["0-1"]*3,
        EMP_TENURE: [0.0]*3,
        "job_level_source": ["markov-promo", "markov-promo", "salary-band"],
        "exited": [False]*3,
        "eligible_for_promotion": [True]*3
    })

@pytest.fixture
def hazard_df(global_params):
    return pd.DataFrame([{
        "simulation_year": 2025,
        EMP_ROLE: "all",
        "tenure_band": "0-1",
        "term_rate": 0.0,
        "comp_raise_pct": 0.0,
        "new_hire_termination_rate": 0.0,
        "cola_pct": 0.0,
        "cfg": global_params,
    }])

def test_run_one_year_smoke(global_params, snapshot_df, hazard_df, tmp_path):
    # ensure census template exists (could be an empty parquet)
    census = tmp_path / "census_template.parquet"
    snapshot_df.to_parquet(census)

    events, snap = run_one_year(
        event_log=pd.DataFrame([], columns=EVENT_COLS),
        prev_snapshot=snapshot_df,
        year=2025,
        global_params=global_params,
        plan_rules=types.SimpleNamespace(),
        hazard_table=hazard_df,
        rng=np.random.default_rng(0),
        census_template_path=str(census),
    )

    # must produce some raises
    raises = events[events.event_type == "EVT_RAISE"]
    assert not raises.empty

    # final salaries should all be > starting
    assert (snap[EMP_GROSS_COMP] > [50_000, 80_000, 120_000]).all()