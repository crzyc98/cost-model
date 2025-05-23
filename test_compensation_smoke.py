import pandas as pd
import numpy as np
import yaml
import types
from pathlib import Path

from cost_model.engines.run_one_year import run_one_year
from cost_model.state.schema import (
    EMP_ID,
    EMP_GROSS_COMP,
    EMP_LEVEL,
    EMP_ROLE,
    EMP_HIRE_DATE,
    EMP_TERM_DATE,
    EMP_BIRTH_DATE,
    EMP_DEFERRAL_RATE,
    EMP_TENURE,
)
from cost_model.state.event_log import EVENT_COLS

def main():
    # 1) Load compensation params
    cfg_path = Path("config/compensation.yaml")
    comp_cfg = yaml.safe_load(cfg_path.read_text())
    global_params = types.SimpleNamespace(
        compensation=types.SimpleNamespace(**comp_cfg),
        days_into_year_for_promotion=0,
        days_into_year_for_cola=0,
        new_hires=types.SimpleNamespace(new_hire_rate=0.0),
        attrition=types.SimpleNamespace(new_hire_termination_rate=0.0),
        promotion_rules={}
    )

    # 2) Prepare a fake snapshot with all required columns
    df0 = pd.DataFrame({
        EMP_ID: ["1", "2", "3"],
        EMP_HIRE_DATE: [pd.Timestamp("2024-01-01")] * 3,
        EMP_BIRTH_DATE: [pd.NaT] * 3,
        EMP_ROLE: ["all"] * 3,
        EMP_LEVEL: [0, 1, 2],
        EMP_GROSS_COMP: [50_000.0, 80_000.0, 120_000.0],
        EMP_TERM_DATE: [pd.NaT] * 3,
        "active": [True] * 3,
        EMP_DEFERRAL_RATE: [pd.NA] * 3,
        "tenure_band": ["0-1"] * 3,
        EMP_TENURE: [0.0] * 3,
        "job_level_source": ["markov-promo", "markov-promo", "salary-band"],
        "exited": [False] * 3,
    })

    # 3) Build a minimal hazard table matching run_one_year expectations
    hazard_df = pd.DataFrame([{
        "simulation_year": 2025,
        EMP_ROLE: "all",
        "tenure_band": "0-1",
        "term_rate": 0.0,
        "comp_raise_pct": 0.0,
        "new_hire_termination_rate": 0.0,
        "cola_pct": 0.0,
        "cfg": global_params,
    }])

    # 4) Run one year
    events, snap = run_one_year(
        event_log=pd.DataFrame([], columns=EVENT_COLS),
        prev_snapshot=df0,
        year=2025,
        global_params=global_params,
        plan_rules=types.SimpleNamespace(),  # no plan rules for this smoke
        hazard_table=hazard_df,
        rng=np.random.default_rng(0),
        census_template_path="config/census_template.parquet",
    )

    # 5) Inspect and verify
    print("\n=== Raise Events ===")
    print(events[events.event_type == "EVT_RAISE"].to_string(index=False))

    print("\n=== Final Snapshot Salaries ===")
    print(snap[[EMP_ID, EMP_GROSS_COMP]].to_string(index=False))


if __name__ == "__main__":
    main()