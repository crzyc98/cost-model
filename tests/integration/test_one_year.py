import pandas as pd
import numpy as np
from math import ceil

from cost_model.engines.run_one_year import run_one_year
from cost_model.state.event_log import EVT_COMP, EVT_TERM, EVT_HIRE, EVENT_COLS
from cost_model.state import snapshot


def make_tiny_events():
    # Minimal event log for snapshot bootstrap: 3 actives, with role and tenure_band
    rows = [
        {
            "event_time": "2024-01-01",
            "employee_id": "E1",
            "event_type": EVT_HIRE,
            "value_num": np.nan,
            "value_json": None,
            "meta": "{}",
            "role": "A",
            "tenure_band": "0-1",
        },
        {
            "event_time": "2024-01-01",
            "employee_id": "E2",
            "event_type": EVT_HIRE,
            "value_num": np.nan,
            "value_json": None,
            "meta": "{}",
            "role": "B",
            "tenure_band": "1-3",
        },
        {
            "event_time": "2024-01-01",
            "employee_id": "E3",
            "event_type": EVT_HIRE,
            "value_num": np.nan,
            "value_json": None,
            "meta": "{}",
            "role": "C",
            "tenure_band": "5+",
        },
    ]
    df = pd.DataFrame(rows)
    # Ensure all EVENT_COLS are present
    for col in EVENT_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def make_tiny_hazard():
    # 2025 hazard rates for 3 roles
    return pd.DataFrame(
        [
            {
                "year": 2025,
                "role": "A",
                "tenure_band": "0-1",
                "comp_raise_pct": 0.03,
                "term_rate": 0.10,
                "nh_term_rate": 0.20,
                "growth_rate": 0.10,
                "hire_comp": 55000,
            },
            {
                "year": 2025,
                "role": "B",
                "tenure_band": "1-3",
                "comp_raise_pct": 0.04,
                "term_rate": 0.05,
                "nh_term_rate": 0.20,
                "growth_rate": 0.10,
                "hire_comp": 65000,
            },
            {
                "year": 2025,
                "role": "C",
                "tenure_band": "5+",
                "comp_raise_pct": 0.05,
                "term_rate": 0.02,
                "nh_term_rate": 0.20,
                "growth_rate": 0.10,
                "hire_comp": 75000,
            },
        ]
    )


def test_one_year_cycle():
    events = make_tiny_events()
    snap = snapshot.build_full(events, 2025)
    # Patch in role and tenure_band for all employees from events
    for col in ["role", "tenure_band"]:
        if col in events.columns:
            snap[col] = snap.index.map(
                events.set_index("employee_id")[col].to_dict()
            ).astype(str)
    hazard = make_tiny_hazard()
    rng = np.random.default_rng(42)
    # Run for 2025
    updated_snapshot, new_log = run_one_year(
        2025, snap, events, hazard, rng, deterministic_term=True
    )
    # Check headcount > 0
    survivors = updated_snapshot[
        updated_snapshot["term_date"].isna()
        | (updated_snapshot["term_date"] > pd.Timestamp("2025-01-01"))
    ]
    assert survivors.shape[0] > 0
    # Check event log types
    assert new_log["event_type"].isin([EVT_COMP, EVT_TERM, EVT_HIRE]).all()
    # Check event log grew by sum of engine outputs
    n_initial = len(events)
    n_new = len(new_log) - n_initial
    # Expect 3 comp, 1 term, 2 hire, 2 hire comp events = 8 new events
    assert n_new == 8
    # Check no comp after term dates
    for _, row in updated_snapshot.iterrows():
        if pd.notna(row.get("term_date")):
            comp_events = new_log[
                (new_log["employee_id"] == row.name)
                & (new_log["event_type"] == EVT_COMP)
            ]
            for _, ce in comp_events.iterrows():
                assert pd.Timestamp(ce["event_time"]) <= row["term_date"]
    # Check headcount moves toward growth target
    survivors_eoy = updated_snapshot[
        updated_snapshot["term_date"].isna()
        | (updated_snapshot["term_date"] > pd.Timestamp("2025-12-31"))
    ]
    expected_target = ceil(survivors.shape[0] * 1.1)
    assert survivors_eoy.shape[0] <= expected_target
