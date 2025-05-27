import pandas as pd
import numpy as np
from cost_model.engines import comp, term, hire
from cost_model.state.event_log import EVENT_COLS, EVT_COMP, EVT_TERM, EVT_HIRE


def make_mini_snapshot():
    # 3 employees, covering tenure bands 0-1, 1-3, 3+
    # Add term_date (all pd.NaT) and current_compensation (distinct values)
    return pd.DataFrame(
        [
            {
                "employee_id": "E1",
                "role": "A",
                "tenure_band": "0-1",
                "active": True,
                "term_date": pd.NaT,
                "current_compensation": 50000,
            },
            {
                "employee_id": "E2",
                "role": "B",
                "tenure_band": "1-3",
                "active": True,
                "term_date": pd.NaT,
                "current_compensation": 60000,
            },
            {
                "employee_id": "E3",
                "role": "C",
                "tenure_band": "5+",
                "active": True,
                "term_date": pd.NaT,
                "current_compensation": 70000,
            },
        ]
    ).set_index("employee_id")


def make_hazard_slice_comp():
    # Distinct comp_raise_pct for each role/tenure
    return pd.DataFrame(
        [
            {"role": "A", "tenure_band": "0-1", "comp_raise_pct": 0.03},
            {"role": "B", "tenure_band": "1-3", "comp_raise_pct": 0.04},
            {"role": "C", "tenure_band": "5+", "comp_raise_pct": 0.05},
        ]
    )


def make_hazard_slice_term():
    # Term rates for each role/tenure
    return pd.DataFrame(
        [
            {"role": "A", "tenure_band": "0-1", "term_rate": 0.10},
            {"role": "B", "tenure_band": "1-3", "term_rate": 0.05},
            {"role": "C", "tenure_band": "5+", "term_rate": 0.02},
        ]
    )


def make_hazard_slice_hire():
    # New-hire term rate for 0-1
    return pd.DataFrame(
        [
            {"role": "A", "tenure_band": "0-1", "term_rate": 0.20},
        ]
    )


def test_comp_bump_interface():
    snap = make_mini_snapshot()
    hazard = make_hazard_slice_comp()
    as_of = pd.Timestamp("2025-01-01")
    result = comp.bump(snap, hazard, as_of)
    assert isinstance(result, list)
    assert len(result) == 1
    df = result[0]
    assert set(df.columns) == set(EVENT_COLS)
    assert len(df) == 3
    assert all(df["event_type"] == EVT_COMP)
    # Check values match hazard slice
    expected = [0.03, 0.04, 0.05]
    assert all(
        abs(a - b) < 1e-4 for a, b in zip(sorted(df["value_num"]), sorted(expected))
    )
    # Check meta contains old/new comp and pct
    for meta in df["meta"]:
        m = eval(meta) if isinstance(meta, str) else meta
        assert "old_comp" in m and "new_comp" in m and "pct" in m


def test_term_run_interface():
    snap = make_mini_snapshot()
    hazard = make_hazard_slice_term()
    rng = np.random.default_rng(42)
    result = term.run(snap, hazard, rng, deterministic=True)
    assert isinstance(result, list)
    assert len(result) == 1
    df = result[0]
    assert set(df.columns) == set(EVENT_COLS)
    # For deterministic, expect ceil(3 * mean([.1, .05, .02])) = 1
    assert len(df) == 1
    assert all(df["event_type"] == EVT_TERM)
    # Event time should be in 2025
    for t in df["event_time"]:
        assert (
            pd.Timestamp("2025-01-01") <= pd.Timestamp(t) <= pd.Timestamp("2025-12-31")
        )


def test_hire_run_interface():
    snap = make_mini_snapshot()
    hazard = make_hazard_slice_hire()
    rng = np.random.default_rng(42)
    target_eoy = 5
    result = hire.run(snap, target_eoy, hazard, rng)
    assert isinstance(result, list)
    assert len(result) == 2
    hires_df, comp_df = result
    assert set(hires_df.columns) == set(EVENT_COLS)
    assert set(comp_df.columns) == set(EVENT_COLS)
    # 3 survivors, need 2 more, nh_term_rate = 0.20, so hires = ceil(2/0.8) = 3
    assert len(hires_df) == 3
    assert len(comp_df) == 3
    assert all(hires_df["event_type"] == EVT_HIRE)
    assert all(comp_df["event_type"] == EVT_COMP)
    # Hire dates in 2025
    for t in hires_df["event_time"]:
        assert (
            pd.Timestamp("2025-01-01") <= pd.Timestamp(t) <= pd.Timestamp("2025-12-31")
        )
