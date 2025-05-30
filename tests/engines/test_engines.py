import pandas as pd
import numpy as np
from cost_model.engines import comp, term, hire
from cost_model.state.event_log import EVENT_COLS, EVT_COMP, EVT_TERM, EVT_HIRE, EVT_COLA
from cost_model.state.schema import EMP_ID, EMP_LEVEL, EMP_TENURE_BAND, EMP_GROSS_COMP, EMP_TERM_DATE


def make_mini_snapshot():
    # 3 employees, covering tenure bands 0-1, 1-3, 3+
    # Add term_date (all pd.NaT) and current_compensation (distinct values)
    return pd.DataFrame(
        [
            {
                EMP_ID: "E1",
                EMP_LEVEL: 1,
                EMP_TENURE_BAND: "0-1",
                "active": True,
                EMP_TERM_DATE: pd.NaT,
                EMP_GROSS_COMP: 50000,
                "employee_hire_date": "2020-01-01",
            },
            {
                EMP_ID: "E2",
                EMP_LEVEL: 2,
                EMP_TENURE_BAND: "1-3",
                "active": True,
                EMP_TERM_DATE: pd.NaT,
                EMP_GROSS_COMP: 60000,
                "employee_hire_date": "2019-01-01",
            },
            {
                EMP_ID: "E3",
                EMP_LEVEL: 3,
                EMP_TENURE_BAND: "5+",
                "active": True,
                EMP_TERM_DATE: pd.NaT,
                EMP_GROSS_COMP: 70000,
                "employee_hire_date": "2018-01-01",
            },
        ]
    ).set_index(EMP_ID)


def make_hazard_slice_comp():
    # Distinct comp_raise_pct for each level/tenure
    return pd.DataFrame(
        [
            {"level": 1, EMP_TENURE_BAND: "0-1", "comp_raise_pct": 0.03, "simulation_year": 2025, "cola_pct": 0.02},
            {"level": 2, EMP_TENURE_BAND: "1-3", "comp_raise_pct": 0.04, "simulation_year": 2025, "cola_pct": 0.02},
            {"level": 3, EMP_TENURE_BAND: "5+", "comp_raise_pct": 0.05, "simulation_year": 2025, "cola_pct": 0.02},
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
    rng = np.random.default_rng(42)
    result = comp.bump(snap, hazard, as_of, rng)
    assert isinstance(result, list)
    assert len(result) == 2  # Now returns [comp_events, cola_events]

    # Test comp events (first DataFrame)
    comp_df = result[0]
    assert set(comp_df.columns) == set(EVENT_COLS)
    assert len(comp_df) == 3
    assert all(comp_df["event_type"] == EVT_COMP)
    # Check values match hazard slice - these are now new compensation values, not percentages
    # Expected new comp values: 50000*1.03=51500, 60000*1.04=62400, 70000*1.05=73500
    expected_new_comp = [51500, 62400, 73500]
    actual_new_comp = sorted(comp_df["value_num"])
    assert all(
        abs(a - b) < 1e-4 for a, b in zip(actual_new_comp, expected_new_comp)
    )
    # Check meta contains employee info
    for meta in comp_df["meta"]:
        assert isinstance(meta, str)
        assert "Annual raise for" in meta

    # Test COLA events (second DataFrame)
    cola_df = result[1]
    assert set(cola_df.columns) == set(EVENT_COLS)
    assert len(cola_df) == 3
    assert all(cola_df["event_type"] == EVT_COLA)


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
