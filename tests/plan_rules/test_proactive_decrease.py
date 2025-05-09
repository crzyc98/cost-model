import pandas as pd
import pytest
from datetime import datetime, timedelta

from cost_model.plan_rules.proactive_decrease import run as proactive_decrease_run
from cost_model.config.plan_rules import ProactiveDecreaseConfig

# Constants for event types
EVT_CONTRIB       = "EVT_CONTRIB"
EVT_PROACTIVE_DECREASE = getattr(ProactiveDecreaseConfig, "event_type", "EVT_PROACTIVE_DECREASE")


@pytest.fixture
def snapshot():
    # Current deferral rates at as_of
    df = pd.DataFrame({
        "employee_id": ["A", "B", "C"],
        "employee_deferral_rate": [0.10, 0.10, 0.10],
    }).set_index("employee_id")
    return df

@pytest.fixture
def contribution_history():
    """
    Build a small event log of contributions over the last 3 months
    as of 2025-04-01:
      - A: steady at 0.10
      - B: dropped from 0.12 → 0.10 (no need to decrease)
      - C: jumped to 0.15 then back to 0.10 (should trigger decrease from 0.15 → 0.10)
    """
    base = datetime(2025, 1, 1)
    evs = []
    # A: contributions perfectly match deferral 0.10
    for m in range(3):
        evs.append({
            "event_id": f"A{m}",
            "employee_id": "A",
            "event_type": EVT_CONTRIB,
            "event_time": base + timedelta(days=30 * m),
            "value_num": 0.10,
            "value_json": None,
            "meta": None,
        })
    # B: started at 0.12 then down to 0.10
    evs.append({
        "event_id": "B0",
        "employee_id": "B",
        "event_type": EVT_CONTRIB,
        "event_time": base,
        "value_num": 0.12,
        "value_json": None,
        "meta": None,
    })
    evs.append({
        "event_id": "B1",
        "employee_id": "B",
        "event_type": EVT_CONTRIB,
        "event_time": base + timedelta(days=30),
        "value_num": 0.10,
        "value_json": None,
        "meta": None,
    })
    # C: over-deferral then back
    evs.append({
        "event_id": "C0",
        "employee_id": "C",
        "event_type": EVT_CONTRIB,
        "event_time": base,
        "value_num": 0.15,
        "value_json": None,
        "meta": None,
    })
    evs.append({
        "event_id": "C1",
        "employee_id": "C",
        "event_type": EVT_CONTRIB,
        "event_time": base + timedelta(days=30),
        "value_num": 0.10,
        "value_json": None,
        "meta": None,
    })
    return pd.DataFrame(evs)

def test_no_decrease_when_within_threshold(snapshot, contribution_history):
    """
    If historical high-water mark ≤ current_rate + threshold, no decrease.
    Here threshold = 0.05, A never exceeded 0.10, B exceeded to 0.12 → current 0.10
    but drop of 0.02 < threshold 0.05 ⇒ no event for both.
    """
    as_of = pd.Timestamp("2025-04-01")
    cfg = ProactiveDecreaseConfig(
        lookback_months=3,
        threshold_pct=0.05,
        event_type="EVT_PROACTIVE_DECREASE"
    )
    out = proactive_decrease_run(snapshot, contribution_history, as_of, cfg)
    assert out == []  # no employees over-deferred enough to trigger


def test_emit_decrease_when_over_threshold(snapshot, contribution_history):
    """
    C had a peak at 0.15 and current is 0.10 → delta = 0.05 == threshold → should emit.
    B’s drop = 0.02 < threshold → no emit.
    """
    as_of = pd.Timestamp("2025-04-01")
    cfg = ProactiveDecreaseConfig(
        lookback_months=3,
        threshold_pct=0.04,
        event_type="DECREASE_EVT"
    )
    out = proactive_decrease_run(snapshot, contribution_history, as_of, cfg)
    df = pd.concat(out, ignore_index=True)

    # Only C should have a proactive decrease
    assert set(df.employee_id) == {"C"}
    import json
    row = df.iloc[0]
    assert row.event_type == "DECREASE_EVT"
    payload = json.loads(row.value_json)
    # Expect JSON/dict with old_rate=0.15, new_rate=0.10, delta = -0.05
    assert pytest.approx(payload["old_rate"]) == 0.15
    assert pytest.approx(payload["new_rate"]) == 0.10
    assert pytest.approx(payload["delta"]) == pytest.approx(-0.05)


def test_honors_custom_event_type(snapshot, contribution_history):
    """
    If the config overrides event_type, use that.
    And ensure event_time == as_of.
    """
    as_of = pd.Timestamp("2025-04-01")
    cfg = ProactiveDecreaseConfig(
        lookback_months=3,
        threshold_pct=0.01,
        event_type="MY_PROACTIVE_DEC"
    )
    out = proactive_decrease_run(snapshot, contribution_history, as_of, cfg)
    df = pd.concat(out, ignore_index=True)
    assert all(df.event_type == "MY_PROACTIVE_DEC")
    assert all(df.event_time == as_of)