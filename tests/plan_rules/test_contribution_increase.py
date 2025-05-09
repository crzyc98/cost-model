# cost_model/tests/plan_rules/test_contribution_increase.py

import pandas as pd
import pytest
from datetime import datetime

from cost_model.plan_rules.contribution_increase import run as contribution_increase_run
from cost_model.config.plan_rules import ContributionIncreaseConfig

# These should match the constants your engine emits
EVT_ENROLL = "EVT_ENROLL"
DEFAULT_INCREASE_EVT = getattr(ContributionIncreaseConfig, "event_type", "EVT_CONTRIB_INCREASE")


@pytest.fixture
def snapshot():
    # New-year rates for three employees
    return (
        pd.DataFrame({
            "employee_id": ["X", "Y", "Z"],
            "employee_deferral_rate": [0.05, 0.05, 0.05],
        })
        .set_index("employee_id")
    )

@pytest.fixture
def prior_events():
    # X: no prior event → old_rate = 0.0
    # Y: prior enrollment at 0.04
    # Z: prior enrollment at 0.045
    return pd.DataFrame([
        {
            "event_id": "e1",
            "employee_id": "Y",
            "event_type": EVT_ENROLL,
            "event_time": datetime(2024, 1, 1),
            "value_num": 0.04,
            "value_json": None,
            "meta": None,
        },
        {
            "event_id": "e2",
            "employee_id": "Z",
            "event_type": EVT_ENROLL,
            "event_time": datetime(2024, 1, 1),
            "value_num": 0.045,
            "value_json": None,
            "meta": None,
        },
    ])


def test_emits_for_new_and_significant_increases(snapshot, prior_events):
    """
    - X: old_rate=0 → new_rate=0.05 → delta=0.05 ≥ 0.01 ⇒ event
    - Y: old_rate=0.04 → new_rate=0.05 → delta=0.01 ≥ 0.01 ⇒ event
    - Z: old_rate=0.045 → new_rate=0.05 → delta=0.005 < 0.01 ⇒ NO event
    """
    as_of = pd.Timestamp("2025-01-01")
    cfg = ContributionIncreaseConfig(min_increase_pct=0.01)
    out = contribution_increase_run(snapshot, prior_events, as_of, cfg)
    # concat all returned DataFrames
    df = pd.concat(out, ignore_index=True)

    # we should get exactly X and Y
    assert set(df.employee_id) == {"X", "Y"}

    # validate the JSON payload
    import json
    for row in df.itertuples():
        payload = json.loads(row.value_json)
        assert pytest.approx(payload["new_rate"]) == 0.05
        if row.employee_id == "X":
            assert pytest.approx(payload["old_rate"]) == 0.0
            assert pytest.approx(payload["delta"]) == 0.05
        else:  # Y
            assert pytest.approx(payload["old_rate"]) == 0.04
            assert pytest.approx(payload["delta"]) == 0.01


def test_filters_out_small_increases(snapshot, prior_events):
    """
    If min_increase_pct is 0.02, only X (delta=0.05) qualifies; Y’s delta=0.01 is too small.
    """
    as_of = pd.Timestamp("2025-01-01")
    cfg = ContributionIncreaseConfig(min_increase_pct=0.02)
    out = contribution_increase_run(snapshot, prior_events, as_of, cfg)
    df = pd.concat(out, ignore_index=True)

    assert set(df.employee_id) == {"X"}


def test_honors_custom_event_type(snapshot, prior_events):
    """
    If the config overrides the `event_type`, the emitted rows use that instead of the default.
    """
    as_of = pd.Timestamp("2025-01-01")
    cfg = ContributionIncreaseConfig(min_increase_pct=0.01, event_type="MY_CUSTOM_EVT")
    out = contribution_increase_run(snapshot, prior_events, as_of, cfg)
    df = pd.concat(out, ignore_index=True)

    assert set(df.event_type) == {"MY_CUSTOM_EVT"}