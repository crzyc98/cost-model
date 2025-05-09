# tests/plan_rules/test_enrollment.py
from cost_model.config.plan_rules import AutoEnrollmentConfig
from cost_model.config.plan_rules import EnrollmentConfig

import pandas as pd
from cost_model.plan_rules.enrollment import run as enrollment_run


def make_snapshot(records):
    return pd.DataFrame(records)


def make_events(records):
    return pd.DataFrame(records)


def test_not_yet_eligible():
    today = pd.Timestamp("2025-01-01")
    snap = make_snapshot(
        [
            {"employee_id": 1},
        ]
    )
    events = make_events([])  # No eligibility events
    cfg = EnrollmentConfig(window_days=30)
    result = enrollment_run(snap, events, today, cfg)
    assert len(result) == 0


def test_already_opted_out():
    today = pd.Timestamp("2025-01-01")
    snap = make_snapshot(
        [
            {"employee_id": 2},
        ]
    )
    events = make_events(
        [
            {
                "employee_id": 2,
                "event_type": "EVT_ELIGIBLE",
                "event_time": pd.Timestamp("2024-12-01"),
            },
            {
                "employee_id": 2,
                "event_type": "EVT_OPT_OUT",
                "event_time": pd.Timestamp("2024-12-15"),
            },
        ]
    )
    cfg = EnrollmentConfig(window_days=30)
    result = enrollment_run(snap, events, today, cfg)
    assert len(result) == 0


def test_already_enrolled():
    today = pd.Timestamp("2025-01-01")
    snap = make_snapshot(
        [
            {"employee_id": 3},
        ]
    )
    events = make_events(
        [
            {
                "employee_id": 3,
                "event_type": "EVT_ELIGIBLE",
                "event_time": pd.Timestamp("2024-12-01"),
            },
            {
                "employee_id": 3,
                "event_type": "EVT_ENROLL",
                "event_time": pd.Timestamp("2024-12-20"),
            },
        ]
    )
    cfg = EnrollmentConfig(window_days=30)
    result = enrollment_run(snap, events, today, cfg)
    assert len(result) == 0





def snap():
    # All four are eligible as of Jan 1
    return pd.DataFrame(index=["A", "B", "C", "D"])


def base_events():
    from datetime import datetime

    return pd.DataFrame(
        [
            {
                "event_time": datetime(2025, 1, 1),
                "employee_id": "A",
                "event_type": "EVT_ELIGIBLE",
            },
            {
                "event_time": datetime(2025, 1, 1),
                "employee_id": "B",
                "event_type": "EVT_ELIGIBLE",
            },
            {
                "event_time": datetime(2025, 1, 1),
                "employee_id": "C",
                "event_type": "EVT_ELIGIBLE",
            },
            {
                "event_time": datetime(2025, 1, 1),
                "employee_id": "D",
                "event_type": "EVT_ELIGIBLE",
            },
            # B already auto-enrolled
            {
                "event_time": datetime(2025, 1, 1),
                "employee_id": "B",
                "event_type": "EVT_AUTO_ENROLL",
            },
            # C already voluntarily enrolled
            {
                "event_time": datetime(2025, 1, 1),
                "employee_id": "C",
                "event_type": "EVT_ENROLL",
            },
        ]
    )


def test_auto_and_voluntary_enrollment():
    cfg = EnrollmentConfig(
        auto_enrollment=AutoEnrollmentConfig(
            enabled=True, default_rate=0.03, window_days=90
        ),
        voluntary_enrollment_rate=0.5,
        default_rate=0.02,
    )
    out = enrollment_run(snap(), base_events(), pd.Timestamp("2025-01-01"), cfg)
    df = pd.concat(out, ignore_index=True)
    # A: should get EVT_AUTO_ENROLL; D: 50% chance to get EVT_ENROLL
    assert "A" in df.employee_id[df.event_type == "EVT_AUTO_ENROLL"].values
    assert set(df.employee_id[df.event_type == "EVT_ENROLL"]).issubset({"A", "D"})


def test_auto_and_voluntary_match_multiplier():
    # If both auto_enrollment.enabled and voluntary_match_multiplier are set, only auto-enrollment should occur
    class DummyConfig:
        auto_enrollment = AutoEnrollmentConfig(
            enabled=True, default_rate=0.05, window_days=0
        )
        voluntary_enrollment_rate = 0.3
        voluntary_match_multiplier = 2.0
        window_days = 0
        default_rate = 0.05

    cfg = DummyConfig()
    out = enrollment_run(snap(), base_events(), pd.Timestamp("2025-01-01"), cfg)
    df = pd.concat(out, ignore_index=True)
    # Only auto-enroll events should be present, no voluntary enrolls
    assert set(df["event_type"].unique()) == {"EVT_AUTO_ENROLL"}
    assert set(df.employee_id) == {"A", "D"}


def test_rng_seed_override():
    # Test that setting random_seed in config gives deterministic voluntary enrollment
    class DummyConfig:
        auto_enrollment = None
        voluntary_enrollment_rate = 0.3
        window_days = 0
        default_rate = 0.05
        random_seed = 12345

    cfg = DummyConfig()
    snap_df = pd.DataFrame(index=[str(i) for i in range(10)])
    events_df = pd.DataFrame(
        [
            {
                "employee_id": str(i),
                "event_type": "EVT_ELIGIBLE",
                "event_time": pd.Timestamp("2025-01-01"),
            }
            for i in range(10)
        ]
    )
    out1 = enrollment_run(snap_df, events_df, pd.Timestamp("2025-01-02"), cfg)
    out2 = enrollment_run(snap_df, events_df, pd.Timestamp("2025-01-02"), cfg)
    df1 = pd.concat(out1, ignore_index=True)
    df2 = pd.concat(out2, ignore_index=True)
    assert df1.equals(
        df2
    ), "Voluntary enrollment should be deterministic if random_seed is set"


def test_match_incentive_effect():
    # 100 eligible employees
    N = 100
    employees = [str(i) for i in range(N)]
    snap = pd.DataFrame(index=employees)
    events = pd.DataFrame(
        [
            {
                "employee_id": emp,
                "event_type": "EVT_ELIGIBLE",
                "event_time": pd.Timestamp("2025-01-01"),
            }
            for emp in employees
        ]
    )

    # Plan with match and voluntary_match_multiplier
    class DummyConfig:
        auto_enrollment = None
        voluntary_enrollment_rate = 0.2
        voluntary_match_multiplier = 2.0
        window_days = 0
        default_rate = 0.05
        employer_match_tiers = [{"match_rate": 0.5}]

    cfg = DummyConfig()
    from cost_model.plan_rules.enrollment import run as enroll_run

    out = enroll_run(snap, events, pd.Timestamp("2025-01-01"), cfg)
    df = pd.concat(out, ignore_index=True)
    # Should see ~40 enrollments (0.2 * 2.0 * 100)
    assert 30 < len(df) < 50, f"Expected ~40 enrollments, got {len(df)}"
    assert all(df["event_type"] == "EVT_ENROLL")


def test_eligible_not_opted_out_or_enrolled():
    today = pd.Timestamp("2025-01-01")
    snap = make_snapshot(
        [
            {"employee_id": 4},
        ]
    )
    events = make_events(
        [
            {
                "employee_id": 4,
                "event_type": "EVT_ELIGIBLE",
                "event_time": pd.Timestamp("2024-12-01"),
            },
        ]
    )
    cfg = EnrollmentConfig(window_days=30, default_rate=0.05)
    result = enrollment_run(snap, events, today, cfg)
    assert len(result) == 1
    evt = result[0]
    assert evt["employee_id"].iloc[0] == "4"
    assert evt["event_type"].iloc[0] == "EVT_ENROLL"
    expected_date = pd.Timestamp("2024-12-01") + pd.Timedelta(days=30)
    assert evt["event_time"].iloc[0] == expected_date
    assert evt["value"].iloc[0] == 0.05
