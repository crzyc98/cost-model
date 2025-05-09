# tests/state/test_snapshot.py

import pandas as pd
import pytest  # Make sure pytest is installed
from pathlib import Path
import uuid  # For creating test event IDs if needed
import json  # For value_json

# Modules to test
from cost_model.state import snapshot, event_log
from cost_model.utils.columns import EMP_ID  # Make sure this is correct

# --- Test Fixtures (Setup) ---


@pytest.fixture(scope="module")  # Run once for all tests in this module
def bootstrap_events_df() -> pd.DataFrame:
    """
    Fixture to load the initial events created by the bootstrap script.
    Assumes the bootstrap script has been run and saved events.
    """
    # Path to our generated bootstrap events
    bootstrap_event_path = Path(
        "output_dev/testing/bootstrap_events_for_snapshot_test.parquet"
    )

    # ---- IMPORTANT: Run your bootstrap script first! ----
    # You would typically run your scripts/seed_events_from_census.py script
    # BEFORE running pytest, or call its main function here if appropriate.
    # For now, we assume the file exists.
    if not bootstrap_event_path.exists():
        pytest.skip(
            f"Bootstrap event file not found at {bootstrap_event_path}. Run bootstrap script first."
        )

    events = event_log.load_log(bootstrap_event_path)
    # Basic check on loaded events
    assert not events.empty, "Loaded bootstrap events are empty."
    assert EMP_ID in events.columns, f"'{EMP_ID}' column missing in bootstrap events."
    return events


# --- Tests for build_full ---


def test_build_full_structure(bootstrap_events_df):
    """Verify the structure, columns, dtypes, and index of the full snapshot."""
    snap = snapshot.build_full(bootstrap_events_df)

    assert isinstance(snap, pd.DataFrame), "Result should be a DataFrame."
    assert snap.index.name == EMP_ID, f"Index name should be '{EMP_ID}'."

    # Check columns exist (order doesn't strictly matter here, but presence does)
    expected_cols = set(snapshot.SNAPSHOT_COLS)
    actual_cols = set(snap.columns)
    assert (
        actual_cols == expected_cols
    ), f"Snapshot columns mismatch. Expected: {expected_cols}, Got: {actual_cols}"

    # Check dtypes (using the definition from snapshot.py)
    for col, expected_dtype in snapshot.SNAPSHOT_DTYPES.items():
        assert str(snap[col].dtype) == str(
            expected_dtype
        ), f"Dtype mismatch for {col}. Expected: {expected_dtype}, Got: {snap[col].dtype}"


def test_build_full_active_count(bootstrap_events_df):
    """Verify the count of active employees matches expectations (e.g., 100)."""
    snap = snapshot.build_full(bootstrap_events_df)
    # Assuming your bootstrap events accurately reflect the initial 100 active
    expected_active_count = 100  # Based on your census description
    assert (
        snap["active"].sum() == expected_active_count
    ), f"Expected {expected_active_count} active employees."


def test_build_full_known_employee(bootstrap_events_df):
    """Verify details for a specific employee known from the census."""
    snap = snapshot.build_full(bootstrap_events_df)

    # Use a real employee ID from our bootstrap events
    known_active_emp_id = "DUMMY_EX_716045_000063"
    expected_hire_date = pd.Timestamp("1999-04-05")
    expected_birth_date = pd.Timestamp("1967-04-02")
    expected_role = "Manager"
    expected_comp = 189230.15  # Actual compensation value from snapshot

    assert (
        known_active_emp_id in snap.index
    ), f"Employee {known_active_emp_id} not found in snapshot."
    emp_data = snap.loc[known_active_emp_id]

    assert (
        emp_data["active"]
    ), f"Employee {known_active_emp_id} should be active"

    # Verify other fields for this known employee
    assert pd.notna(
        emp_data["hire_date"]
    ), f"Employee {known_active_emp_id} should have a hire date"
    assert pd.notna(
        emp_data["birth_date"]
    ), f"Employee {known_active_emp_id} should have a birth date"
    assert pd.notna(
        emp_data["role"]
    ), f"Employee {known_active_emp_id} should have a role"
    assert pd.notna(
        emp_data["current_comp"]
    ), f"Employee {known_active_emp_id} should have current compensation"
    assert pd.isna(
        emp_data["term_date"]
    ), f"Employee {known_active_emp_id} should not have a termination date"
    assert (
        emp_data["hire_date"] == expected_hire_date
    ), f"Employee {known_active_emp_id} hire date mismatch. Expected: {expected_hire_date}, Got: {emp_data['hire_date']}"
    assert (
        emp_data["birth_date"] == expected_birth_date
    ), f"Employee {known_active_emp_id} birth date mismatch. Expected: {expected_birth_date}, Got: {emp_data['birth_date']}"
    assert (
        emp_data["role"] == expected_role
    ), f"Employee {known_active_emp_id} role mismatch. Expected: {expected_role}, Got: {emp_data['role']}"
    assert (
        emp_data["current_comp"] == expected_comp
    ), f"Employee {known_active_emp_id} compensation mismatch. Expected: {expected_comp}, Got: {emp_data['current_comp']}"


# --- Helper to create sample events ---
# Use the helper from event_log.py if available, otherwise define one here
def _create_test_event(
    emp_id, time_str, type, value_num=None, value_json_dict=None, meta_str=None
):
    event_data = {
        "event_id": str(uuid.uuid4()),
        "event_time": pd.Timestamp(time_str),
        EMP_ID: str(emp_id),
        "event_type": type,
        "value_num": value_num,
        "value_json": json.dumps(value_json_dict) if value_json_dict else None,
        "meta": meta_str,
    }
    # Ensure variant constraint
    if value_num is not None and value_json_dict is not None:
        raise ValueError("Cannot provide both value_num and value_json_dict")
    return event_data


# --- Tests for update ---


@pytest.fixture
def initial_snapshot(bootstrap_events_df) -> pd.DataFrame:
    """Fixture providing the snapshot as of the start of the first simulation year."""
    # Depending on how bootstrap events are dated, you might need an as_of date here
    # For now, assume build_full gives the state after initial events.
    return snapshot.build_full(bootstrap_events_df)


def test_update_new_hire(initial_snapshot):
    """Test adding a completely new employee."""
    prev_snap = initial_snapshot
    new_emp_id = "NH_001"  # An ID not in initial_snapshot
    assert new_emp_id not in prev_snap.index

    new_events_list = [
        _create_test_event(
            new_emp_id,
            "2025-03-15 09:00:00",
            event_log.EVT_HIRE,
            value_json_dict={"role": "Analyst", "birth_date": "1995-05-20"},
        ),
        _create_test_event(
            new_emp_id, "2025-03-15 09:00:01", event_log.EVT_COMP, value_num=55000.0
        ),
    ]
    new_events_df = pd.DataFrame(new_events_list)
    # Ensure correct dtypes for the new events DataFrame before passing to update
    new_events_df = new_events_df[event_log.EVENT_COLS].astype(
        event_log.EVENT_PANDAS_DTYPES
    )

    updated_snap = snapshot.update(prev_snap, new_events_df)

    assert len(updated_snap) == len(prev_snap) + 1, "Snapshot should have one more row."
    assert new_emp_id in updated_snap.index, "New hire ID should be in the index."

    new_hire_data = updated_snap.loc[new_emp_id]
    assert new_hire_data["hire_date"] == pd.Timestamp("2025-03-15 09:00:00")
    assert new_hire_data["role"] == "Analyst"
    assert new_hire_data["birth_date"] == pd.Timestamp("1995-05-20")
    assert new_hire_data["current_comp"] == 55000.0
    assert pd.isna(new_hire_data["term_date"])
    assert new_hire_data["active"]


def test_update_termination(initial_snapshot):
    """Test terminating an existing employee."""
    prev_snap = initial_snapshot
    # TODO: Choose an ID known to be active in the initial snapshot
    emp_to_term = initial_snapshot[initial_snapshot["active"]].index[0]

    new_events_list = [
        _create_test_event(emp_to_term, "2025-07-31 17:00:00", event_log.EVT_TERM)
    ]
    new_events_df = pd.DataFrame(new_events_list)
    new_events_df = new_events_df[event_log.EVENT_COLS].astype(
        event_log.EVENT_PANDAS_DTYPES
    )

    updated_snap = snapshot.update(prev_snap, new_events_df)

    assert len(updated_snap) == len(prev_snap), "Snapshot row count should not change."
    assert emp_to_term in updated_snap.index, "Terminated employee should still exist."

    term_emp_data = updated_snap.loc[emp_to_term]
    assert term_emp_data["term_date"] == pd.Timestamp("2025-07-31 17:00:00")
    assert not term_emp_data["active"]
    # Verify other fields haven't unexpectedly changed
    assert term_emp_data["hire_date"] == prev_snap.loc[emp_to_term, "hire_date"]


def test_update_comp_change(initial_snapshot):
    """Test changing compensation for an existing employee."""
    prev_snap = initial_snapshot
    # TODO: Choose an ID known to be active
    emp_to_update = initial_snapshot[initial_snapshot["active"]].index[1]
    original_comp = prev_snap.loc[emp_to_update, "current_comp"]
    new_comp = original_comp * 1.10  # Example 10% raise

    new_events_list = [
        _create_test_event(
            emp_to_update, "2025-04-01 00:00:00", event_log.EVT_COMP, value_num=new_comp
        )
    ]
    new_events_df = pd.DataFrame(new_events_list)
    new_events_df = new_events_df[event_log.EVENT_COLS].astype(
        event_log.EVENT_PANDAS_DTYPES
    )

    updated_snap = snapshot.update(prev_snap, new_events_df)

    assert len(updated_snap) == len(prev_snap)
    assert emp_to_update in updated_snap.index

    updated_emp_data = updated_snap.loc[emp_to_update]
    assert (
        updated_emp_data["current_comp"] == new_comp
    ), "Compensation should be updated."
    assert updated_emp_data["active"]


def test_update_multiple_events_same_employee(initial_snapshot):
    """Test handling multiple events for one employee in the batch."""
    prev_snap = initial_snapshot
    # TODO: Choose an ID known to be active
    emp_to_update = initial_snapshot[initial_snapshot["active"]].index[2]

    new_events_list = [
        _create_test_event(
            emp_to_update, "2025-02-01 09:00:00", event_log.EVT_COMP, value_num=70000.0
        ),
        _create_test_event(
            emp_to_update, "2025-08-15 10:00:00", event_log.EVT_COMP, value_num=75000.0
        ),  # Last comp event
        _create_test_event(
            emp_to_update, "2025-11-30 17:00:00", event_log.EVT_TERM
        ),  # Last term event
    ]
    new_events_df = pd.DataFrame(new_events_list)
    new_events_df = new_events_df[event_log.EVENT_COLS].astype(
        event_log.EVENT_PANDAS_DTYPES
    )

    updated_snap = snapshot.update(prev_snap, new_events_df)

    assert len(updated_snap) == len(prev_snap)
    assert emp_to_update in updated_snap.index

    updated_emp_data = updated_snap.loc[emp_to_update]
    assert (
        updated_emp_data["current_comp"] == 75000.0
    ), "Should reflect the last comp event."
    assert updated_emp_data["term_date"] == pd.Timestamp(
        "2025-11-30 17:00:00"
    ), "Should reflect the last term event."
    assert not updated_emp_data["active"]


def test_update_empty_events(initial_snapshot):
    """Test update with an empty new_events DataFrame."""
    prev_snap = initial_snapshot
    empty_new_events = pd.DataFrame(columns=event_log.EVENT_COLS).astype(
        event_log.EVENT_PANDAS_DTYPES
    )

    updated_snap = snapshot.update(prev_snap, empty_new_events)

    pd.testing.assert_frame_equal(
        updated_snap, prev_snap
    ), "Snapshot should be unchanged if new_events is empty."


def test_update_terminated_employee_again(initial_snapshot):
    """Test terminating an employee who is already terminated."""
    prev_snap = initial_snapshot
    # Get an active employee
    emp_to_term = initial_snapshot[initial_snapshot["active"]].index[0]

    # First terminate them
    first_term_event = _create_test_event(
        emp_to_term, "2025-07-31 17:00:00", event_log.EVT_TERM
    )
    first_term_df = pd.DataFrame([first_term_event], columns=event_log.EVENT_COLS)
    first_term_df = first_term_df.astype(event_log.EVENT_PANDAS_DTYPES)

    # Apply first termination
    snap_after_first_term = snapshot.update(prev_snap, first_term_df)

    # Try to terminate them again
    second_term_event = _create_test_event(
        emp_to_term, "2025-08-15 17:00:00", event_log.EVT_TERM
    )
    second_term_df = pd.DataFrame([second_term_event], columns=event_log.EVENT_COLS)
    second_term_df = second_term_df.astype(event_log.EVENT_PANDAS_DTYPES)

    final_snap = snapshot.update(snap_after_first_term, second_term_df)

    assert len(final_snap) == len(prev_snap)
    assert emp_to_term in final_snap.index

    emp_data = final_snap.loc[emp_to_term]
    assert not emp_data["active"]
    # Term date should be updated to the latest termination
    assert emp_data["term_date"] == pd.Timestamp("2025-08-15 17:00:00")


def test_update_compensation_for_terminated_employee(initial_snapshot):
    """Test changing compensation for an employee who is already terminated."""
    prev_snap = initial_snapshot
    # Get an active employee
    emp_to_update = initial_snapshot[initial_snapshot["active"]].index[0]

    # First terminate them
    term_event = _create_test_event(
        emp_to_update, "2025-07-31 17:00:00", event_log.EVT_TERM
    )
    term_df = pd.DataFrame([term_event], columns=event_log.EVENT_COLS)
    term_df = term_df.astype(event_log.EVENT_PANDAS_DTYPES)

    # Apply termination
    snap_after_term = snapshot.update(prev_snap, term_df)

    # Try to update compensation
    comp_event = _create_test_event(
        emp_to_update, "2025-08-15 00:00:00", event_log.EVT_COMP, value_num=100000.0
    )
    comp_df = pd.DataFrame([comp_event], columns=event_log.EVENT_COLS)
    comp_df = comp_df.astype(event_log.EVENT_PANDAS_DTYPES)

    final_snap = snapshot.update(snap_after_term, comp_df)

    assert len(final_snap) == len(prev_snap)
    assert emp_to_update in final_snap.index

    emp_data = final_snap.loc[emp_to_update]
    assert not emp_data["active"]
    assert emp_data["term_date"] == pd.Timestamp("2025-07-31 17:00:00")
    # Compensation should not be updated since employee is terminated
    # Compensation should not change for terminated employees
    assert (
        emp_data["current_comp"] == snap_after_term.loc[emp_to_update, "current_comp"]
    )


def test_update_multiple_new_hires(initial_snapshot):
    """Test handling multiple new hires in one batch."""
    prev_snap = initial_snapshot

    # Create three new hires
    new_hires = [
        _create_test_event(
            "NH_001",
            "2025-03-15 09:00:00",
            event_log.EVT_HIRE,
            value_json_dict={"role": "Analyst", "birth_date": "1995-05-20"},
        ),
        _create_test_event(
            "NH_002",
            "2025-03-15 09:00:00",
            event_log.EVT_HIRE,
            value_json_dict={"role": "Developer", "birth_date": "1992-08-15"},
        ),
        _create_test_event(
            "NH_003",
            "2025-03-15 09:00:00",
            event_log.EVT_HIRE,
            value_json_dict={"role": "Manager", "birth_date": "1985-03-10"},
        ),
    ]

    # Add compensations for each new hire
    compensations = [
        _create_test_event(
            "NH_001", "2025-03-15 09:00:01", event_log.EVT_COMP, value_num=55000.0
        ),
        _create_test_event(
            "NH_002", "2025-03-15 09:00:01", event_log.EVT_COMP, value_num=70000.0
        ),
        _create_test_event(
            "NH_003", "2025-03-15 09:00:01", event_log.EVT_COMP, value_num=90000.0
        ),
    ]

    # Combine all events
    all_events = new_hires + compensations
    events_df = pd.DataFrame(all_events, columns=event_log.EVENT_COLS)
    events_df = events_df.astype(event_log.EVENT_PANDAS_DTYPES)

    updated_snap = snapshot.update(prev_snap, events_df)

    assert len(updated_snap) == len(prev_snap) + 3

    # Verify each new hire
    for emp_id, role, comp in zip(
        ["NH_001", "NH_002", "NH_003"],
        ["Analyst", "Developer", "Manager"],
        [55000.0, 70000.0, 90000.0],
    ):
        assert emp_id in updated_snap.index
        emp_data = updated_snap.loc[emp_id]
        assert emp_data["active"]
        assert emp_data["role"] == role
        assert emp_data["current_comp"] == comp
        assert pd.isna(emp_data["term_date"]) or emp_data["term_date"] == pd.NaT


def test_update_invalid_event_data(initial_snapshot):
    """Test handling events with invalid data types."""
    prev_snap = initial_snapshot

    # Create events with invalid data
    invalid_events = [
        # Invalid hire event (missing required fields)
        _create_test_event(
            "NH_001",
            "2025-03-15 09:00:00",
            event_log.EVT_HIRE,
            value_json_dict={"role": "Analyst"},
        ),  # Missing birth_date
        # Invalid comp event (non-numeric value)
        _create_test_event(
            "NH_002",
            "2025-03-15 09:00:00",
            event_log.EVT_COMP,
            value_num="not_a_number",
        ),
        # Invalid term event (invalid timestamp)
        _create_test_event(
            "NH_003",
            "2025-03-15 09:00:00",
            event_log.EVT_TERM,
            value_json_dict={"invalid": "data"},
        ),
    ]

    try:
        events_df = pd.DataFrame(invalid_events, columns=event_log.EVENT_COLS)
        events_df = events_df.astype(event_log.EVENT_PANDAS_DTYPES)

        # Should not raise exceptions but log warnings
        updated_snap = snapshot.update(prev_snap, events_df)

        assert len(updated_snap) == len(prev_snap)
        # No new employees should be added
        for emp_id in ["NH_001", "NH_002", "NH_003"]:
            assert emp_id not in updated_snap.index
    except ValueError as e:
        # Expecting a ValueError due to invalid data types
        assert "could not convert string to float" in str(e)


# - Events exactly on the boundary timestamp (if relevant)
