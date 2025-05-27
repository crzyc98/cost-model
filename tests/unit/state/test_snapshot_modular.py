import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from cost_model.state.schema import (
    EMP_ID, EMP_HIRE_DATE, EMP_GROSS_COMP, EMP_TERM_DATE, EMP_DEFERRAL_RATE,
    EMP_TENURE, EMP_LEVEL, EMP_LEVEL_SOURCE, EMP_EXITED, EMP_ACTIVE,
    EMP_TENURE_BAND, EMP_BIRTH_DATE
)

from cost_model.state.snapshot import (
    build_full,
    update,
    _get_first_event,
    _get_last_event,
    _assign_tenure_band,
    _ensure_columns_and_types,
    _extract_hire_details
)
from cost_model.state.snapshot.helpers import (
    get_first_event,
    get_last_event,
    ensure_columns_and_types,
    validate_snapshot
)
from cost_model.state.snapshot.details import extract_hire_details
from cost_model.state.snapshot.tenure import (
    assign_tenure_band,
    compute_tenure
)
from cost_model.state.snapshot.snapshot_build import build_full
from cost_model.state.snapshot.snapshot_update import update

def test_constants():
    """Verify constants are properly defined."""
    assert isinstance(SNAPSHOT_COLS, list)
    assert isinstance(SNAPSHOT_DTYPES, dict)
    assert len(SNAPSHOT_COLS) > 0
    assert len(SNAPSHOT_DTYPES) > 0
    
    # Check that all columns have corresponding dtypes
    assert set(SNAPSHOT_COLS) == set(SNAPSHOT_DTYPES.keys())

def test_helpers():
    """Test helper functions."""
    # Create sample events
    events = pd.DataFrame({
        "event_id": ["e1", "e2", "e3", "e4"],
        "event_time": ["2023-01-01", "2023-01-01", "2023-02-01", "2023-02-01"],
        "employee_id": ["emp1", "emp1", "emp2", "emp2"],
        "event_type": ["EVT_HIRE", "EVT_COMP", "EVT_HIRE", "EVT_COMP"],
        "value_num": [100000, 105000, 110000, 115000],
        "value_json": ["{}", "{}", "{}", "{}"],
        "meta": ["{}", "{}", "{}", "{}"]
    })
    
    # Test get_first_event
    first_hires = get_first_event(events, "EVT_HIRE")
    assert len(first_hires) == 2
    assert set(first_hires["employee_id"]) == {"emp1", "emp2"}
    
    # Test get_last_event
    last_comps = get_last_event(events, "EVT_COMP")
    assert len(last_comps) == 2
    assert set(last_comps["employee_id"]) == {"emp1", "emp2"}
    
    # Test ensure_columns_and_types
    df = pd.DataFrame({"employee_id": ["emp1"], "employee_hire_date": ["2023-01-01"]})
    result = ensure_columns_and_types(df)
    assert set(result.columns) == set(const.SNAPSHOT_COLS)
    assert all(result.dtypes[col] == const.SNAPSHOT_DTYPES[col] for col in const.SNAPSHOT_COLS)

def test_details():
    """Test hire details extraction."""
    # Create sample hire events with JSON data
    events = pd.DataFrame({
        "event_id": ["e1", "e2"],
        "event_time": ["2023-01-01", "2023-02-01"],
        "employee_id": ["emp1", "emp2"],
        "event_type": ["EVT_HIRE", "EVT_HIRE"],
        "value_json": [
            '{"employee_level": "Engineer", "birth_date": "1990-01-01"}',
            '{"employee_level": "Manager", "birth_date": "1985-01-01"}'
        ]
    })
    
    details = extract_hire_details(events)
    assert len(details) == 2
    assert details.loc["emp1", EMP_LEVEL] == "Engineer"
    assert details.loc["emp2", EMP_LEVEL] == "Manager"
    assert pd.to_datetime(details.loc["emp1", EMP_BIRTH_DATE]) == pd.Timestamp("1990-01-01")
    assert pd.to_datetime(details.loc["emp2", EMP_BIRTH_DATE]) == pd.Timestamp("1985-01-01")

def test_tenure():
    """Test tenure calculations."""
    # Create sample data with proper index
    df = pd.DataFrame({
        const.EMP_HIRE_DATE: ["2023-01-01", "2021-01-01"]
    }, index=pd.Index(["emp1", "emp2"], name=const.EMP_ID))
    
    # Test tenure band assignment
    assert assign_tenure_band(0.5) == "0-1"
    assert assign_tenure_band(2.5) == "1-3"
    assert assign_tenure_band(4.5) == "3-5"
    assert assign_tenure_band(6.0) == "5+"
    
    # Test tenure calculation
    result = compute_tenure(
        df=df,
        as_of=pd.Timestamp("2023-12-31"),
        hire_date_col=const.EMP_HIRE_DATE,
        out_tenure_col=const.EMP_TENURE,
        out_band_col=const.EMP_TENURE_BAND
    )
    
    # Employee hired in 2023 should have 0-1 year tenure
    assert result.loc["emp1", EMP_TENURE_BAND] == "0-1"
    # Employee hired in 2021 should have 2.99 years tenure -> 1-3 band
    assert result.loc["emp2", EMP_TENURE_BAND] == "1-3"

def test_snapshot_build():
    """Test building a full snapshot."""
    # Create sample events
    events = pd.DataFrame({
        "event_id": ["e1", "e2", "e3", "e4"],
        "event_time": ["2023-01-01", "2023-01-01", "2023-01-01", "2023-02-01"],
        "employee_id": ["emp1", "emp1", "emp1", "emp1"],
        "event_type": ["EVT_HIRE", "EVT_COMP", "EVT_COMP", "EVT_TERM"],
        "value_num": [100000, 105000, 110000, None],
        "value_json": [
            "{\"employee_level\": \"Engineer\", \"birth_date\": \"1990-01-01\"}",
            "{}", "{}", "{}"
        ],
        "meta": ["{}", "{}", "{}", "{}"]
    })
    
    snapshot = build_full(events, 2023)
    
    # Verify snapshot structure
    assert len(snapshot) == 1
    assert snapshot.index.name == EMP_ID
    assert snapshot.loc["emp1", EMP_HIRE_DATE] == pd.Timestamp("2023-01-01")
    assert snapshot.loc["emp1", EMP_GROSS_COMP] == 110000
    assert snapshot.loc["emp1", EMP_TERM_DATE] == pd.Timestamp("2023-02-01")
    assert snapshot.loc["emp1", EMP_LEVEL] == "Engineer"
    assert pd.notna(snapshot.loc["emp1", EMP_BIRTH_DATE])
    assert pd.notna(snapshot.loc["emp1", EMP_LEVEL])  # Last comp value
    assert not snapshot.loc["emp1", EMP_EXITED]
    
    # Verify tenure calculation
    # Employee hired in 2023-01-01, as_of is 2023-12-31 -> ~11.99 months
    assert snapshot.loc["emp1", EMP_TENURE] == pytest.approx(0.997, abs=1e-3)  # ~11.99 months
    assert snapshot.loc["emp1", EMP_TENURE_BAND] == "0-1"  # Less than 1 year
    assert snapshot.loc["emp1", EMP_TENURE_BAND] == "0-1"  # Less than 1 year
    assert emp1[const.EMP_TENURE_BAND] == "0-1"  # Less than 1 year

def test_snapshot_update():
    """Test updating an existing snapshot."""
    # Test 1: Basic update
    # Create initial snapshot
    prev_snapshot = pd.DataFrame({
        EMP_ID: ["emp1"],
        EMP_HIRE_DATE: ["2023-01-01"],
        EMP_GROSS_COMP: [100000],
        EMP_ACTIVE: [True]
    }).set_index(EMP_ID)
    
    # Create new events
    new_events = pd.DataFrame({
        "event_id": ["e1", "e2"],
        "event_time": ["2023-02-01", "2023-03-01"],
        "employee_id": ["emp1", "emp2"],
        "event_type": ["EVT_COMP", "EVT_HIRE"],
        "value_num": [105000, 110000],
        "value_json": ["{}", '{"employee_level": "Engineer", "birth_date": "1990-01-01"}'],
        "meta": ["{}", "{}"]
    })
    
    updated = update(prev_snapshot, new_events, 2023)
    assert updated.loc["emp1", EMP_GROSS_COMP] == 105000
    assert updated.loc["emp2", EMP_LEVEL] == "Engineer"
    
    # Test 2: Multiple updates for same employee
    prev_snapshot = pd.DataFrame({
        const.EMP_ID: ["emp1"],
        const.EMP_HIRE_DATE: ["2023-01-01"],
        const.EMP_GROSS_COMP: [100000],
        const.EMP_ACTIVE: [True]
    }).set_index(const.EMP_ID)
    
    new_events = pd.DataFrame({
        "event_id": ["e1", "e2", "e3"],
        "event_time": ["2023-02-01", "2023-02-02", "2023-02-03"],
        "employee_id": ["emp1", "emp1", "emp1"],
        "event_type": ["EVT_COMP", "EVT_COMP", "EVT_COMP"],
        "value_num": [105000, 110000, 115000],
        "value_json": ["{}", "{}", "{}"],
        "meta": ["{}", "{}", "{}"]
    })
    
    updated = update(prev_snapshot, new_events, 2023)
    assert updated.loc["emp1", const.EMP_GROSS_COMP] == 115000  # Last value wins
    
    # Test 3: Multiple terminations
    prev_snapshot = pd.DataFrame({
        const.EMP_ID: ["emp1"],
        const.EMP_HIRE_DATE: ["2023-01-01"],
        const.EMP_GROSS_COMP: [100000],
        const.EMP_ACTIVE: [True]
    }).set_index(const.EMP_ID)
    
    new_events = pd.DataFrame({
        "event_id": ["e1", "e2"],
        "event_time": ["2023-02-01", "2023-02-02"],
        "employee_id": ["emp1", "emp1"],
        "event_type": ["EVT_TERM", "EVT_TERM"],
        "value_num": [None, None],
        "value_json": ["{}", "{}"],
        "meta": ["{}", "{}"]
    })
    
    updated = update(prev_snapshot, new_events, 2023)
    assert not updated.loc["emp1", const.EMP_ACTIVE]
    
    # Test 4: Invalid JSON in hire events
    prev_snapshot = pd.DataFrame({
        const.EMP_ID: ["emp1"],
        const.EMP_HIRE_DATE: ["2023-01-01"],
        const.EMP_GROSS_COMP: [100000],
        const.EMP_ACTIVE: [True]
    }).set_index(const.EMP_ID)
    
    new_events = pd.DataFrame({
        "event_id": ["e1", "e2"],
        "event_time": ["2023-02-01", "2023-02-01"],
        "employee_id": ["emp1", "emp2"],
        "event_type": ["EVT_HIRE", "EVT_HIRE"],
        "value_num": [None, 110000],  # Add value_num column
        "value_json": ["invalid json", "{\"role\": \"Engineer\"}"],
        "meta": ["{}", "{}"]
    })
    
    updated = update(prev_snapshot, new_events, 2023)
    assert len(updated) == 2  # Should have both employees
    assert updated.loc["emp1", const.EMP_GROSS_COMP] == 100000  # Original comp should be preserved
    assert updated.loc["emp2", const.EMP_GROSS_COMP] == 110000  # New hire comp should be set
    
    # Test 5: Missing required columns in snapshot
    prev_snapshot = pd.DataFrame({
        const.EMP_ID: ["emp1"],
        const.EMP_HIRE_DATE: ["2023-01-01"],
        const.EMP_ACTIVE: [True]
    }).set_index(const.EMP_ID)  # Missing EMP_GROSS_COMP
    
    new_events = pd.DataFrame({
        "event_id": ["e1"],
        "event_time": ["2023-02-01"],
        "employee_id": ["emp1"],
        "event_type": ["EVT_COMP"],
        "value_num": [105000],
        "value_json": ["{}"],
        "meta": ["{}"]
    })
    
    updated = update(prev_snapshot, new_events, 2023)
    assert const.EMP_GROSS_COMP in updated.columns  # Column added
    assert updated.loc["emp1", const.EMP_GROSS_COMP] == 105000
    
    # Test 6: Empty events
    prev_snapshot = pd.DataFrame({
        const.EMP_ID: ["emp1"],
        const.EMP_HIRE_DATE: ["2023-01-01"],
        const.EMP_GROSS_COMP: [100000],
        const.EMP_ACTIVE: [True]
    }).set_index(const.EMP_ID)
    
    new_events = pd.DataFrame()
    updated = update(prev_snapshot, new_events, 2023)
    assert updated.equals(prev_snapshot)  # No changes
    
    # Test 7: Invalid event types
    prev_snapshot = pd.DataFrame({
        const.EMP_ID: ["emp1"],
        const.EMP_HIRE_DATE: ["2023-01-01"],
        const.EMP_GROSS_COMP: [100000],
        const.EMP_ACTIVE: [True]
    }).set_index(const.EMP_ID)
    
    new_events = pd.DataFrame({
        "event_id": ["e1"],
        "event_time": ["2023-02-01"],
        "employee_id": ["emp1"],
        "event_type": ["INVALID_EVENT"],  # Invalid event type
        "value_num": [105000],
        "value_json": ["{}"],
        "meta": ["{}"]
    })
    
    updated = update(prev_snapshot, new_events, 2023)
    assert updated.equals(prev_snapshot)  # No changes
    
    # Test 8: Duplicate events
    prev_snapshot = pd.DataFrame({
        const.EMP_ID: ["emp1"],
        const.EMP_HIRE_DATE: ["2023-01-01"],
        const.EMP_GROSS_COMP: [100000],
        const.EMP_ACTIVE: [True]
    }).set_index(const.EMP_ID)
    
    new_events = pd.DataFrame({
        "event_id": ["e1", "e1"],  # Duplicate event_id
        "event_time": ["2023-02-01", "2023-02-01"],
        "employee_id": ["emp1", "emp1"],
        "event_type": ["EVT_COMP", "EVT_COMP"],
        "value_num": [105000, 110000],
        "value_json": ["{}", "{}"],
        "meta": ["{}", "{}"]
    })
    
    updated = update(prev_snapshot, new_events, 2023)
    assert updated.loc["emp1", const.EMP_GROSS_COMP] == 110000  # Last value wins
    
    # Create new events
    new_events = pd.DataFrame({
        "event_id": ["e1", "e2"],
        "event_time": ["2023-02-01", "2023-03-01"],
        "employee_id": ["emp1", "emp2"],
        "event_type": ["EVT_COMP", "EVT_HIRE"],
        "value_num": [105000, 110000],
        "value_json": ["{}", '{"role": "Engineer", "birth_date": "1990-01-01"}'],
        "meta": ["{}", "{}"]
    })
    
    updated = update(prev_snapshot, new_events, 2023)
    
    # Verify emp1's comp update
    assert updated.loc["emp1", const.EMP_GROSS_COMP] == 105000
    
    # Verify new hire emp2
    assert updated.loc["emp2", const.EMP_ROLE] == "Engineer"
    assert updated.loc["emp2", const.EMP_ACTIVE]
    
    # Verify tenure bands
    assert updated.loc["emp1", const.EMP_TENURE_BAND] == "0-1"
    assert updated.loc["emp2", const.EMP_TENURE_BAND] == "0-1"
