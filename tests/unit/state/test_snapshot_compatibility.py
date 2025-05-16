import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from cost_model.state.schema import (
    EMP_ID, EMP_HIRE_DATE, EMP_GROSS_COMP, EMP_TERM_DATE, EMP_ACTIVE,
    EMP_BIRTH_DATE, EMP_TENURE, EMP_TENURE_BAND, EMP_LEVEL, EMP_LEVEL_SOURCE,
    EMP_EXITED, EMP_LEVEL_SOURCE
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
from cost_model.state.snapshot.tenure import assign_tenure_band as _assign_tenure_band
from cost_model.state.snapshot.details import extract_hire_details as _extract_hire_details

def test_compatibility_build_full():
    """Test compatibility layer for build_full."""
    # Create sample events
    events = pd.DataFrame({
        "event_id": ["e1", "e2", "e3", "e4", "e5"],
        "event_time": ["2023-01-01", "2023-01-01", "2023-01-01", "2023-02-01", "2023-03-01"],
        "employee_id": ["emp1", "emp1", "emp1", "emp1", "emp1"],
        "event_type": ["EVT_HIRE", "EVT_COMP", "EVT_COMP", "EVT_COMP", "EVT_TERM"],
        "value_num": [100000, 105000, 110000, 115000, None],
        "value_json": [
            "{\"employee_level\": \"Engineer\", \"birth_date\": \"1990-01-01\"}",
            "{}", "{}", "{}", "{}"
        ],
        "meta": ["{}", "{}", "{}", "{}", "{}"]
    })
    
    # Build snapshot through compatibility layer
    snapshot = build_full(events, 2023)
    
    # Verify deprecation warning was issued
    with pytest.warns(DeprecationWarning):
        snapshot = build_full(events, 2023)
    
    # Verify snapshot structure
    assert isinstance(snapshot, pd.DataFrame)
    assert snapshot.index.name == EMP_ID
    assert EMP_LEVEL in snapshot.columns
    assert EMP_GROSS_COMP in snapshot.columns
    assert EMP_TERM_DATE in snapshot.columns
    assert EMP_ACTIVE in snapshot.columns
    
    # Verify specific values
    emp1 = snapshot.loc["emp1"]
    assert emp1[EMP_LEVEL] == "Engineer"
    assert emp1[EMP_GROSS_COMP] == 115000
    assert emp1[EMP_TERM_DATE] == pd.Timestamp("2023-03-01")
    assert not emp1[EMP_ACTIVE]
    assert pd.notna(emp1[EMP_LEVEL])
    assert not emp1["employee_active"]

def test_compatibility_update():
    """Test compatibility layer for update."""
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
        "value_json": ["{}", "{\"employee_level\": \"Engineer\", \"birth_date\": \"1990-01-01\"}"]
    })
    
    # Update through compatibility layer
    with pytest.warns(DeprecationWarning):
        updated = update(prev_snapshot, new_events, 2023)
    
    # Verify emp1's comp update
    assert updated.loc["emp1", EMP_GROSS_COMP] == 105000
    
    # Verify new hire emp2
    assert updated.loc["emp2", EMP_LEVEL] == "Engineer"
    assert updated.loc["emp2", EMP_ACTIVE] == True

def test_compatibility_helpers():
    """Test compatibility layer helper functions."""
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
    with pytest.warns(DeprecationWarning):
        first_hires = _get_first_event(events, "EVT_HIRE")
    assert len(first_hires) == 2
    assert set(first_hires["employee_id"]) == {"emp1", "emp2"}
    
    # Test get_last_event
    with pytest.warns(DeprecationWarning):
        last_comps = _get_last_event(events, "EVT_COMP")
    assert len(last_comps) == 2
    assert set(last_comps["employee_id"]) == {"emp1", "emp2"}
    
    # Test assign_tenure_band
    with pytest.warns(DeprecationWarning):
        band = _assign_tenure_band(2.5)
    assert band == "1-3"
    
    # Test ensure_columns_and_types
    df = pd.DataFrame({"employee_id": ["emp1"], "employee_hire_date": ["2023-01-01"]})
    with pytest.warns(DeprecationWarning):
        result = _ensure_columns_and_types(df)
    assert set(result.columns) == {"employee_id", "employee_hire_date", "employee_role", "employee_gross_compensation", "employee_termination_date", "employee_active"}
    
    # Test extract_hire_details
    hire_events = pd.DataFrame({
        "event_id": ["e1", "e2"],
        "event_time": ["2023-01-01", "2023-02-01"],
        "employee_id": ["emp1", "emp2"],
        "event_type": ["EVT_HIRE", "EVT_HIRE"],
        "value_json": [
            '{"role": "Engineer", "birth_date": "1990-01-01"}',
            '{"role": "Manager", "birth_date": "1985-01-01"}'
        ]
    })
    
    with pytest.warns(DeprecationWarning):
        details = _extract_hire_details(hire_events)
    assert len(details) == 2
    assert details.loc["emp1", "employee_role"] == "Engineer"
    assert details.loc["emp2", "employee_role"] == "Manager"
