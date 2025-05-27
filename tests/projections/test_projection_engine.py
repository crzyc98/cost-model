"""
Tests for the projection engine functionality.

These tests validate that the run_projection_engine function correctly
processes snapshots and event logs to produce expected summary metrics.
"""
import pytest
import pandas as pd
import numpy as np
from types import SimpleNamespace
from typing import Dict, Tuple
from datetime import datetime, timedelta

from cost_model.projections.runner import run_projection_engine
from cost_model.state.schema import (
    EMP_ID, EMP_HIRE_DATE, EMP_TERM_DATE, EMP_BIRTH_DATE, 
    EMP_GROSS_COMP, EMP_DEFERRAL_RATE
)
from cost_model.state.event_log import EVT_HIRE, EVT_TERM, EVT_COMP, EVT_CONTRIB


@pytest.fixture
def simple_config():
    """Create a minimal configuration for testing."""
    return SimpleNamespace(
        global_parameters=SimpleNamespace(
            start_year=2025,
            projection_years=2,
            random_seed=42,
            target_headcount=5,
        ),
        plan_rules=SimpleNamespace(
            eligibility=SimpleNamespace(
                minimum_age=21,
                service_requirement_months=3,
            ),
            employer_match=SimpleNamespace(
                tiers=[
                    SimpleNamespace(
                        employee_contribution_min=0.0,
                        employee_contribution_max=6.0,
                        employer_match_rate=0.5,
                    ),
                ],
                cap=3.0,
            ),
            auto_enrollment=SimpleNamespace(
                enabled=True,
                default_rate=3.0,
            ),
        ),
    )


@pytest.fixture
def tiny_snapshot():
    """Create a minimal snapshot with 5 employees."""
    today = datetime.now()
    start_year = today.year
    
    # Create 5 employees with different characteristics
    data = {
        EMP_ID: ["A", "B", "C", "D", "E"],
        EMP_HIRE_DATE: [
            f"{start_year-2}-01-15",  # 2 years tenure
            f"{start_year-1}-06-01",  # 1 year tenure
            f"{start_year}-01-01",    # New hire this year
            f"{start_year-5}-03-10",  # 5 years tenure, will be terminated
            f"{start_year-3}-11-20",  # 3 years tenure
        ],
        EMP_TERM_DATE: [
            None,                     # Active
            None,                     # Active
            None,                     # Active
            f"{start_year}-07-15",    # Terminated mid-year
            None,                     # Active
        ],
        EMP_BIRTH_DATE: [
            f"{start_year-45}-05-10", # 45 years old
            f"{start_year-32}-02-20", # 32 years old
            f"{start_year-28}-11-05", # 28 years old
            f"{start_year-52}-08-30", # 52 years old
            f"{start_year-38}-04-15", # 38 years old
        ],
        EMP_GROSS_COMP: [75000, 60000, 50000, 85000, 90000],
        EMP_DEFERRAL_RATE: [6.0, 4.0, 0.0, 8.0, 5.0],
        "active": [True, True, True, False, True],
        "eligible": [True, True, False, True, True],
        "enrolled": [True, True, False, True, True],
        "tenure_band": ["1-3", "0-1", "0-1", "5+", "3-5"],
    }
    
    df = pd.DataFrame(data)
    df.set_index(EMP_ID, inplace=True)
    
    return df


@pytest.fixture
def tiny_event_log():
    """Create a minimal event log with hire, term, and comp events."""
    today = datetime.now()
    start_year = today.year
    
    events = []
    
    # Hire events
    for emp_id, hire_date, comp in [
        ("A", f"{start_year-2}-01-15", 70000),
        ("B", f"{start_year-1}-06-01", 55000),
        ("C", f"{start_year}-01-01", 50000),
        ("D", f"{start_year-5}-03-10", 80000),
        ("E", f"{start_year-3}-11-20", 85000),
    ]:
        events.append({
            "event_id": f"hire_{emp_id}",
            "event_time": pd.Timestamp(hire_date),
            EMP_ID: emp_id,
            "event_type": EVT_HIRE,
            "value_num": comp,
            "value_json": None,
            "meta": f"Initial hire for {emp_id}"
        })
    
    # Comp increase events
    for emp_id, event_time, new_comp in [
        ("A", f"{start_year-1}-01-15", 75000),
        ("D", f"{start_year-1}-04-01", 85000),
        ("E", f"{start_year-1}-12-01", 90000),
    ]:
        events.append({
            "event_id": f"comp_{emp_id}_{event_time}",
            "event_time": pd.Timestamp(event_time),
            EMP_ID: emp_id,
            "event_type": EVT_COMP,
            "value_num": new_comp,
            "value_json": None,
            "meta": f"Compensation change for {emp_id}"
        })
    
    # Termination event
    events.append({
        "event_id": "term_D",
        "event_time": pd.Timestamp(f"{start_year}-07-15"),
        EMP_ID: "D",
        "event_type": EVT_TERM,
        "value_num": None,
        "value_json": None,
        "meta": "Termination"
    })
    
    # Contribution events
    for emp_id, event_time, rate in [
        ("A", f"{start_year-1}-02-01", 6.0),
        ("B", f"{start_year-1}-07-15", 4.0),
        ("D", f"{start_year-1}-05-10", 8.0),
        ("E", f"{start_year-1}-12-15", 5.0),
    ]:
        events.append({
            "event_id": f"contrib_{emp_id}_{event_time}",
            "event_time": pd.Timestamp(event_time),
            EMP_ID: emp_id,
            "event_type": EVT_CONTRIB,
            "value_num": rate,
            "value_json": None,
            "meta": f"Contribution rate change for {emp_id}"
        })
    
    return pd.DataFrame(events)


def test_basic_projection(simple_config, tiny_snapshot, tiny_event_log):
    """
    Test that run_projection_engine produces expected results with simple inputs.
    
    This test validates:
    1. The correct number of years in yearly_snapshots
    2. The employment status counts match expectations
    3. Active headcount matches expectations
    """
    # Run the projection engine
    (
        yearly_snapshots,
        all_events,
        core_summary,
        employment_summary,
        participant_summary
    ) = run_projection_engine(
        simple_config,
        tiny_snapshot,
        tiny_event_log
    )
    
    # Test 1: Verify we have the expected number of years in the output
    expected_years = simple_config.global_parameters.projection_years + 1  # +1 for initial year
    assert len(yearly_snapshots) == expected_years
    
    # Test 2: Verify employment summary metrics for the initial year
    start_year = simple_config.global_parameters.start_year
    initial_employment = employment_summary[employment_summary['Year'] == start_year].iloc[0]
    
    # Expected counts based on our fixture data
    assert initial_employment['Continuous Active'] == 3  # A, B, E
    assert initial_employment['New Hire Active'] == 1    # C
    assert initial_employment['Experienced Terminated'] == 1  # D
    assert initial_employment['New Hire Terminated'] == 0
    assert initial_employment['Total Terminated'] == 1   # D
    assert initial_employment['Active'] == 4             # A, B, C, E
    
    # Test 3: Verify core summary metrics
    initial_core = core_summary[core_summary['Year'] == start_year].iloc[0]
    assert initial_core['Active Headcount'] == 4
    
    # Test 4: Check that the final year has expected metrics
    final_year = start_year + simple_config.global_parameters.projection_years
    final_employment = employment_summary[employment_summary['Year'] == final_year].iloc[0]
    
    # We don't know exact values for the final year since it depends on the simulation,
    # but we can check that values are reasonable
    assert final_employment['Active'] > 0
    assert final_employment['Continuous Active'] + final_employment['New Hire Active'] == final_employment['Active']


def test_empty_inputs(simple_config):
    """Test that run_projection_engine handles empty inputs gracefully."""
    empty_snapshot = pd.DataFrame(columns=[
        EMP_ID, EMP_HIRE_DATE, EMP_TERM_DATE, EMP_BIRTH_DATE, 
        EMP_GROSS_COMP, EMP_DEFERRAL_RATE, "active", "eligible", "enrolled"
    ]).set_index(EMP_ID)
    
    empty_events = pd.DataFrame(columns=[
        "event_id", "event_time", EMP_ID, "event_type", 
        "value_num", "value_json", "meta"
    ])
    
    # Run the projection engine with empty inputs
    (
        yearly_snapshots,
        all_events,
        core_summary,
        employment_summary,
        participant_summary
    ) = run_projection_engine(
        simple_config,
        empty_snapshot,
        empty_events
    )
    
    # Verify we still get the expected structure
    assert isinstance(yearly_snapshots, dict)
    assert isinstance(all_events, pd.DataFrame)
    assert isinstance(core_summary, pd.DataFrame)
    assert isinstance(employment_summary, pd.DataFrame)
    assert isinstance(participant_summary, pd.DataFrame)
    
    # Check that summaries have the expected years
    start_year = simple_config.global_parameters.start_year
    expected_years = [start_year + i for i in range(simple_config.global_parameters.projection_years + 1)]
    
    for year in expected_years:
        assert year in employment_summary['Year'].values
        assert year in core_summary['Year'].values


def test_experienced_terminated_counts(simple_config, tiny_snapshot, tiny_event_log):
    """
    Specifically test that experienced terminated employees are correctly counted.
    This test focuses on the issue with missing experienced terminated counts.
    """
    # Run the projection engine
    (
        yearly_snapshots,
        all_events,
        core_summary,
        employment_summary,
        participant_summary
    ) = run_projection_engine(
        simple_config,
        tiny_snapshot,
        tiny_event_log
    )
    
    # Get the first year's employment summary
    start_year = simple_config.global_parameters.start_year
    first_year = employment_summary[employment_summary['Year'] == start_year].iloc[0]
    
    # Employee D should be counted as "Experienced Terminated"
    assert first_year['Experienced Terminated'] == 1
    
    # Check the snapshot for this employee
    first_snapshot = yearly_snapshots[start_year]
    if 'D' in first_snapshot.index:
        employee_d = first_snapshot.loc['D']
        # Verify the employee has the correct status
        assert not employee_d.get('active', True)
        assert pd.notna(employee_d.get(EMP_TERM_DATE))
    
    # Alternatively, if the employee is filtered out, check the event log
    term_events = all_events[
        (all_events['event_type'] == EVT_TERM) & 
        (all_events[EMP_ID] == 'D')
    ]
    assert not term_events.empty, "Termination event for employee D is missing"


if __name__ == "__main__":
    # This allows running the tests directly with python
    pytest.main(["-xvs", __file__])
