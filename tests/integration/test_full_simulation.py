# tests/integration/test_full_simulation.py
import pandas as pd
import numpy as np
import pytest
from types import SimpleNamespace # For creating simple config objects in tests

# Import the function to test
from cost_model.engines.run_one_year_engine import run_one_year

# Import config models used in the test
# These should match the actual Pydantic models if your engines expect them
# For this test, SimpleNamespace is used for plan_rules, so specific imports might not be needed
# if the engines called by run_one_year are robust to SimpleNamespace or if run_one_year
# correctly extracts/passes config.
# However, if engines expect typed Pydantic models, this test setup for 'plan_rules'
# might need to use the actual Pydantic models from cost_model.config.models.
# For now, assuming SimpleNamespace is sufficient for how run_one_year uses cfg.
from cost_model.plan_rules.eligibility import EligibilityConfig
from cost_model.plan_rules.eligibility_events import EligibilityEventsConfig
# from cost_model.plan_rules.enrollment import EnrollmentConfig # This was an old name
# from cost_model.plan_rules.contribution_increase import ContributionIncreaseConfig # This was an old name
# from cost_model.plan_rules.proactive_decrease import ProactiveDecreaseConfig # This was an old name

# If your plan rule engines now expect the Pydantic models from config.models:
from cost_model.config.models import (
    PlanRules, # Assuming run_one_year might pass this or parts of it
    AutoEnrollmentRules, # If EnrollmentConfig was a wrapper for this
    BehavioralParams,    # If EnrollmentConfig used this
    # ContributionIncreaseConfig, # Need to find the actual model name
    # ProactiveDecreaseConfig   # Need to find the actual model name
)
# For the test, we'll use SimpleNamespace as per the original test structure,
# assuming run_one_year correctly passes sub-configs to the rule engines.
# If not, the plan_rules instantiation below would need to use the full Pydantic models.

# Import event type constants if they are centrally defined
try:
    from cost_model.state.event_log import (
        EVT_ELIGIBLE, EVT_ENROLL, EVT_CONTRIB_INCR, EVT_COMP, EVT_TERM, EVT_HIRE
        # Add other specific event types used by your engines, e.g., for milestones or proactive decrease
    )
    # Define event types used in this test's 'expected' set if not imported
    EVT_1YR = "EVT_1YR" # As defined in test config
    EVT_PROACTIVE_DECREASE = "EVT_PROACTIVE_DECREASE" # As defined in test config
except ImportError:
    # Fallbacks if central constants aren't set up yet for testing
    EVT_ELIGIBLE = "EVT_ELIGIBLE"
    EVT_1YR = "EVT_1YR"
    EVT_ENROLL = "EVT_ENROLL"
    EVT_CONTRIB_INCR = "EVT_CONTRIB_INCR"
    EVT_PROACTIVE_DECREASE = "EVT_PROACTIVE_DECREASE"
    EVT_COMP = "EVT_COMP"
    EVT_TERM = "EVT_TERM"
    EVT_HIRE = "EVT_HIRE"


@pytest.fixture
def tiny_census():
    # This snapshot is the state at the START of the simulation year (e.g., 2025-01-01)
    df = pd.DataFrame({
        "employee_id":             ["A","B","C"],
        "birth_date":              [pd.Timestamp("1990-01-01"), pd.Timestamp("1985-06-15"), pd.Timestamp("1980-03-20")],
        "hire_date":               [pd.Timestamp("2024-01-01"), pd.Timestamp("2022-01-01"), pd.Timestamp("2020-01-01")],
        "deferral_rate":           [0.00, 0.10, 0.02], # Current deferral rate at start of year
        # For snapshot compatibility with engines that might expect these specific names:
        "employee_birth_date":     [pd.Timestamp("1990-01-01"), pd.Timestamp("1985-06-15"), pd.Timestamp("1980-03-20")],
        "employee_hire_date":      [pd.Timestamp("2024-01-01"), pd.Timestamp("2022-01-01"), pd.Timestamp("2020-01-01")],
        "employee_deferral_rate":  [0.00, 0.10, 0.02],
        "current_comp":            [60_000.0, 80_000.0, 120_000.0], # Compensation at start of year
        "term_date":               [pd.NaT, pd.NaT, pd.NaT],
        "role":                    ["all",  "all",  "all"], # Role for hazard table lookup
        "tenure_band":             ["all",  "all",  "all"], # Needed for comp.bump in run_one_year
        "active":                  [True,   True,   True],
    }).astype({
        'deferral_rate': 'float64', 'employee_deferral_rate': 'float64',
        'current_comp': 'float64', 'active': 'boolean', 'role': 'string'
    })
    return df.set_index("employee_id", drop=False)

@pytest.fixture
def base_event_log():
    # Historical events from *before* the simulation year starts (e.g., up to 2024-12-31)
    df = pd.DataFrame([
        {
            "event_id": "prior_evt_B", # Use string IDs for consistency
            "event_time": pd.Timestamp("2024-01-01"), # Date of prior rate election
            "employee_id": "B",
            "event_type": "EVT_CONTRIB_INCR", # Represents a prior deferral rate
            "value_num": 0.05, # B's rate in 2024 was 5%
            "value_json": None,
            "meta": None,
        },
        {
            "event_id": "prior_evt_C",
            "event_time": pd.Timestamp("2024-01-01"), # Date of prior rate election
            "employee_id": "C",
            "event_type": "EVT_CONTRIB_INCR", # Represents a prior deferral rate
            "value_num": 0.10, # C's rate in 2024 was 10%
            "value_json": None,
            "meta": None,
        },
    ], columns=[ # Ensure all standard event columns are present
        "event_id", "event_time", "employee_id", "event_type", "value_num", "value_json", "meta"
    ]).astype({ # Apply dtypes to match event_log.py
        "event_id": "string", "event_time": "datetime64[ns]", "employee_id": "string",
        "event_type": "string", "value_num": "Float64", # Use nullable float
        "value_json": "string", "meta": "string"
    })
    print("[BASE_EVENT_LOG DEBUG]\n", df.to_string())
    return df

def test_run_one_year_produces_all_events(tiny_census, base_event_log):
    # Config classes used by the test setup
    # These might need to be the actual Pydantic models if your engines require strict typing
    # For now, using SimpleNamespace as per the original test structure.
    # If EnrollmentConfig, etc., are actual classes, import and use them.
    # Assuming the plan_rules structure in run_one_year.py accesses attributes like cfg.eligibility
    class EligibilityConfig(SimpleNamespace): pass
    class EligibilityEventsConfig(SimpleNamespace): pass
    class EnrollmentConfig(SimpleNamespace): pass # Placeholder, actual model is AutoEnrollmentRules + BehavioralParams
    class ContributionIncreaseConfig(SimpleNamespace): pass
    class ProactiveDecreaseConfig(SimpleNamespace): pass


    # Minimal scenario config (in-memory)
    plan_rules = SimpleNamespace(
        eligibility=EligibilityConfig(min_age=21, min_service_months=12),
        eligibility_events=EligibilityEventsConfig(
            milestone_months=[12], milestone_years=[], event_type_map={12: EVT_1YR} # Use constant
        ),
        # For enrollment, run_one_year likely passes cfg.enrollment.
        # This should align with how PlanRules is structured if using Pydantic.
        # If enrollment.run expects PlanRules, then plan_rules itself would be passed.
        # For this test, assuming run_one_year handles passing the right sub-config.
        enrollment=EnrollmentConfig(window_days=30, allow_opt_out=True, default_rate=0.05), # Corresponds to auto_enroll + behavioral
        contribution_increase=ContributionIncreaseConfig(min_increase_pct=0.01, event_type=EVT_CONTRIB_INCR),
        proactive_decrease=ProactiveDecreaseConfig(lookback_months=12, threshold_pct=0.05, event_type=EVT_PROACTIVE_DECREASE),
    )
    # Minimal hazard table for year=2025
    hazard = pd.DataFrame({
        "year": [2025],
        "growth_rate": [0.0],  # No hires
        "cfg": [plan_rules],   # The SimpleNamespace config
        "role": ["all"],
        "tenure_band": ["all"],
        "comp_raise_pct": [0.0], # No comp events
        "term_rate": [0.0],      # No term events
    })
    rng = np.random.default_rng(42)

    # Run the simulation for one year
    # Assuming run_one_year is imported from your refactored engine structure
    from cost_model.engines.run_one_year import run_one_year
    
    # The test calls run_one_year twice. We only need to check the result of one call.
    # Focusing on the first call as per the original assertion structure.
    # The parameters for run_one_year in your test are:
    # year, prev_snapshot, event_log, hazard_table, rng, deterministic_term
    final_snapshot, combined_events = run_one_year(
        year=2025,
        prev_snapshot=tiny_census,    # Snapshot at start of 2025
        event_log=base_event_log,     # Events *before* 2025
        hazard_table=hazard,
        rng=rng,
        deterministic_term=True
    )

    # Assert at least one event of each expected type
    event_types_generated = set(combined_events["event_type"].dropna().unique())
    
    # --- CORRECTED EXPECTED SET ---
    # Based on the test config:
    # - Eligibility will fire for A, B, C.
    # - 1YR milestone will fire for A.
    # - Enrollment will fire for A (current rate 0.00, default 0.05).
    # - Contribution Increase will fire for B (current 0.10, prev 0.05 from base_event_log).
    # - Proactive Decrease will fire for C (current 0.02, prev 0.10 from base_event_log).
    # - EVT_COMP, EVT_TERM, EVT_HIRE will NOT fire due to 0 rates in hazard table.
    expected_event_types_for_this_test = {
        EVT_ELIGIBLE,
        EVT_1YR,
        EVT_ENROLL, # Assuming enrollment.run generates EVT_ENROLL for default/auto
        EVT_CONTRIB_INCR,
        EVT_PROACTIVE_DECREASE
    }
    # Also include event types from the base_event_log if they are expected to persist
    # In this case, base_event_log has EVT_CONTRIB_INCR, which is already in expected.
    # --- END OF CORRECTION ---

    missing = expected_event_types_for_this_test - event_types_generated
    assert not missing, f"Missing event types: {missing}. Generated: {event_types_generated}"

    # Assert snapshot has correct columns (basic check)
    assert "employee_id" in final_snapshot.columns # Since set_index(drop=False)
    assert isinstance(final_snapshot, pd.DataFrame)

    # --- Remove the second call to run_one_year and its assertions, ---
    # --- as it was causing confusion and the first call is sufficient ---
    # --- for testing what run_one_year produces in one cycle.       ---

    # Example: Check specific outcomes for employee C if needed
    c_events = combined_events[combined_events['employee_id'] == 'C']
    assert EVT_PROACTIVE_DECREASE in c_events['event_type'].values, "Employee C should have a proactive decrease event"

    # Headcount check (no hires, no terms in this specific config)
    assert set(final_snapshot.index) == set(tiny_census.index), "Index of employees should remain the same (no hires/terms)"
    assert final_snapshot.shape[0] == tiny_census.shape[0], "Number of employees should remain the same"

