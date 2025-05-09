# tests/plan_rules/test_contributions.py

import pandas as pd
import pytest
import json # Import json
import uuid # Needed for event_id generation
from datetime import datetime
from cost_model.plan_rules.contributions import run as contrib_run
# Import the necessary Pydantic models for config
from cost_model.config.models import PlanRules, EmployerMatchRules, MatchTier
# Import constants and schema info from event_log (assuming it's needed for assertions)
try:
    from cost_model.state.event_log import EVT_CONTRIB, EVT_ENROLL, EVENT_COLS, EVENT_PANDAS_DTYPES, EMP_ID
except ImportError:
    # Fallbacks for testing if needed
    EVT_CONTRIB, EVT_ENROLL = 'contribution', 'enrollment'
    EMP_ID = 'employee_id'


@pytest.fixture
def base_snapshot():
    return pd.DataFrame({
        EMP_ID: ['E1','E2','E3'], # Use EMP_ID constant
        'employee_gross_compensation': [100_000.0, 50_000.0, 200_000.0], # Use floats
        'employee_deferral_rate': [0.05, 0.10, 0.00],
        # Add other columns needed by snapshot.py if run standalone
        'hire_date': pd.to_datetime(['2020-01-01', '2021-01-01', '2022-01-01']),
        'birth_date': pd.to_datetime(['1980-01-01', '1990-01-01', '1985-01-01']),
        'role': ['Staff', 'Staff', 'Staff'],
        'term_date': pd.NaT,
        'active': True
    }).set_index(EMP_ID).astype({'active': pd.BooleanDtype(), 'role': pd.StringDtype()}) # Set index and dtypes


@pytest.fixture
def base_events():
    # Ensure events have columns matching EVENT_COLS schema expectation
    events_list = [
        {'event_id': 'evt_e1', 'event_time': datetime(2025,1,1), EMP_ID: 'E1', 'event_type': EVT_ENROLL, 'value_num': None, 'value_json': None, 'meta': None},
        {'event_id': 'evt_e2', 'event_time': datetime(2025,1,1), EMP_ID: 'E2', 'event_type': EVT_ENROLL, 'value_num': None, 'value_json': None, 'meta': None},
    ]
    df = pd.DataFrame(events_list)
    # Optionally set dtypes if event_log module not available here
    # df = df.astype(...)
    return df


def test_contrib_emitted_and_calculated_correctly(base_snapshot, base_events):
    # --- Correctly create nested config using Pydantic models ---
    match_config = EmployerMatchRules(
        tiers=[
            MatchTier(match_rate=1.0, cap_deferral_pct=0.01),
            MatchTier(match_rate=0.5, cap_deferral_pct=0.06)
        ]
        # dollar_cap=None # Optional
    )
    # Assuming contrib_run expects the EmployerMatchRules part directly:
    cfg_to_pass = match_config
    # OR If contrib_run expects the whole PlanRules object:
    # plan_rules_cfg = PlanRules(employer_match=match_config)
    # cfg_to_pass = plan_rules_cfg
    # --- End of config creation ---

    events_out_df = contrib_run(
        snapshot=base_snapshot,
        events=base_events,
        as_of=pd.Timestamp('2025-01-01'),
        cfg=cfg_to_pass # Pass the correctly structured config object
    )

    assert not events_out_df.empty, "Expected contribution events were not generated."
    assert set(events_out_df[EMP_ID]) == {'E1', 'E2'}

    # Calculate expected match using CORRECT tiered logic
    # E1: 100k, 5% -> ee=5k
    expected_er_e1 = (min(5000, 100000 * 0.01) * 1.0) + \
                     (max(0, min(5000, 100000 * 0.06) - (100000 * 0.01)) * 0.5)
                     # = (min(5k, 1k)*1.0) + (max(0, min(5k, 6k) - 1k)*0.5)
                     # = (1k * 1.0) + (max(0, 5k - 1k)*0.5) = 1k + (4k*0.5) = 1k + 2k = 3000

    # E2: 50k, 10% -> ee=5k
    expected_er_e2 = (min(5000, 50000 * 0.01) * 1.0) + \
                     (max(0, min(5000, 50000 * 0.06) - (50000 * 0.01)) * 0.5)
                     # = (min(5k, 500)*1.0) + (max(0, min(5k, 3k) - 500)*0.5)
                     # = (500 * 1.0) + (max(0, 3k - 500)*0.5) = 500 + (2.5k*0.5) = 500 + 1250 = 1750

    row1 = events_out_df[events_out_df[EMP_ID]=='E1'].iloc[0]
    val1 = json.loads(row1['value_json']) # Parse JSON string
    assert pytest.approx(val1['employee_deferral']) == 5000
    assert pytest.approx(val1['employer_match']) == expected_er_e1 # Expect 3000

    row2 = events_out_df[events_out_df[EMP_ID]=='E2'].iloc[0]
    val2 = json.loads(row2['value_json'])
    assert pytest.approx(val2['employee_deferral']) == 5000
    assert pytest.approx(val2['employer_match']) == expected_er_e2 # Expect 1750


def test_no_duplicate_if_already_contributed(base_snapshot, base_events):
    prior = base_events.copy()
    # Add a valid prior contribution event matching the schema
    new_row = pd.DataFrame([{
      "event_id": str(uuid.uuid4()), # Add ID
      'event_time': pd.Timestamp('2025-01-01'),
      EMP_ID: 'E1', # Use constant
      'event_type': EVT_CONTRIB,
      'value_num': None, # Use correct schema columns
      'value_json': json.dumps({'employee_deferral': 5000, 'employer_match': 3000}), # Use JSON string & corrected match
      'meta': None
    }])
    # Ensure dtypes match before concat if needed
    # new_row = new_row[event_log.EVENT_COLS].astype(event_log.EVENT_PANDAS_DTYPES) # If needed
    prior = pd.concat([prior, new_row], ignore_index=True)

    # Create config (empty tiers is fine for this test's purpose)
    cfg_match_empty = EmployerMatchRules(tiers=[])
    cfg_to_pass = cfg_match_empty # Assuming run expects EmployerMatchRules

    out_df = contrib_run(
        snapshot=base_snapshot,
        events=prior,
        as_of=pd.Timestamp('2025-01-01'),
        cfg=cfg_to_pass
    )

    # E1 already contributed today, E2 has positive deferral but 0 match -> expect event for E2
    assert len(out_df) == 1, "Expected exactly one event (for E2)"
    assert out_df.iloc[0][EMP_ID] == 'E2'
    # Check E2 payload is correct (ee=5k, er=0)
    val_e2 = json.loads(out_df.iloc[0]['value_json'])
    assert pytest.approx(val_e2['employee_deferral']) == 5000
    assert pytest.approx(val_e2['employer_match']) == 0.0