"""Tests for Markov promotion and exit functionality."""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from cost_model.engines.markov_promotion import apply_markov_promotions, create_promotion_raise_events
from cost_model.state.schema import (
    EMP_ID, EMP_LEVEL, EMP_ROLE, EMP_EXITED, EMP_LEVEL_SOURCE, EMP_GROSS_COMP,
    EMP_TERMINATION_DATE, EMP_HIRE_DATE, EMP_AGE, EMP_GENDER, EMP_ETHNICITY
)
from cost_model.state.event_log import EVENT_COLS, EVT_TERM, EVT_PROMOTION, EVT_RAISE

# Sample promotion matrix for testing
TEST_PROMOTION_MATRIX = pd.DataFrame(
    data={
        1: [0.6, 0.2, 0.1, 0.1, 0.0],  # Level 1: 60% stay, 20% to 2, 10% to 3, 10% exit
        2: [0.0, 0.5, 0.3, 0.1, 0.1],  # Level 2: 50% stay, 30% to 3, 10% to 4, 10% exit
        3: [0.0, 0.0, 0.6, 0.3, 0.1],  # Level 3: 60% stay, 30% to 4, 10% exit
        4: [0.0, 0.0, 0.0, 0.9, 0.1],  # Level 4: 90% stay, 10% exit
        5: [0.0, 0.0, 0.0, 0.0, 1.0],  # Level 5: 100% stay (no exits from top level)
        'exit': [0.0, 0.0, 0.0, 0.0, 0.0]  # Exit row (not used in this format)
    },
    index=[1, 2, 3, 4, 5, 'exit']
)

# Sample promotion raise configuration
TEST_RAISE_CONFIG = {
    "1_to_2": 0.05,  # 5% raise for 1→2
    "2_to_3": 0.08,  # 8% raise for 2→3
    "3_to_4": 0.10,  # 10% raise for 3→4
    "4_to_5": 0.12,  # 12% raise for 4→5
    "default": 0.10  # Default 10% for any other promotions
}

@pytest.fixture
def sample_workforce():
    """Create a sample workforce DataFrame for testing."""
    np.random.seed(42)  # For reproducible tests
    
    # Create 100 employees at each level 1-4
    num_employees = 100
    data = []
    
    for level in [1, 2, 3, 4]:
        for i in range(num_employees):
            emp_id = f"{level}_{i+1:03d}"
            data.append({
                EMP_ID: emp_id,
                EMP_LEVEL: level,
                EMP_GROSS_COMP: 50000 + (level - 1) * 10000 + np.random.normal(0, 2000),
                EMP_ROLE: f"Role_{level}",
                EMP_LEVEL_SOURCE: 'hire',
                EMP_HIRE_DATE: pd.Timestamp('2020-01-01') - pd.Timedelta(days=np.random.randint(100, 1000)),
                EMP_AGE: 25 + level * 5 + np.random.normal(0, 3),
                EMP_GENDER: np.random.choice(['M', 'F', 'Other']),
                EMP_ETHNICITY: np.random.choice(['White', 'Black', 'Asian', 'Hispanic', 'Other']),
                'simulation_year': 2025
            })
    
    df = pd.DataFrame(data)
    df[EMP_TERMINATION_DATE] = pd.NaT
    df[EMP_EXITED] = False
    return df

def test_apply_markov_promotions_with_exits(sample_workforce):
    """Test that employees exit according to Markov transition probabilities."""
    # Create a fixed random state for reproducibility
    rng = np.random.RandomState(42)
    
    # Apply promotions with our test matrix
    promo_time = pd.Timestamp('2025-06-30')
    promotions, raises, exits = apply_markov_promotions(
        sample_workforce,
        promo_time=promo_time,
        rng=rng,
        promotion_raise_config=TEST_RAISE_CONFIG,
        simulation_year=2025
    )
    
    # Verify that the output DataFrames have the expected columns
    assert all(col in promotions.columns for col in EVENT_COLS)
    assert all(col in raises.columns for col in EVENT_COLS)
    
    # Check that exits have the expected structure
    assert not exits.empty, "Expected some employees to exit"
    assert all(col in exits.columns for col in [EMP_ID, EMP_TERMINATION_DATE, EMP_LEVEL])
    assert all(exits[EMP_EXITED] == True)
    
    # Check that termination dates are within the simulation year
    assert all((pd.Timestamp('2025-01-01') <= exits[EMP_TERMINATION_DATE]) & 
               (exits[EMP_TERMINATION_DATE] <= pd.Timestamp('2025-12-31')))
    
    # Check that the exit rate is roughly as expected (should be close to 10% for most levels)
    # Note: This is a statistical test and might occasionally fail due to randomness
    for level in [1, 2, 3, 4]:
        level_emps = sample_workforce[sample_workforce[EMP_LEVEL] == level]
        level_exits = exits[exits[EMP_LEVEL] == level]
        exit_rate = len(level_exits) / len(level_emps)
        expected_rate = TEST_PROMOTION_MATRIX[level]['exit']
        
        # Allow for some statistical variation
        assert abs(exit_rate - expected_rate) < 0.1, f"Unexpected exit rate for level {level}"

def test_promotion_raise_events():
    """Test that promotion and raise events are created correctly."""
    # Create a sample snapshot
    snapshot_data = {
        EMP_ID: ['emp1', 'emp2', 'emp3'],
        EMP_LEVEL: [1, 2, 3],
        EMP_GROSS_COMP: [50000, 60000, 70000],
        EMP_ROLE: ['Role1', 'Role2', 'Role3'],
        EMP_LEVEL_SOURCE: ['hire'] * 3,
        EMP_EXITED: [False, False, False]
    }
    snapshot = pd.DataFrame(snapshot_data).set_index(EMP_ID)
    
    # Create promoted employees (emp1: 1->2, emp2: 2->3, emp3: 3->4)
    promoted_data = {
        EMP_ID: ['emp1', 'emp2', 'emp3'],
        EMP_LEVEL: [2, 3, 4],
        EMP_EXITED: [False, False, False]
    }
    promoted = pd.DataFrame(promoted_data).set_index(EMP_ID)
    
    # Create promotion and raise events
    promo_time = pd.Timestamp('2025-06-30')
    promotions, raises = create_promotion_raise_events(
        snapshot, promoted, promo_time, TEST_RAISE_CONFIG
    )
    
    # Verify the promotion events
    assert len(promotions) == 3
    assert all(promotions['event_type'] == EVT_PROMOTION)
    
    # Verify the raise events
    assert len(raises) == 3
    assert all(raises['event_type'] == EVT_RAISE)
    
    # Verify the raise amounts are correct
    emp1_raise = float(raises[raises['employee_id'] == 'emp1']['value_json'].iloc[0]["raise_pct"])
    assert emp1_raise == 0.05  # 1->2 should be 5%
    
    emp2_raise = float(raises[raises['employee_id'] == 'emp2']['value_json'].iloc[0]["raise_pct"])
    assert emp2_raise == 0.08  # 2->3 should be 8%
    
    emp3_raise = float(raises[raises['employee_id'] == 'emp3']['value_json'].iloc[0]["raise_pct"])
    assert emp3_raise == 0.12  # 3->4 should be 12% (4->5 in config)

def test_no_promotions_with_empty_input():
    """Test that empty input returns empty DataFrames."""
    empty_df = pd.DataFrame(columns=[EMP_ID, EMP_LEVEL, EMP_GROSS_COMP, EMP_EXITED])
    promotions, raises, exits = apply_markov_promotions(
        empty_df,
        pd.Timestamp('2025-06-30'),
        rng=np.random.RandomState(42)
    )
    
    assert promotions.empty
    assert raises.empty
    assert exits.empty

def test_promotion_with_all_exits():
    """Test scenario where all employees exit."""
    # Create a test matrix where everyone exits
    exit_matrix = pd.DataFrame(
        data={
            1: [0.0, 0.0, 0.0, 0.0, 1.0],  # 100% exit
            2: [0.0, 0.0, 0.0, 0.0, 1.0],  # 100% exit
            3: [0.0, 0.0, 0.0, 0.0, 1.0],  # 100% exit
            4: [0.0, 0.0, 0.0, 0.0, 1.0],  # 100% exit
            5: [0.0, 0.0, 0.0, 0.0, 1.0],  # 100% exit
            'exit': [0.0, 0.0, 0.0, 0.0, 0.0]
        },
        index=[1, 2, 3, 4, 5, 'exit']
    )
    
    # Monkey patch the promotion matrix for this test
    from cost_model.state.job_levels import transitions
    original_matrix = transitions.PROMOTION_MATRIX
    transitions.PROMOTION_MATRIX = exit_matrix
    
    try:
        # Create a small sample workforce
        data = {
            EMP_ID: [f'emp{i}' for i in range(10)],
            EMP_LEVEL: [1] * 5 + [2] * 5,  # 5 at level 1, 5 at level 2
            EMP_GROSS_COMP: [50000] * 10,
            EMP_EXITED: [False] * 10,
            'simulation_year': 2025
        }
        df = pd.DataFrame(data).set_index(EMP_ID)
        
        # Apply promotions - should all exit
        promotions, raises, exits = apply_markov_promotions(
            df,
            pd.Timestamp('2025-06-30'),
            rng=np.random.RandomState(42),
            simulation_year=2025
        )
        
        # Should have no promotions or raises, but all employees exited
        assert promotions.empty
        assert raises.empty
        assert len(exits) == 10
        assert all(exits[EMP_EXITED] == True)
        
    finally:
        # Restore the original matrix
        transitions.PROMOTION_MATRIX = original_matrix
