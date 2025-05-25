"""
Tests for promotion matrix validation and application.

These tests verify that the promotion matrix validation works correctly and that
promotions/terminations are applied as expected.
"""
import pytest
import numpy as np
import pandas as pd
import logging
from unittest.mock import patch, MagicMock

# Import the functions we want to test
from cost_model.state.job_levels.sampling import (
    validate_promotion_matrix,
    apply_promotion_markov
)

# Set up test logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Test data
VALID_MATRIX = pd.DataFrame({
    0:     [0.80, 0.15, 0.00, 0.00, 0.00],
    1:     [0.00, 0.83, 0.12, 0.00, 0.00],
    2:     [0.00, 0.00, 0.84, 0.10, 0.00],
    3:     [0.00, 0.00, 0.00, 0.85, 0.08],
    4:     [0.00, 0.00, 0.00, 0.00, 0.92],
    "exit": [0.20, 0.02, 0.04, 0.05, 0.00]  # Rows sum to 1.0
}, index=[0, 1, 2, 3, 4])

INVALID_MATRIX = pd.DataFrame({
    0: [0.5, 0.1],
    1: [0.4, 0.9],
    "exit": [0.1, 0.1]  # Rows don't sum to 1.0
}, index=[0, 1])

def test_validate_promotion_matrix_valid():
    """Test that a valid promotion matrix passes validation."""
    # This should not raise an exception
    validate_promotion_matrix(VALID_MATRIX)

def test_validate_promotion_matrix_none():
    """Test that None matrix raises ValueError."""
    with pytest.raises(ValueError, match="cannot be None"):
        validate_promotion_matrix(None)

def test_validate_promotion_matrix_not_dataframe():
    """Test that non-DataFrame raises ValueError."""
    with pytest.raises(ValueError, match="Expected promotion matrix to be a pandas DataFrame"):
        validate_promotion_matrix([[0.5, 0.5]])

def test_validate_promotion_matrix_empty():
    """Test that empty DataFrame raises ValueError."""
    with pytest.raises(ValueError, match="is empty"):
        validate_promotion_matrix(pd.DataFrame())

def test_validate_promotion_matrix_invalid_sums():
    """Test that matrix with rows not summing to 1.0 raises ValueError."""
    with pytest.raises(ValueError, match="Rows must sum to 1.0"):
        validate_promotion_matrix(INVALID_MATRIX)

def test_validate_promotion_matrix_negative_prob():
    """Test that matrix with negative probabilities raises ValueError."""
    # Test with a negative probability
    bad_matrix = VALID_MATRIX.copy()
    bad_matrix.iloc[0, 0] = -0.1  # Set a negative probability
    with pytest.raises(ValueError, match="Invalid transition probability -0.1000 at level=0, next_state=0"):
        validate_promotion_matrix(bad_matrix)
    
    # Test with a probability > 1
    bad_matrix = VALID_MATRIX.copy()
    bad_matrix.iloc[0, 0] = 1.1  # Set a probability > 1
    with pytest.raises(ValueError, match="Invalid transition probability 1.1000 at level=0, next_state=0"):
        validate_promotion_matrix(bad_matrix)

def test_validate_promotion_matrix_prob_gt_one():
    """Test that matrix with probabilities > 1 raises ValueError."""
    bad_matrix = VALID_MATRIX.copy()
    bad_matrix.iloc[0, 0] = 1.1  # Set a probability > 1
    with pytest.raises(ValueError, match="Invalid transition probability 1.1000 at level=0, next_state=0"):
        validate_promotion_matrix(bad_matrix)

@patch('numpy.random.RandomState')
def test_apply_promotion_markov_happy_path(mock_random, caplog):
    """Test that promotions are applied correctly with valid input."""
    # Set up test data
    employees = pd.DataFrame({
        'employee_id': [1, 2, 3, 4],
        'employee_level': [0, 1, 2, 3],
        'employee_termination_date': pd.NaT
    })
    
    # Configure the mock random number generator
    # For these tests, we'll make the RNG return specific indices to test different paths
    mock_rng = MagicMock()
    mock_rng.choice.side_effect = [1, 2, 0, 3]  # Different next states for each employee
    mock_random.return_value = mock_rng
    
    # Run the function
    with caplog.at_level(logging.DEBUG):
        result = apply_promotion_markov(
            employees,
            level_col='employee_level',
            matrix=VALID_MATRIX,
            rng=mock_rng,
            simulation_year=2025,
            logger=logger
        )
    
    # Verify the results
    assert len(result) == 4
    assert 'exited' in result.columns
    assert 'employee_termination_date' in result.columns
    
    # Check that the RNG was called with the correct probabilities for each employee
    assert mock_rng.choice.call_count == 4
    
    # Check that the debug logs were generated
    assert "Processing 4 employees for promotion/termination" in caplog.text
    assert "Promotion/termination summary" in caplog.text

def test_apply_promotion_markov_missing_levels():
    """Test that employees with levels not in the matrix raise an error."""
    # Create test data with a level not in the matrix
    employees = pd.DataFrame({
        'employee_id': [1, 2],
        'employee_level': [0, 99],  # 99 is not in VALID_MATRIX
        'employee_termination_date': pd.NaT
    })

    # This should raise a ValueError because level 99 is not in the matrix
    with pytest.raises(ValueError, match="exist in data but not in the promotion matrix"):
        apply_promotion_markov(
            employees,
            level_col='employee_level',
            matrix=VALID_MATRIX,
            simulation_year=2025,
            logger=logger
        )
    
    # Test with a level that is in the matrix to ensure that works
    employees = pd.DataFrame({
        'employee_id': [1, 2],
        'employee_level': [0, 1],
        'employee_termination_date': pd.NaT
    })
    
    # This should work without raising an exception
    result = apply_promotion_markov(
        employees,
        level_col='employee_level',
        matrix=VALID_MATRIX,
        simulation_year=2025,
        logger=logger
    )
    
    # Should have the same number of employees
    assert len(result) == 2

@patch('numpy.random.RandomState')
def test_apply_promotion_markov_terminations(mock_random):
    """Test that terminations are handled correctly."""
    # Set up test data with employees that will terminate (next state is 'exit')
    employees = pd.DataFrame({
        'employee_id': [1, 2],
        'employee_level': [0, 1],
        'employee_termination_date': pd.NaT
    })
    
    # Configure the mock random number generator to always return 'exit' state
    mock_rng = MagicMock()
    mock_rng.choice.side_effect = [len(VALID_MATRIX.columns) - 1] * 2  # Index of 'exit' column
    mock_random.return_value = mock_rng
    
    # Run the function
    result = apply_promotion_markov(
        employees,
        level_col='employee_level',
        matrix=VALID_MATRIX,
        rng=mock_rng,
        simulation_year=2025,
        logger=logger
    )
    
    # Both employees should be marked as exited
    assert result['exited'].all()
    # Termination dates should be set to a date in 2025
    assert all(result['employee_termination_date'].dt.year == 2025)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
