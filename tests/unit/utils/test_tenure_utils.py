"""
Tests for tenure_utils standardization functions.
"""
import pandas as pd
import pytest
from cost_model.utils.tenure_utils import standardize_tenure_band


def test_standardize_tenure_band():
    """Test the standardize_tenure_band function with various input formats."""
    # Test standard formats (should remain unchanged)
    assert standardize_tenure_band("0-1") == "0-1"
    assert standardize_tenure_band("1-3") == "1-3"
    assert standardize_tenure_band("3-5") == "3-5"
    assert standardize_tenure_band("5+") == "5+"
    
    # Test non-standard formats (should be standardized)
    assert standardize_tenure_band("<1") == "0-1"
    assert standardize_tenure_band("0-1yr") == "0-1"
    assert standardize_tenure_band("0-1 years") == "0-1"
    assert standardize_tenure_band("1-3yr") == "1-3"
    assert standardize_tenure_band("3-5 years") == "3-5"
    assert standardize_tenure_band(">5") == "5+"
    assert standardize_tenure_band("5+yr") == "5+"
    assert standardize_tenure_band("3+") == "5+"  # Old format to new format
    
    # Test numeric inputs
    assert standardize_tenure_band(0.5) == "0-1"
    assert standardize_tenure_band(2.0) == "1-3"
    assert standardize_tenure_band(4.0) == "3-5"
    assert standardize_tenure_band(6.0) == "5+"
    
    # Test NA values
    assert pd.isna(standardize_tenure_band(pd.NA))
    assert pd.isna(standardize_tenure_band(None))
