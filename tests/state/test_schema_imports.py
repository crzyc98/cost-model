"""Test that required schema constants are always available."""
import pytest
from cost_model.state import schema


def test_required_constants_are_present():
    """Ensure critical schema constants are always available."""
    assert hasattr(schema, "EMP_ID"), "EMP_ID is missing from cost_model.state.schema"
    assert hasattr(schema, "SIMULATION_YEAR"), "SIMULATION_YEAR is missing from cost_model.state.schema"
    
    # Test that the values are not None or empty strings
    assert schema.EMP_ID, "EMP_ID should not be empty"
    assert schema.SIMULATION_YEAR, "SIMULATION_YEAR should not be empty"
    
    # Test that the values are strings
    assert isinstance(schema.EMP_ID, str), "EMP_ID should be a string"
    assert isinstance(schema.SIMULATION_YEAR, str), "SIMULATION_YEAR should be a string"


def test_schema_constants_are_in_all():
    """Ensure critical constants are included in __all__."""
    assert "EMP_ID" in schema.__all__, "EMP_ID should be in __all__"
    assert "SIMULATION_YEAR" in schema.__all__, "SIMULATION_YEAR should be in __all__"
