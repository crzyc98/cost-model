# tests/state/job_levels/test_state_job_levels_stub.py
import pytest
import cost_model.state.job_levels as jl

def test_public_api_exports():
    """Ensure every name in __all__ is actually exported by the stub."""
    for name in jl.__all__:
        assert hasattr(jl, name), f"Missing public API: {name}"

@pytest.mark.parametrize("fn", [
    "init_job_levels",
    "refresh_job_levels",
    "get_level_by_compensation",
    "assign_levels_to_dataframe",
    "get_level_distribution",
    "get_warning_counts",
    "sample_new_hire_compensation",
    "sample_new_hires_vectorized",
    "sample_mixed_new_hires",
    "JobLevel",
    "ConfigError",
    "load_from_yaml",
])
def test_api_callable_or_class(fn):
    """Basic sanity check that each API symbol is callable or a class."""
    obj = getattr(jl, fn)
    # Classes should be callable, functions too; just ensure no AttributeError
    assert callable(obj), f"{fn} should be callable"

def test_init_and_basic_usage():
    """Integration smoke test: init job levels and do a basic comp lookup."""
    import logging
    import sys
    
    # Configure logging to capture debug messages
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    
    # Initialize using the stub
    success = jl.init_job_levels()
    assert success, "Initialization should succeed"
    
    # Check that levels exist and are properly initialized
    assert jl.LEVEL_TAXONOMY, "Level taxonomy should not be empty"
    assert len(jl.LEVEL_TAXONOMY) > 0, "Should have at least one level"
    
    # Check intervals
    assert jl._COMP_INTERVALS is not None, "Compensation intervals should be built"
    assert len(jl._COMP_INTERVALS) > 0, "Should have at least one interval"
    
    # Clean up logging
    logger.removeHandler(handler)
    
    # Get lowest level
    min_level = min(jl.LEVEL_TAXONOMY.keys())
    assert min_level >= 0, "Minimum level ID should be non-negative"
    
    # Test compensation lookup
    lvl = jl.get_level_by_compensation(0.0)
    assert lvl is not None, "Should get a level for minimum compensation"
    assert lvl.level_id == min_level, "Minimum compensation should map to minimum level"
    
    # Test with a compensation value in the middle of the range
    middle_comp = (jl.LEVEL_TAXONOMY[min_level].min_compensation + 
                  jl.LEVEL_TAXONOMY[min_level].max_compensation) / 2
    lvl = jl.get_level_by_compensation(middle_comp)
    assert lvl is not None, "Should get a level for middle compensation"
    assert lvl.level_id == min_level, "Middle compensation should map to minimum level"
    
    # Test with a very high compensation
    max_level = max(jl.LEVEL_TAXONOMY.keys())
    high_comp = jl.LEVEL_TAXONOMY[max_level].max_compensation * 2
    lvl = jl.get_level_by_compensation(high_comp)
    assert lvl is not None, "Should get a level for high compensation"
    assert lvl.level_id == max_level, "High compensation should map to maximum level"