import pytest
from cost_model.state.job_levels.models import JobLevel, ConfigError

def test_midpoint_auto_calculation():
    lvl = JobLevel(
        level_id=0, name="Test", description="",
        min_compensation=50, max_compensation=150,
        comp_base_salary=50, comp_age_factor=0, comp_stochastic_std_dev=0.1
    )
    assert lvl.mid_compensation == pytest.approx(100)

def test_no_overlap_assertion():
    with pytest.raises(AssertionError):
        JobLevel(
            level_id=1, name="Bad", description="",
            min_compensation=200, max_compensation=100,
            comp_base_salary=150, comp_age_factor=0, comp_stochastic_std_dev=0.1
        )

def test_compa_ratio():
    lvl = JobLevel(
        level_id=2, name="Ratio", description="",
        min_compensation=0, max_compensation=100,
        comp_base_salary=0, comp_age_factor=0, comp_stochastic_std_dev=0.1,
        mid_compensation=50
    )
    assert lvl.calculate_compa_ratio(75) == pytest.approx(1.5)
    assert lvl.calculate_compa_ratio(0) == pytest.approx(0)