import pandas as pd
import pytest
from cost_model.state.job_levels import get_level_by_compensation, assign_levels_to_dataframe
from cost_model.utils.columns import EMP_GROSS_COMP, EMP_LEVEL

@pytest.fixture(autouse=True, scope="module")
def initialize_job_levels():
    """Fixture to ensure job levels are initialized for each test module.
    
    This fixture:
    1. Clears any existing global state
    2. Initializes job levels with defaults
    3. Builds compensation intervals
    """
    # Clear existing state and initialize with defaults
    from cost_model.state.job_levels import init_job_levels
    init_job_levels(reset_warnings=True)
    
    # Verify initialization
    from cost_model.state.job_levels import LEVEL_TAXONOMY, _COMP_INTERVALS
    assert LEVEL_TAXONOMY, "Job levels taxonomy should not be empty"
    assert _COMP_INTERVALS is not None, "Compensation intervals should be built"


def test_job_level_source_after_band_assignment():
    """Verify that job_level_source is set to 'salary-band' for valid assignments."""
    # Given two salaries—one in-range, one out-of-range
    df = pd.DataFrame({"salary": [40000, 1_000_000]})
    # Assign levels
    df2 = assign_levels_to_dataframe(df.rename(columns={"salary": EMP_GROSS_COMP}))
    
    # All non-null levels should have source "salary-band"
    srcs = df2["job_level_source"].dropna().unique()
    assert list(srcs) == ["salary-band"]
    
    # Verify that rows with assigned levels have the correct source
    assigned_mask = df2[EMP_LEVEL].notna()
    assert (df2.loc[assigned_mask, "job_level_source"] == "salary-band").all()

def test_get_level_by_compensation_edges():
    from cost_model.state.job_levels import get_level_by_compensation
    lvl_low  = get_level_by_compensation(0)
    lvl_high = get_level_by_compensation(1000000)
    assert lvl_low is not None
    assert lvl_high is not None

def test_assign_levels_to_dataframe(tmp_path):
    from cost_model.state.job_levels import assign_levels_to_dataframe
    df = pd.DataFrame({
        'compensation': [50000, 150000, 250000],
        'level_id': [None, None, None]
    })
    assign_levels_to_dataframe(df, comp_column='compensation')  # Use correct column name
    assert df['level_id'].notna().all()