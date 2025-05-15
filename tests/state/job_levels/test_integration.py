import pandas as pd
import pytest
from cost_model.state.job_levels import assign_levels_to_dataframe
from cost_model.state.job_levels.engine import infer_job_level_by_percentile
from cost_model.utils.columns import EMP_GROSS_COMP, EMP_LEVEL


def test_full_pipeline_sets_source():
    """Verify that job_level_source is set correctly throughout the full pipeline."""
    # Given a range of salaries
    data = {"salary": [50000, 200000, 1_000_000]}
    df = pd.DataFrame(data)
    
    # 1) Band assignment
    df1 = assign_levels_to_dataframe(df.rename(columns={"salary": EMP_GROSS_COMP}))
    
    # 2) Imputation
    df2 = infer_job_level_by_percentile(df1, EMP_GROSS_COMP)
    
    # 3) Backfill
    df2[EMP_LEVEL] = df2[EMP_LEVEL].fillna(df2["imputed_level"])
    
    # 1) No missing sources
    assert df2["job_level_source"].notna().all()
    
    # 2) Only valid tags
    actual = set(df2["job_level_source"].unique())
    valid = {"salary-band", "percentile-impute"}
    assert actual <= valid

    # 3) Since assign_levels never leaves NaNs, static band should cover everything
    assert actual == {"salary-band"}, f"Expected only 'salary-band' here but got {actual}"
    
    # Verify that all assigned levels have correct source
    assigned_mask = df2[EMP_LEVEL].notna()
    assert (df2.loc[assigned_mask, "job_level_source"] == "salary-band").all()
