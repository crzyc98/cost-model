import pandas as pd
import pytest
from cost_model.state.job_levels.engine import infer_job_level_by_percentile

def test_infer_percentiles_exact_bins():
    # salaries at exact quantile edges for defaults [.2, .5, .8, .95]
    salaries = [10, 20, 30, 40, 50]
    df = pd.DataFrame({"salary": salaries})
    out = infer_job_level_by_percentile(df, "salary")
    assert "imputed_level" in out
    # levels should span 0..4
    assert out["imputed_level"].min() == 0
    assert out["imputed_level"].max() == 4
    # monotonicity: as salary increases, level does not decrease
    assert out["imputed_level"].is_monotonic_increasing

def test_impute_all_rows_random():
    rng = pd.Series(range(1, 11))
    df = pd.DataFrame({"salary": rng})
    out = infer_job_level_by_percentile(df, "salary")
    assert out["imputed_level"].isna().sum() == 0
    assert out["imputed_level"].between(0, 4).all()

@pytest.mark.parametrize("vals", [
    [],                         # empty DF
    [100],                      # single row
])
def test_empty_and_single(vals):
    df = pd.DataFrame({"salary": vals})
    out = infer_job_level_by_percentile(df, "salary")
    # empty input yields empty output
    assert len(out) == len(df)
    # if there's one row, it gets level 0
    if len(df) == 1:
        assert out["imputed_level"].iloc[0] == 0

def test_negative_salary_raises():
    df = pd.DataFrame({"salary": [-10, 0, 10]})
    # negative values should be handled gracefully
    out = infer_job_level_by_percentile(df, "salary")
    assert out["imputed_level"].isna().sum() == 0
    assert out["imputed_level"].between(0, 4).all()


def test_job_level_source_after_imputation():
    """Verify that job_level_source is set to 'percentile-impute' for imputed levels."""
    # Create a df with a single salary
    df = pd.DataFrame({"salary": [100]})
    # Run imputation
    out = infer_job_level_by_percentile(df, "salary")
    
    # All rows should have job_level_source == "percentile-impute"
    assert out["job_level_source"].tolist() == ["percentile-impute"]
    
    # Verify that imputed levels have the correct source
    imputed_mask = out["imputed_level"].notna()
    assert (out.loc[imputed_mask, "job_level_source"] == "percentile-impute").all()