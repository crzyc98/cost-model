from typing import Sequence
import pandas as pd


def infer_job_level_by_percentile(
    df: pd.DataFrame,
    salary_col: str,
    level_percentiles: Sequence[float] = (0.20, 0.50, 0.80, 0.95)
) -> pd.DataFrame:
    """
    Impute job_level for rows where it's missing using global compensation percentiles.

    - `level_percentiles` defines the cut-points between levels (e.g., [P20, P50, P80, P95]).
    - For edge cases (0 or 1 row), returns level 0.
    - For small datasets, uses quantile interpolation to maintain consistency.

    Returns a DataFrame with a new column `imputed_level` indicating assigned levels based
    purely on compensation percentiles, and a job_level_source column for audit trail.
    """
    n = len(df)
    if n == 0:
        out = df.copy()
        out["imputed_level"] = pd.Series(dtype="Int64")
        out["job_level_source"] = pd.Series(dtype="category")
        return out

    if n == 1:
        out = df.copy()
        out["imputed_level"] = 0
        # If it didn’t already have a source, mark it as imputed
        if "job_level_source" in out.columns and pd.notna(out["job_level_source"]).any():
            pass  # preserve existing
        else:
            out["job_level_source"] = "percentile-impute"
        return out

    out = df.copy()
    out["_pct"] = out[salary_col].rank(pct=True)
    bins = [0.0, *level_percentiles, 1.0]
    labels = list(range(len(bins) - 1))
    out["imputed_level"] = pd.cut(
        out["_pct"], bins=bins, labels=labels, include_lowest=True
    ).astype(int)

    # Ensure the source column exists
    if "job_level_source" not in out.columns:
        out["job_level_source"] = pd.Series(dtype="category", index=out.index)

    # Only tag rows where the original level is missing
    level_col = None
    for candidate in ["job_level", "level_id", "employee_level"]:
        if candidate in out.columns:
            level_col = candidate
            break
    if level_col is None:
        # fallback: tag everything
        out["job_level_source"] = "percentile-impute"
    else:
        missing_mask = out[level_col].isna()
        out.loc[missing_mask, "job_level_source"] = "percentile-impute"
    return out.drop(columns=["_pct"])


