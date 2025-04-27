import pandas as pd
import numpy as np

def sample_new_hire_compensation(
    df: pd.DataFrame,
    comp_col: str,
    prev_salaries: np.ndarray,
    seed: int = None
) -> pd.DataFrame:
    """
    Sample starting compensation for new hires by drawing from historical salaries.

    Args:
        df: DataFrame with new hire records.
        comp_col: Name of the compensation column to assign.
        prev_salaries: Array-like of past salaries to sample from.
        seed: Optional random seed for reproducibility.

    Returns:
        DataFrame with updated compensation in `comp_col`.
    """
    df = df.copy()
    if seed is not None:
        np.random.seed(seed)
    df[comp_col] = np.random.choice(prev_salaries, size=len(df), replace=True)
    return df
