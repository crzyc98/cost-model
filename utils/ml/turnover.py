import pandas as pd
import numpy as np

def apply_ml_turnover(
    df: pd.DataFrame,
    model,
    feature_cols: list,
    year_end: pd.Timestamp,
    seed: int = None
) -> pd.DataFrame:
    """
    Encapsulate ML turnover prediction, imputation fallback, and stochastic termination.

    Args:
        df: DataFrame with features and 'hire_date'.
        model: Trained ML model with predict_proba and classes_.
        feature_cols: List of column names for prediction.
        year_end: Timestamp marking end of period.
        seed: Optional random seed.

    Returns:
        DataFrame copy with updated 'termination_date' and 'status' for ML-driven terminations.
    """
    df = df.copy()
    X = df[feature_cols].copy()
    # Simple median imputation
    if X.isnull().any().any():
        X = X.fillna(X.median())
    # Predict probabilities
    probs = model.predict_proba(X)
    class_idx = np.where(model.classes_ == 1)[0][0]
    term_probs = probs[:, class_idx]
    # Stochastic decisions
    if seed is not None:
        np.random.seed(seed)
    decisions = np.random.rand(len(df)) < term_probs
    # Assign termination dates uniformly
    days_until_end = (year_end - df['hire_date']).dt.days.clip(lower=0) + 1
    rand_off = np.floor(np.random.rand(len(df)) * days_until_end.values).astype(int)
    rand_td = pd.to_timedelta(rand_off, unit='D')
    df.loc[decisions, 'termination_date'] = df.loc[decisions, 'hire_date'] + rand_td[decisions]
    df.loc[decisions, 'status'] = 'Terminated'
    return df
