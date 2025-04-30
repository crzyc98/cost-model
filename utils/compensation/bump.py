# utils/compensation/bump.py
import pandas as pd
import numpy as np

def apply_comp_increase(
    series: pd.Series,
    increase_rule,
    seed: int = None
) -> pd.Series:
    """
    Apply a compensation bump to a pandas Series.

    Args:
        series: Original compensation values.
        increase_rule: float for flat pct, or dict{'type':'normal','mean':..., 'std':...}.
        seed: Optional random seed.

    Returns:
        New pandas Series with bumped values.
    """
    if seed is not None:
        np.random.seed(seed)
    if isinstance(increase_rule, float):
        return series * (1 + increase_rule)
    elif isinstance(increase_rule, dict) and increase_rule.get('type')=='normal':
        bumps = np.random.normal(loc=increase_rule.get('mean',0), scale=increase_rule.get('std',0), size=len(series))
        return series * (1 + bumps)
    else:
        raise ValueError(f"Unsupported increase_rule: {increase_rule}")
