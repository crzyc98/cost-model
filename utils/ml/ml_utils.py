# utils/ml/ml_utils.py

import logging
from pathlib import Path
from typing import Protocol, Optional, Tuple, List
import joblib
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class MLModel(Protocol):
    """Any pipeline with .classes_ array and .predict_proba."""
    classes_: np.ndarray
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...

def try_load_ml_model(
    model_path: str,
    features_path: str
) -> Optional[Tuple[MLModel, List[str]]]:
    """
    Load a joblib pipeline and its feature‐list. Returns (model, feature_names) or None on error.
    """
    try:
        model = joblib.load(Path(model_path))
        feature_names = joblib.load(Path(features_path))
        if not isinstance(feature_names, list):
            raise ValueError("feature_names must be a list")
        logger.info("Loaded ML model with %d features", len(feature_names))
        return model, feature_names
    except Exception as e:
        logger.warning("ML load failed (%s / %s): %s", model_path, features_path, e)
        return None

def predict_turnover(
    df: pd.DataFrame,
    model: MLModel,
    feature_cols: List[str],
    as_of_date: pd.Timestamp,
    seed: Optional[int] = None
) -> pd.Series:
    """
    Given df and a fitted model, return a Series of P(terminate) for class==1, indexed by df.index.
    """
    if seed is not None:
        np.random.seed(seed)
    X = df[feature_cols].copy()
    # median-impute missing
    for c in feature_cols:
        if X[c].isna().any():
            X[c].fillna(X[c].median(), inplace=True)
    probs = model.predict_proba(X)
    idx1 = int(np.where(model.classes_ == 1)[0][0])
    return pd.Series(probs[:, idx1], index=df.index)