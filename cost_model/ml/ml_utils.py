# utils/ml/ml_utils.py

import logging
from pathlib import Path
from typing import Protocol, Optional, Sequence, Tuple, Union

import joblib
import pandas as pd
import numpy as np
from sklearn.utils import check_random_state

logger = logging.getLogger(__name__)


class MLModel(Protocol):
    """Any model with a .classes_ attribute and .predict_proba method."""

    classes_: np.ndarray

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


def try_load_ml_model(
    model_path: Union[str, Path], features_path: Union[str, Path]
) -> Optional[Tuple[MLModel, list[str]]]:
    """
    Load a joblib pipeline and its feature‐list.
    Returns (model, feature_names) or None on error.
    """
    model_path = Path(model_path)
    features_path = Path(features_path)
    if not model_path.exists() or not features_path.exists():
        logger.warning("ML artifacts not found: %s, %s", model_path, features_path)
        return None

    try:
        model = joblib.load(model_path)
        feature_names = joblib.load(features_path)
    except (
        FileNotFoundError,
        joblib.externals.loky.process_executor.JoblibException,
    ) as e:
        logger.warning("Failed to load ML model or features: %s", e)
        return None

    if not isinstance(feature_names, Sequence):
        logger.error(
            "Expected feature_names to be a sequence, got %r", type(feature_names)
        )
        return None

    logger.info("Loaded ML model with %d features", len(feature_names))
    return model, list(feature_names)


def predict_turnover(
    df: pd.DataFrame,
    model: MLModel,
    feature_cols: Sequence[str],
    *,
    random_state: Union[int, np.random.RandomState, None] = None,
) -> pd.Series:
    """
    Given df and a fitted model, return P(terminate) for class==1, indexed by df.index.
    """
    # reproducible random state
    check_random_state(random_state)

    # Subset & median‐impute in one go
    X = df[feature_cols].copy()
    X = X.fillna(X.median())

    # Predict
    probs = model.predict_proba(X)

    # Find index of the ‘1’ class
    try:
        idx1 = int(np.where(model.classes_ == 1)[0][0])
    except IndexError:
        msg = f"Class 1 not found in model.classes_: {model.classes_}"
        logger.error(msg)
        raise ValueError(msg)

    return pd.Series(probs[:, idx1], index=df.index)
