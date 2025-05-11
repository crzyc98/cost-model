# utils/ml_logic.py

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Sequence, Dict

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb

from utils.data_processing import load_and_clean_census
from utils.date_utils import calculate_age, calculate_tenure
from utils.columns import EMP_SSN, EMP_BIRTH_DATE, EMP_HIRE_DATE, EMP_TERM_DATE
from utils.columns import GROSS_COMP

logger = logging.getLogger(__name__)


def build_training_data(
    historical_files: Sequence[Path],
    feature_names: List[str],
    rng_seed: Optional[int] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Build turnover training set (X, y) from a time-ordered list of census files.
    """
    logger.info("Building training data for turnover model")
    all_periods = []  # List[pd.DataFrame]

    # ensure reproducibility for any sampling we do here
    np.random.default_rng(rng_seed)

    # sort by filename so year order is consistent
    historical_files = sorted(historical_files)

    prev_df = None
    prev_date = None

    for filepath in historical_files:
        df = load_and_clean_census(
            str(filepath),
            {"required": [EMP_SSN, EMP_BIRTH_DATE, EMP_HIRE_DATE, GROSS_COMP]},
        )
        if df is None:
            logger.warning(f"Skipping {filepath.name}: load failed")
            continue

        period_end = df["plan_year_end_date"].iloc[0]
        logger.debug("Period end date for %s: %s", filepath.name, period_end)

        # feature engineering at period start
        df["age_start"] = calculate_age(df[EMP_BIRTH_DATE], period_end)
        df["tenure_start"] = calculate_tenure(df[EMP_HIRE_DATE], period_end)
        df["comp_start"] = df[GROSS_COMP]

        # select only the model’s features + key
        cols = [EMP_SSN] + [f for f in feature_names if f in df.columns]
        feats = df[cols].copy()

        if prev_df is not None:
            # who survived into this period?
            active_prev = prev_df[EMP_TERM_DATE].isna() | (
                prev_df[EMP_TERM_DATE] > prev_date
            )
            active_ssns = set(prev_df.loc[active_prev, EMP_SSN])

            # who terminated in the window (prev_date, period_end]?
            terminated = (
                df[EMP_TERM_DATE].notna()
                & (df[EMP_TERM_DATE] > prev_date)
                & (df[EMP_TERM_DATE] <= period_end)
            )
            term_ssns = set(df.loc[terminated, EMP_SSN])

            # build training rows
            mask = feats[EMP_SSN].isin(active_ssns)
            period_data = feats.loc[mask].copy()
            period_data["y"] = period_data[EMP_SSN].apply(lambda s: int(s in term_ssns))

            # drop any rows missing feature columns
            before = len(period_data)
            period_data.dropna(
                subset=[c for c in feature_names if c != EMP_SSN], inplace=True
            )
            dropped = before - len(period_data)
            if dropped:
                logger.debug(
                    "Dropped %d rows with missing features in %s",
                    dropped,
                    filepath.name,
                )

            all_periods.append(period_data)

        prev_df = df
        prev_date = period_end

    if not all_periods:
        logger.error("No training data assembled!")
        return None, None

    training = pd.concat(all_periods, ignore_index=True)
    X = training[feature_names].copy()
    y = training["y"].copy().astype(int)

    logger.info(
        "Training data shape: %s, turnover rate: %.2f%%", X.shape, y.mean() * 100
    )
    return X, y


def train_turnover_model(
    X: pd.DataFrame, y: pd.Series, rng_seed: Optional[int] = None
) -> Optional[Pipeline]:
    """
    Train a LightGBM pipeline for turnover.
    """
    if X is None or y is None or X.empty or y.empty:
        logger.error("Empty training data; cannot train turnover model.")
        return None

    # drop SSN if accidentally present
    if EMP_SSN in X.columns:
        logger.warning("Dropping SSN column from features before training")
        X = X.drop(columns=[EMP_SSN])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=rng_seed, stratify=y
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", lgb.LGBMClassifier(random_state=rng_seed)),
        ]
    )

    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    logger.info("Validation AUC: %.4f", auc)

    logger.debug(
        "Classification report:\n%s", classification_report(y_val, model.predict(X_val))
    )
    return model


def prepare_features_for_prediction(
    df: pd.DataFrame, feature_names: List[str], reference_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Recompute exactly the same features you used in training.
    """
    logger.info("Preparing features for prediction as of %s", reference_date.date())
    feats: Dict[str, pd.Series] = {}

    if "age_start" in feature_names:
        feats["age_start"] = calculate_age(df[EMP_BIRTH_DATE], reference_date)

    from cost_model.utils.columns import EMP_TENURE
    if "tenure_start" in feature_names:
        if EMP_TENURE in df.columns:
            feats["tenure_start"] = df[EMP_TENURE]
        else:
            feats["tenure_start"] = calculate_tenure(df[EMP_HIRE_DATE], reference_date)

    if "comp_start" in feature_names and GROSS_COMP in df.columns:
        feats["comp_start"] = df[GROSS_COMP]

    result = pd.DataFrame(feats, index=df.index)[feature_names]

    # impute any remaining NaNs
    for col in result:
        if result[col].isna().any():
            med = result[col].median()
            logger.debug(
                "Imputing %d NaNs in %s with %.2f", result[col].isna().sum(), col, med
            )
            result[col].fillna(med, inplace=True)

    return result


def apply_stochastic_termination(
    df: pd.DataFrame,
    turnover_probs: pd.Series,
    rng: Optional[np.random.Generator] = None,
) -> pd.Series:
    """
    Given a Series of p(terminate), return a Boolean mask of who actually terminates.
    """
    if rng is None:
        rng = np.random.default_rng()
    draws = rng.random(len(df))
    return draws < turnover_probs.values
