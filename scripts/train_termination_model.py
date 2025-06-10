#!/usr/bin/env python3
"""
scripts/train_termination_model.py

CLI‐driven training of a termination‐prediction model:
1. Load multiple census CSVs.
2. Create a next‐year termination target.
3. Compute vectorized age/tenure features.
4. Clean & prepare data.
5. Time‐based train/test split.
6. Build a preprocessing+LightGBM pipeline.
7. Evaluate and save model + feature names.
"""
import argparse
import glob
import logging
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Default constants (overridden by CLI)
ID_COLUMN = "ssn"
DATE_COLUMNS = ["birth_date", "hire_date", "termination_date"]
TARGET_VARIABLE = "terminated_next_year"
CATEGORICAL_FEATURES = []
NUMERICAL_FEATURES = [
    "age",
    "tenure",
    "gross_compensation",
    "pre_tax_deferral_percentage",
]
FEATURES_TO_USE = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
MODEL_OUTPUT_FILENAME = "termination_model_pipeline.joblib"
FEATURE_NAMES_OUTPUT_FILENAME = "termination_model_features.joblib"
RANDOM_STATE = 42  # reproducibility


def load_all_census_data(data_dir: Path, pattern: str, date_cols: list) -> pd.DataFrame:
    files = sorted(glob.glob(str(data_dir / pattern)))
    if not files:
        logging.error("No files matching %s in %s", pattern, data_dir)
        sys.exit(1)

    df_list = []
    logging.info("Loading %d census files", len(files))
    for f in files:
        year = int(Path(f).stem.split("_")[-1])
        df = pd.read_csv(f, parse_dates=date_cols, low_memory=False)
        df["year"] = year
        df["census_date"] = pd.to_datetime(f"{year}-12-31")
        df_list.append(df)
    df_list = [df for df in df_list if not df.empty]
    if df_list:
        combined = pd.concat(df_list, ignore_index=True)
    else:
        combined = pd.DataFrame()
    logging.info("Combined data shape: %s", combined.shape)
    return combined


def add_termination_target(
    df: pd.DataFrame, id_col: str, year_col: str, target: str
) -> pd.DataFrame:
    unique_years = sorted(df[year_col].unique())
    if len(unique_years) < 2:
        logging.error("Need at least 2 distinct years to build target; got %s", unique_years)
        sys.exit(1)

    dfs = []
    for i, yr in enumerate(unique_years[:-1]):
        next_yr = unique_years[i + 1]
        sub = df[df[year_col] == yr].copy()
        ids_next = set(df[df[year_col] == next_yr][id_col])
        sub[target] = (~sub[id_col].isin(ids_next)).astype(int)
        dfs.append(sub)
        logging.info("Year %d → %d: %d terminations", yr, next_yr, sub[target].sum())
    dfs = [df for df in dfs if not df.empty]
    if dfs:
        final = pd.concat(dfs, ignore_index=True)
    else:
        final = pd.DataFrame()
    logging.info("After target creation: %s", final.shape)
    return final


def calculate_features(
    df: pd.DataFrame, birth_col: str, hire_col: str, ref_col: str
) -> pd.DataFrame:
    ref = df[ref_col]
    # vectorized age
    df["age"] = (
        ref.dt.year
        - df[birth_col].dt.year
        - ((ref.dt.month, ref.dt.day) < (df[birth_col].dt.month, df[birth_col].dt.day))
    )
    # vectorized tenure
    df["tenure"] = (ref - df[hire_col]).dt.days / 365.25
    logging.info("Calculated age/tenure for %d rows", len(df))
    return df


def clean_and_prepare_data(df: pd.DataFrame, feature_cols: list, target: str) -> pd.DataFrame:
    keep = feature_cols + [target, "year"]
    dfm = df[keep].copy()
    dfm.dropna(subset=feature_cols + [target], inplace=True)
    for col in feature_cols:
        dfm[col] = pd.to_numeric(dfm[col], errors="coerce")
    dfm.dropna(subset=feature_cols, inplace=True)
    if dfm.empty:
        logging.error("No data after cleaning; check inputs/features")
        sys.exit(1)
    logging.info("Cleaned data shape: %s", dfm.shape)
    return dfm


def split_data(df: pd.DataFrame, target: str, year_col: str):
    max_year = df[year_col].max()
    train = df[df[year_col] < max_year]
    test = df[df[year_col] == max_year]
    if train.empty or test.empty:
        logging.error("Train/test split failed on year %d", max_year)
        sys.exit(1)
    logging.info("Train size %d; Test size %d", len(train), len(test))
    return train, test


def build_pipeline(num_cols: list, cat_cols: list):
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(sparse=False), cat_cols))
    pre = ColumnTransformer(transformers, remainder="passthrough")
    model = lgb.LGBMClassifier(random_state=RANDOM_STATE)
    pipe = Pipeline([("preprocessor", pre), ("classifier", model)])
    return pipe


def train_and_evaluate(pipe, X_tr, y_tr, X_te, y_te):
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, preds)
    logging.info("Test AUC: %.4f", auc)
    y_pred = (preds > 0.5).astype(int)
    logging.info("Classification report:\n%s", classification_report(y_te, y_pred))
    logging.info("Confusion matrix:\n%s", confusion_matrix(y_te, y_pred))
    return pipe, pipe.named_steps["preprocessor"].get_feature_names_out()


def save_pipeline(pipe, feat_names, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / MODEL_OUTPUT_FILENAME)
    joblib.dump(list(feat_names), out_dir / FEATURE_NAMES_OUTPUT_FILENAME)
    logging.info("Saved pipeline and %d feature names", len(feat_names))


def main():
    p = argparse.ArgumentParser(description="Train a termination prediction model.")
    p.add_argument("--data-dir", type=Path, default=Path("./termination_model_example"))
    p.add_argument("--pattern", type=str, default="example_census_*.csv")
    p.add_argument("--output-dir", type=Path, default=Path("."))
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    )

    # 1. Load & combine
    all_data = load_all_census_data(args.data_dir, args.pattern, DATE_COLUMNS)
    # 2. Target
    data_tgt = add_termination_target(all_data, ID_COLUMN, "year", TARGET_VARIABLE)
    # 3. Features
    data_feat = calculate_features(data_tgt, "birth_date", "hire_date", "census_date")
    # 4. Clean
    clean = clean_and_prepare_data(data_feat, FEATURES_TO_USE, TARGET_VARIABLE)
    # 5. Split
    train, test = split_data(clean, TARGET_VARIABLE, "year")
    X_tr, y_tr = train[FEATURES_TO_USE], train[TARGET_VARIABLE]
    X_te, y_te = test[FEATURES_TO_USE], test[TARGET_VARIABLE]
    # 6. Pipeline
    pipeline = build_pipeline(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
    # 7. Train & eval
    pipeline, feat_names = train_and_evaluate(pipeline, X_tr, y_tr, X_te, y_te)
    # 8. Save
    save_pipeline(pipeline, feat_names, args.output_dir)

    logging.info("Training complete. Artifacts in %s", args.output_dir)


if __name__ == "__main__":
    main()
