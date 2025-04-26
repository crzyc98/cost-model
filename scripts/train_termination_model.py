# train_termination_model.py
import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib
import argparse
import warnings
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore', category=FutureWarning) # Suppress specific FutureWarnings from sklearn/pandas interactions

# --- Configuration ---
DATA_DIR = './termination_model_example' # Updated path
FILE_PATTERN = 'example_census_*.csv' # Updated pattern
ID_COLUMN = 'ssn'
DATE_COLUMNS = ['birth_date', 'hire_date', 'termination_date']
TARGET_VARIABLE = 'terminated_next_year'
CATEGORICAL_FEATURES = [] # No categorical features used now
# Use gross_compensation as the primary comp feature for simplicity first
NUMERICAL_FEATURES = ['age', 'tenure', 'gross_compensation', 'pre_tax_deferral_percentage']
FEATURES_TO_USE = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
MODEL_OUTPUT_FILENAME = 'termination_model_pipeline.joblib'
FEATURE_NAMES_OUTPUT_FILENAME = 'termination_model_features.joblib'
RANDOM_STATE = 42 # for reproducibility

# --- Helper Functions ---

def calculate_age(birth_date, reference_date):
    """Calculates age as of a reference date."""
    # Ensure dates are datetime objects
    birth_date = pd.to_datetime(birth_date)
    reference_date = pd.to_datetime(reference_date)
    age = reference_date.year - birth_date.year - ((reference_date.month, reference_date.day) < (birth_date.month, birth_date.day))
    return age

def calculate_tenure(hire_date, reference_date):
    """Calculates tenure in years as of a reference date."""
    # Ensure dates are datetime objects
    hire_date = pd.to_datetime(hire_date)
    reference_date = pd.to_datetime(reference_date)
    tenure = (reference_date - hire_date).days / 365.25
    return tenure

def calculate_model_features(df, birth_date_col, hire_date_col, census_date_col):
    """Calculates age and tenure features."""
    print("Calculating features (age, tenure)...")
    # Ensure census_date is datetime
    df[census_date_col] = pd.to_datetime(df[census_date_col])

    # Apply age calculation
    df['age'] = df.apply(lambda row: calculate_age(row[birth_date_col], row[census_date_col]), axis=1)

    # Apply tenure calculation
    df['tenure'] = df.apply(lambda row: calculate_tenure(row[hire_date_col], row[census_date_col]), axis=1)

    print("Features calculated.")
    return df

def clean_and_prepare_data(df, feature_cols, target_col):
    """Selects features, handles missing values, and ensures correct types."""
    print("Cleaning and preparing data...")
    cols_to_keep = feature_cols + [target_col, 'year'] # Keep 'year' for potential time-split
    df_model = df[cols_to_keep].copy()

    # Check for missing values before handling
    missing_before = df_model.isnull().sum()
    missing_features = missing_before[feature_cols].sum()
    missing_target = missing_before[target_col]
    print(f"  Missing values before handling:\n{missing_before[missing_before > 0]}")

    # Drop rows with missing values in features or target
    initial_rows = len(df_model)
    df_model.dropna(subset=feature_cols + [target_col], inplace=True)
    rows_after_na = len(df_model)
    dropped_rows = initial_rows - rows_after_na

    if dropped_rows > 0:
        print(f"  Dropped {dropped_rows} rows due to missing values in features or target.")

    # Ensure correct types (especially for numerical features after potential load issues)
    for col in NUMERICAL_FEATURES:
        if col in df_model.columns: # Check if column exists before conversion
             df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
    # Drop any rows that became NaN during numeric conversion
    rows_before_numeric_dropna = len(df_model)
    df_model.dropna(subset=NUMERICAL_FEATURES, inplace=True)
    if len(df_model) < rows_before_numeric_dropna:
         print(f"  Dropped {rows_before_numeric_dropna - len(df_model)} additional rows due to non-numeric values in numerical features.")

    print(f"Final model data shape: {df_model.shape}")
    if df_model.empty:
        raise ValueError("No data remaining after cleaning. Check input data quality and feature definitions.")

    return df_model

# --- Data Loading and Preparation ---

def load_all_census_data(data_dir, file_pattern, date_cols):
    """Loads all census CSV files from the specified directory."""
    all_files = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
    if not all_files:
        raise FileNotFoundError(f"No files found matching pattern '{file_pattern}' in directory '{data_dir}'")

    df_list = []
    print(f"Found {len(all_files)} census files: {', '.join(os.path.basename(f) for f in all_files)}")
    for f in all_files:
        try:
            year = int(os.path.basename(f).split('_')[-1].split('.')[0])
            df = pd.read_csv(f, parse_dates=date_cols, low_memory=False)
            df['year'] = year
            df['census_date'] = pd.to_datetime(f'{year}-12-31') # Reference date for age/tenure
            df_list.append(df)
            print(f"  Loaded {os.path.basename(f)} ({len(df)} rows)")
        except Exception as e:
            print(f"  Error loading {os.path.basename(f)}: {e}")
            continue # Skip problematic files

    if not df_list:
        raise ValueError("No data loaded successfully.")

    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Combined data shape: {combined_df.shape}")
    return combined_df

def add_termination_target(df, id_col, year_col, target_var):
    """Adds the binary target variable indicating termination in the following year."""
    print("Creating target variable...")
    df = df.sort_values(by=[id_col, year_col])
    years = sorted(df[year_col].unique())
    target_dfs = []

    for i, year_n in enumerate(years):
        if i == len(years) - 1: # Cannot determine target for the last year
            print(f"  Skipping target creation for the last year: {year_n}")
            continue

        year_n_plus_1 = years[i+1]
        print(f"  Processing transition: {year_n} -> {year_n_plus_1}")

        df_n = df[df[year_col] == year_n].copy() # Use .copy() to avoid SettingWithCopyWarning
        ids_n_plus_1 = set(df[df[year_col] == year_n_plus_1][id_col])

        # Default to 0 (not terminated)
        df_n[target_var] = 0
        # Mark as 1 if ID is NOT present in the next year's census
        df_n.loc[~df_n[id_col].isin(ids_n_plus_1), target_var] = 1

        # Optional: Refine using termination_date if available and reliable
        # If termination_date is between year_n end and year_n_plus_1 end, mark as terminated (1)
        # This handles cases where someone might appear in the next census briefly before terming.
        # Requires careful handling of date ranges.
        # Example (needs testing based on exact date formats/logic):
        term_date_col = 'termination_date'
        if term_date_col in df_n.columns:
            # Convert termination_date to datetime, coercing errors to NaT
            df_n[term_date_col] = pd.to_datetime(df_n[term_date_col], errors='coerce')
            term_date_exists = pd.notna(df_n[term_date_col])
            year_n_end = pd.to_datetime(f"{year_n}-12-31")
            year_n_plus_1_end = pd.to_datetime(f"{year_n_plus_1}-12-31")
            terminated_in_period = (
                term_date_exists &
                (df_n[term_date_col] > year_n_end) &
                (df_n[term_date_col] <= year_n_plus_1_end)
            )
            df_n.loc[terminated_in_period, target_var] = 1


        terminated_count = df_n[target_var].sum()
        print(f"    Year {year_n}: {terminated_count} terminations identified out of {len(df_n)} employees.")
        target_dfs.append(df_n)

    if not target_dfs:
        raise ValueError("Could not create target variable. Check if multiple years of data were loaded.")

    final_df = pd.concat(target_dfs, ignore_index=True)
    print(f"Data shape after target creation (excluding last year): {final_df.shape}")
    print(f"Target variable distribution:\n{final_df[target_var].value_counts(normalize=True)}")
    return final_df

# --- Model Training Preparation ---

def split_data(df, target_col, year_col):
    """Splits data into training and testing sets based on the latest year."""
    print("Splitting data into train/test sets (time-based)...")
    latest_year = df[year_col].max()
    test_df = df[df[year_col] == latest_year]
    train_df = df[df[year_col] < latest_year]

    if train_df.empty or test_df.empty:
        raise ValueError(f"Could not perform time-based split. Need data from at least two distinct years. Max year found: {latest_year}")

    X_train = train_df.drop(columns=[target_col, year_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col, year_col])
    y_test = test_df[target_col]

    print(f"  Train set shape: X={X_train.shape}, y={y_train.shape} (Years < {latest_year})")
    print(f"  Test set shape : X={X_test.shape}, y={y_test.shape} (Year == {latest_year})")
    print(f"  Train target distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"  Test target distribution:\n{y_test.value_counts(normalize=True)}")

    # Check for class imbalance
    train_imbalance = y_train.mean()
    if train_imbalance < 0.1 or train_imbalance > 0.9:
         print(f"  WARNING: Potential class imbalance detected in training data (positive class ratio: {train_imbalance:.3f})")


    return X_train, X_test, y_train, y_test

def create_preprocessing_pipeline(numerical_cols, categorical_cols):
    """Creates a Scikit-learn pipeline for preprocessing features."""
    print("Creating preprocessing pipeline...")

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # Use sparse_output instead of sparse

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough' # Keep other columns if any (though we selected specific ones)
    )
    print("  Preprocessor steps: StandardScaler for numerical, OneHotEncoder for categorical.")
    return preprocessor


# --- Model Training and Evaluation ---

def train_and_evaluate_model(preprocessor, X_train, y_train, X_test, y_test):
    """Builds pipeline, trains default LightGBM, evaluates, and returns the pipeline + features."""

    # Calculate scale_pos_weight for LightGBM imbalance handling
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    if pos_count > 0:
        scale_pos_weight = neg_count / pos_count
        print(f"Calculated scale_pos_weight for imbalance: {scale_pos_weight:.2f}")
    else:
        print("Warning: No positive samples in training data for scale_pos_weight calculation. Setting to 1.")
        scale_pos_weight = 1

    # Define the default LightGBM model
    lgbm = lgb.LGBMClassifier(random_state=RANDOM_STATE, scale_pos_weight=scale_pos_weight)

    # Create the pipeline with preprocessor and classifier steps
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', lgbm)])

    print("\n--- Training Default LightGBM Model ---")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # Get feature names after preprocessing from the fitted pipeline
    feature_names_out = None
    try:
        feature_names_out = pipeline.named_steps['preprocessor'].get_feature_names_out()
        print(f"\nSuccessfully extracted {len(feature_names_out)} feature names after preprocessing.")
    except Exception as e:
        print(f"\nError getting feature names: {e}")
        feature_names_out = list(X_train.columns) # Fallback

    # Evaluate the model on the held-out test set
    print("\nEvaluating default LightGBM model performance on the test set...")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, target_names=['Not Terminated (0)', 'Terminated (1)'])
    matrix = confusion_matrix(y_test, y_pred)

    print(f"  AUC-ROC Score: {auc:.4f}")
    print(f"  Classification Report:\n{report}")
    print(f"  Confusion Matrix:\n{matrix}")

    # Return the fitted pipeline and feature names for saving
    return pipeline, feature_names_out

def save_pipeline(pipeline, feature_names, model_path, features_path):
    """Saves the trained pipeline and feature names using joblib."""
    try:
        print(f"Saving trained pipeline to: {model_path}")
        joblib.dump(pipeline, model_path)

        # Ensure feature_names is explicitly a list before saving
        feature_names_list = []
        if isinstance(feature_names, list):
            feature_names_list = feature_names
            print(f"Feature names already a list ({len(feature_names_list)} items). Proceeding to save.")
        elif hasattr(feature_names, 'tolist'): # Handles numpy arrays, pandas Index, etc.
            feature_names_list = feature_names.tolist()
            print(f"Converted feature names from {type(feature_names)} to list ({len(feature_names_list)} items) for saving.")
        else:
            try:
                feature_names_list = list(feature_names)
                print(f"Attempted generic conversion of feature names from {type(feature_names)} to list ({len(feature_names_list)} items). Proceeding to save.")
            except TypeError:
                print(f"Error: Could not convert feature names of type {type(feature_names)} to list. Saving empty list as fallback.")
                feature_names_list = []

        print(f"Saving feature names list to: {features_path}")
        joblib.dump(feature_names_list, features_path)
        print("Pipeline and feature names saved successfully.")

    except Exception as e:
        print(f"Error saving pipeline or features: {e}")


# --- Main Script Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a termination prediction model.")
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save the model and feature names.')
    args = parser.parse_args()

    # Construct full paths for output files
    output_dir = os.path.abspath(args.output_dir)
    MODEL_OUTPUT_PATH = os.path.join(output_dir, MODEL_OUTPUT_FILENAME)
    FEATURE_NAMES_OUTPUT_PATH = os.path.join(output_dir, FEATURE_NAMES_OUTPUT_FILENAME)

    print("Starting termination model training process...")
    print(f"Data directory: {DATA_DIR}")
    print(f"File pattern: {FILE_PATTERN}")
    print(f"Model output path: {MODEL_OUTPUT_PATH}")
    print(f"Features list output path: {FEATURE_NAMES_OUTPUT_PATH}")


    # 1. Load data
    all_data = load_all_census_data(DATA_DIR, FILE_PATTERN, DATE_COLUMNS)

    # 2. Create target variable
    data_with_target = add_termination_target(all_data, ID_COLUMN, 'year', TARGET_VARIABLE)

    # 3. Calculate features
    data_with_features = calculate_model_features(data_with_target, 'birth_date', 'hire_date', 'census_date')

    # 4. Clean data and select final features/target
    model_ready_data = clean_and_prepare_data(data_with_features, FEATURES_TO_USE, TARGET_VARIABLE)

    # 5. Split data and create preprocessor
    X_train, X_test, y_train, y_test = split_data(model_ready_data, TARGET_VARIABLE, 'year')
    preprocessor = create_preprocessing_pipeline(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)

    # 6. Train and evaluate model
    pipeline, feature_names = train_and_evaluate_model(preprocessor, X_train, y_train, X_test, y_test)

    # 7. Save pipeline and feature names
    if pipeline and feature_names is not None:
         save_pipeline(pipeline, feature_names, MODEL_OUTPUT_PATH, FEATURE_NAMES_OUTPUT_PATH)
    else:
         print("Skipping saving pipeline due to issues during training or feature name extraction.")

    print("\nTermination model training script finished successfully.")
