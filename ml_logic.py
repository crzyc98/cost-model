"""
Functions related to the machine learning turnover model.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- ML Dependencies ---
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, classification_report
    from sklearn.preprocessing import StandardScaler # Example scaler
    from sklearn.pipeline import Pipeline
    import lightgbm as lgb
    ML_LIBS_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn or lightgbm not found. ML model training/prediction disabled.")
    print("Install with: pip install scikit-learn lightgbm")
    # Define placeholders if libs not found
    ML_LIBS_AVAILABLE = False
    Pipeline = object
    lgb = None
    StandardScaler = object
    # Define dummy functions or classes if needed for type hinting elsewhere
    def train_test_split(*args, **kwargs): return args
    def roc_auc_score(*args, **kwargs): return 0.0
    def classification_report(*args, **kwargs): return "ML Libraries not available"

from data_processing import load_and_clean_census
from utils import calculate_age, calculate_tenure

# --- Training Data Construction ---
def build_training_data(historical_files, feature_names):
    """ Builds ML training data (X, y) from consecutive census files. """
    print("\n--- Building Training Data for Turnover Model ---")
    all_training_data = []
    required_cols = ['ssn', 'birth_date', 'hire_date', 'termination_date', 'gross_compensation']
    expected_cols = {'required': required_cols} # Add contribution cols if needed later

    # Sort files to ensure chronological order (best effort based on inferred dates later)
    historical_files.sort() # Simple alphabetical sort, assumes filenames like census_YYYY.csv

    df_n_plus_1 = None
    end_date_n_plus_1 = None

    for i in range(len(historical_files) -1, -1, -1): # Iterate backwards N+1 -> N
        filepath_n = historical_files[i]
        print(f"\nProcessing File N: {os.path.basename(filepath_n)}")

        df_n = load_and_clean_census(filepath_n, expected_cols)
        if df_n is None: continue # Skip if loading failed

        end_date_n = df_n['plan_year_end_date'].iloc[0]

        # Calculate features for year N, based on end_date_n
        df_n['age_at_period_start'] = df_n['birth_date'].apply(lambda x: calculate_age(x, end_date_n))
        df_n['tenure_at_period_start'] = df_n.apply(lambda row: calculate_tenure(row['hire_date'], end_date_n), axis=1)
        df_n['comp_at_period_start'] = df_n['gross_compensation'] # Or use normalized comp

        # --- Select features ---
        # Ensure ssn is always included for joining, then add model features
        cols_to_select = ['ssn'] + [f for f in feature_names if f != 'ssn' and f in df_n.columns]
        current_features = df_n[cols_to_select].copy()

        # Check if all required *model* features were successfully calculated
        if not all(f in current_features.columns for f in feature_names if f != 'ssn'):
             print(f"Error: Not all required model features {feature_names} found in calculated features for {os.path.basename(filepath_n)}.")
             print(f"Calculated features available: {list(current_features.columns)}")
             continue # Skip this file's contribution to training data

        if df_n_plus_1 is not None:
            print(f"Comparing with File N+1: {os.path.basename(historical_files[i+1])}")

            # Identify who was active at the end of year N
            active_n_mask = df_n['termination_date'].isna() | (df_n['termination_date'] > end_date_n)
            active_n_ssns = set(df_n.loc[active_n_mask, 'ssn'])

            # Identify who was terminated *during* year N+1 (between end_date_n and end_date_n_plus_1)
            terminated_in_n_plus_1_mask = (
                df_n_plus_1['termination_date'].notna() &
                (df_n_plus_1['termination_date'] > end_date_n) &
                (df_n_plus_1['termination_date'] <= end_date_n_plus_1)
            )
            terminated_ssns_method2 = set(df_n_plus_1.loc[terminated_in_n_plus_1_mask, 'ssn'])

            # Combine active SSNs from year N with their features and target (terminated status)
            year_data = current_features.loc[current_features['ssn'].isin(active_n_ssns)].copy()
            year_data['terminated_next_period'] = year_data['ssn'].apply(lambda s: 1 if s in terminated_ssns_method2 else 0)

            # Drop rows with NaN in features needed for modeling
            initial_rows = len(year_data)
            year_data.dropna(subset=[f for f in feature_names if f != 'ssn'], inplace=True)
            dropped_rows = initial_rows - len(year_data)
            if dropped_rows > 0:
                print(f"Dropped {dropped_rows} rows due to missing feature values for training period ending {end_date_n.year}")

            all_training_data.append(year_data)
            print(f"Added {len(year_data)} training samples from period {end_date_n.year}. {year_data['terminated_next_period'].sum()} terminations.")

        # Current N becomes N+1 for the next iteration (backwards)
        df_n_plus_1 = df_n.copy()
        end_date_n_plus_1 = end_date_n

    if not all_training_data:
        print("Warning: No training data could be generated. Check historical files and feature calculation.")
        return None, None

    training_df = pd.concat(all_training_data, ignore_index=True)
    print(f"\nTotal Training Data Size: {training_df.shape}")
    print(f"Overall Turnover Rate in Training Data: {training_df['terminated_next_period'].mean():.2%}")

    X = training_df[feature_names]
    y = training_df['terminated_next_period']

    return X, y

# --- Model Training Function ---
def train_turnover_model(X_train, y_train):
    """ Trains a turnover prediction model (LGBMClassifier). """
    if not ML_LIBS_AVAILABLE:
        print("ML Libraries not available. Skipping model training.")
        return None

    print("\n--- Training Turnover Model ---")
    if X_train is None or y_train is None or X_train.empty:
        print("Error: Training data (X or y) is empty or None.")
        return None

    if len(X_train) != len(y_train):
         print(f"Error: Mismatch between X_train ({len(X_train)}) and y_train ({len(y_train)}) lengths.")
         return None

    # Basic validation
    if y_train.nunique() < 2:
        print("Warning: Training target variable has less than 2 unique classes. Model may not train well.")
        # Handle this case: maybe return a dummy predictor or raise error?
        # For now, proceed but warn

    # Define the model pipeline (example with scaling)
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', lgb.LGBMClassifier(random_state=42))
        # Add hyperparameter tuning, cross-validation etc. here for robustness
    ])

    try:
        # Drop SSN if it was included in X_train by mistake
        if 'ssn' in X_train.columns:
            print("Warning: 'ssn' column found in X_train, dropping it before training.")
            X_train_processed = X_train.drop(columns=['ssn'])
        else:
            X_train_processed = X_train

        model.fit(X_train_processed, y_train)
        print("Model training complete.")

        # Optional: Evaluate on training data (or better, a validation set)
        y_pred_proba = model.predict_proba(X_train_processed)[:, 1]
        auc = roc_auc_score(y_train, y_pred_proba)
        print(f"Training AUC: {auc:.4f}")
        # print("Training Classification Report:\n", classification_report(y_train, model.predict(X_train_processed)))

        return model
    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
        return None

# --- RULE-BASED Turnover Score Function (Fallback) ---
def calculate_turnover_score_rule_based(age, tenure, comp, comp_p25):
    """ Calculates a simple rule-based turnover score. Higher score = higher likelihood. """
    score = 0
    if pd.isna(age) or pd.isna(tenure) or pd.isna(comp): return 0.5 # Default neutral
    if age < 25 or age > 60: score += 0.1
    if tenure < 2: score += 0.2
    if comp < comp_p25: score += 0.15 # Lower quartile compensation
    # Add more rules based on domain knowledge
    return min(max(score, 0.05), 0.95) # Clamp between 0.05 and 0.95

# --- Feature Preparation Function for Prediction ---
def prepare_features_for_model(df, feature_names, reference_date):
    """ Prepares features for prediction using the trained model/pipeline. """
    print(f"Preparing features for prediction as of {reference_date.date()}...")
    features_df = pd.DataFrame(index=df.index)

    # Calculate features based on the reference date
    if 'current_age' in feature_names or 'age_at_period_start' in feature_names:
        features_df['current_age'] = df['birth_date'].apply(lambda x: calculate_age(x, reference_date))
        # Alias if needed by the model trained feature names
        if 'age_at_period_start' in feature_names and 'current_age' not in feature_names:
             features_df['age_at_period_start'] = features_df['current_age']

    if 'current_tenure' in feature_names or 'tenure_at_period_start' in feature_names:
        features_df['current_tenure'] = df.apply(lambda row: calculate_tenure(row['hire_date'], reference_date), axis=1)
        if 'tenure_at_period_start' in feature_names and 'current_tenure' not in feature_names:
             features_df['tenure_at_period_start'] = features_df['current_tenure']

    if 'gross_compensation' in feature_names or 'comp_at_period_start' in feature_names:
        features_df['gross_compensation'] = df['gross_compensation']
        if 'comp_at_period_start' in feature_names and 'gross_compensation' not in feature_names:
             features_df['comp_at_period_start'] = features_df['gross_compensation']

    # Add other feature calculations as needed based on feature_names

    # Ensure all required features are present, handle missing ones (e.g., impute or error)
    final_features = pd.DataFrame(index=df.index)
    missing_cols = []
    for feat in feature_names:
        if feat == 'ssn': continue # SSN is identifier, not feature
        if feat in features_df.columns:
            final_features[feat] = features_df[feat]
        else:
            missing_cols.append(feat)
            print(f"Warning: Feature '{feat}' could not be calculated for prediction.")
            # Decide how to handle: Add NaN, impute, or raise error?
            final_features[feat] = np.nan # Fill with NaN for now

    if missing_cols:
        print(f"Warning: Missing columns during feature preparation: {missing_cols}")

    # Important: Handle NaNs before prediction (e.g., median imputation)
    # This should ideally mirror the preprocessing done during training
    # Example: Impute with median (calculate median from the *training* data if possible)
    for col in final_features.columns:
        if final_features[col].isnull().any():
            median_val = final_features[col].median() # Use training median ideally
            final_features[col].fillna(median_val, inplace=True)
            print(f"Imputed NaNs in feature '{col}' with median value {median_val:.2f}")

    return final_features # Return only the feature columns needed by model
