"""
Core projection logic for the census data.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from datetime import timedelta
import math
from scipy.stats import truncnorm # Added for new hire generation
import joblib # Import joblib for loading model/features

from utils.date_utils import calculate_age, calculate_tenure
from sandbox.utils import generate_new_ssn, generate_new_hires
from sandbox.ml_logic import prepare_features_for_model, calculate_turnover_score_rule_based, ML_LIBS_AVAILABLE, apply_stochastic_termination, apply_terminations, apply_rule_based_turnover
from utils.plan_rules import determine_eligibility, apply_auto_enrollment, apply_auto_increase, apply_plan_change_deferral_response, calculate_contributions

# --- Placeholder Imports for Plan Rule Engine ---
# These functions will be defined elsewhere (e.g., plan_rules.py)

# --- Constants for New Hire Generation (if not in config) ---
DEFAULT_AGE_MEAN = 35
DEFAULT_AGE_STD_DEV = 8
DEFAULT_MIN_WORKING_AGE = 18
DEFAULT_MAX_WORKING_AGE = 65
DEFAULT_COMP_LOG_MEAN_FACTOR = 1.0
DEFAULT_COMP_SPREAD_SIGMA = 0.25
DEFAULT_COMP_BASE_SALARY = 45000
DEFAULT_COMP_INCREASE_PER_AGE_YEAR = 200
DEFAULT_COMP_INCREASE_PER_TENURE_YEAR = 400 # Tenure is 0 for new hires
DEFAULT_COMP_MIN_SALARY = 30000

def project_census(
    start_df,                   # Starting census DataFrame
    scenario_config,            # Config for the *current* scenario being run
    baseline_scenario_config    # Config for the baseline scenario (for comparison)
    ):
    """
    Projects census forward year by year according to a scenario configuration.
    Integrates population dynamics with (future) plan rule application.

    Args:
        start_df (pd.DataFrame): The cleaned census data for the starting year.
        scenario_config (dict): Configuration dictionary for the simulation scenario.
                                See run_projection.py for expected keys.

    Returns:
        dict: Dictionary of projected census DataFrames keyed by simulation year number (1, 2, ...).
    """
    # --- Extract Parameters from Config ---
    projection_years = scenario_config['projection_years']
    start_year = scenario_config['start_year'] # First year of projection
    comp_increase_rate = scenario_config['comp_increase_rate']
    hire_rate = scenario_config['hire_rate'] # Gross hire rate
    termination_rate = scenario_config['termination_rate'] # Target gross term rate

    # ML Turnover Settings
    use_ml_turnover = scenario_config.get('use_ml_turnover', False)
    ml_model_path = scenario_config.get('ml_model_path', None)
    model_features_path = scenario_config.get('model_features_path', None) # Path to JOBLED feature list
    projection_model = None # Loaded model pipeline
    model_feature_names = [] # Loaded list of feature names

    # New Hire Generation Settings (use defaults if not provided)
    role_distribution = scenario_config.get('role_distribution', None)
    role_compensation_params = scenario_config.get('role_compensation_params', None)
    # Use defaults for age/comp generation if role-specific params aren't given
    age_mean = scenario_config.get('age_mean', DEFAULT_AGE_MEAN)
    age_std_dev = scenario_config.get('age_std_dev', DEFAULT_AGE_STD_DEV)
    min_working_age = scenario_config.get('min_working_age', DEFAULT_MIN_WORKING_AGE)
    max_working_age = scenario_config.get('max_working_age', DEFAULT_MAX_WORKING_AGE)

    # --- Initial Setup ---
    projected_data = {}
    current_df = start_df.copy()
    # Determine the actual starting date for the loop based on config start_year
    # Assume the input df's date is the end of the year *before* the projection starts.
    last_plan_year_end_date = pd.Timestamp(f"{start_year - 1}-12-31")

    # Load ML model and features if specified
    if use_ml_turnover and ml_model_path and model_features_path and ML_LIBS_AVAILABLE:
        try:
            print(f"Loading ML model pipeline from: {ml_model_path}")
            projection_model = joblib.load(ml_model_path)
            print(f"Loading model feature names from: {model_features_path}")
            model_feature_names = joblib.load(model_features_path)
            print(f"ML model and {len(model_feature_names)} features loaded successfully.")
            if not isinstance(model_feature_names, list):
                print("Warning: Loaded feature names is not a list. Fallback likely.")
                model_feature_names = [] # Reset if load failed or wrong type
                projection_model = None # Don't use model if features are wrong

        except FileNotFoundError:
            print(f"Error: ML model or features file not found at specified paths. Using rule-based scoring.")
            projection_model = None
            model_feature_names = []
        except Exception as e:
            print(f"Warning: Failed to load ML model/features: {e}. Using rule-based scoring.")
            projection_model = None # Ensure fallback
            model_feature_names = []
    elif use_ml_turnover:
        print("Warning: ML turnover requested but model path, features path, or ML libs missing/unavailable. Using rule-based scoring.")
        projection_model = None # Ensure fallback
        model_feature_names = []

    print(f"\n--- Starting Projection for Scenario: {scenario_config['scenario_name']} from {start_year} ---")
    print(f"Initial Headcount: {len(current_df)}")

    # --- Yearly Simulation Loop ---
    for year_num in range(1, projection_years + 1):
        current_sim_year = start_year + year_num - 1
        year_start_date = pd.Timestamp(f"{current_sim_year}-01-01")
        year_end_date = pd.Timestamp(f"{current_sim_year}-12-31")
        print(f"\n--- Projecting Year {year_num} ({current_sim_year}) ---")

        # 0. Check if current_df is empty (e.g., all terminated)
        if current_df.empty:
            print(f"Warning: No employees remaining at the start of year {current_sim_year}. Stopping projection for this scenario.")
            break

        # 1. Apply Comp Increase to continuing employees
        # Ensure column exists and is numeric
        if 'gross_compensation' in current_df.columns:
             current_df['gross_compensation'] = pd.to_numeric(current_df['gross_compensation'], errors='coerce').fillna(0)
             current_df['gross_compensation'] *= (1 + comp_increase_rate)
        else:
             print("Warning: 'gross_compensation' column not found. Skipping comp increase.")

        # === Termination Logic ===
        term_count_this_year = 0
        termination_probs = None
        original_index = current_df.index # Keep original index for alignment

        # --- Calculate Dynamic Features for ML Model (if applicable) ---
        # Always calculate age/tenure for potential use, even if ML fallback occurs
        if 'birth_date' in current_df.columns:
            current_df['age'] = calculate_age(current_df['birth_date'], year_end_date)
        else:
            print(f"Warning: 'birth_date' column missing. Cannot calculate age for year {current_sim_year}.")
            current_df['age'] = np.nan # Assign NaN if birth_date is missing

        if 'hire_date' in current_df.columns:
            current_df['tenure'] = calculate_tenure(current_df['hire_date'], year_end_date)
        else:
            print(f"Warning: 'hire_date' column missing. Cannot calculate tenure for year {current_sim_year}.")
            current_df['tenure'] = np.nan # Assign NaN if hire_date is missing

        # --- Attempt ML-based Turnover Prediction ---           
        if projection_model and model_feature_names:
            try:
                # Define the raw features needed based on the updated training script
                RAW_FEATURES_FOR_MODEL = ['age', 'tenure', 'gross_compensation', 'pre_tax_deferral_percentage']

                # Check if all required raw features are present
                missing_raw_features = [col for col in RAW_FEATURES_FOR_MODEL if col not in current_df.columns]
                if missing_raw_features:
                    print(f"Warning: Missing raw features required for ML model: {missing_raw_features}. Falling back to rule-based.")
                    raise ValueError("Missing raw features for prediction")

                # Select only the necessary raw features
                X_predict = current_df[RAW_FEATURES_FOR_MODEL].copy()

                # Optional: Handle potential NaNs in features just before prediction (though dynamic calc should minimize this)
                # Consider a more robust imputation strategy if needed, maybe based on training data stats
                if X_predict.isnull().any().any():
                   print("Warning: NaNs detected in features before ML prediction. Applying simple median fill.")
                   for col in RAW_FEATURES_FOR_MODEL:
                       if X_predict[col].isnull().any():
                           # Use median of the current year's data for imputation - simple approach
                           median_val = X_predict[col].median()
                           X_predict[col].fillna(median_val, inplace=True)
                           print(f"  Imputed NaNs in {col} with median {median_val}")
                   # Check again after imputation
                   if X_predict.isnull().any().any():
                       print("Error: NaNs persist after imputation. Falling back to rule-based.")
                       raise ValueError("NaNs persist after imputation")

                # Predict probabilities (pipeline handles preprocessing)
                print("Predicting turnover probabilities using ML model...")
                predicted_probs_all = projection_model.predict_proba(X_predict)
                
                # Ensure we are getting probability of class 1 (termination)
                # Find index of class 1 in model's classes_ attribute
                class_1_index = np.where(projection_model.classes_ == 1)[0][0]
                termination_probs = predicted_probs_all[:, class_1_index]
                # Align probs with original DataFrame index
                termination_probs = pd.Series(termination_probs, index=original_index)

                # Apply stochastic termination based on probabilities
                print("Applying stochastic termination based on ML probabilities...")
                termination_decisions = apply_stochastic_termination(termination_probs)
                terminated_ids = termination_decisions[termination_decisions].index
                term_count_this_year = len(terminated_ids)
                
                # Apply terminations to the dataframe
                current_df = apply_terminations(current_df, terminated_ids, year_end_date)
                print(f"Stochastic Terminations: {term_count_this_year} ({term_count_this_year / len(original_index):.2%}) based on ML probabilities.")

            except Exception as e:
                print(f"Error during ML prediction: {e}. Falling back to rule-based scoring for this year.")
                projection_model = None # Disable ML for subsequent years if it fails once
                termination_probs = None # Ensure fallback logic triggers

        # --- Fallback or Default: Rule-Based Turnover ---           
        if termination_probs is None: # Triggered if ML disabled, failed, or wasn't requested
            print("Using rule-based turnover scoring...")
            # (Ensure age/tenure were calculated above for rule-based too)
            # Apply rule-based terminations with random dates across the year
            current_df = apply_rule_based_turnover(
                current_df,
                termination_rate,
                year_start_date,
                year_end_date
            )
            # Count how many were actually terminated by the rule-based logic
            # This assumes apply_terminations adds 'termination_date'
            term_count_this_year = current_df['termination_date'].notna().sum() - start_df['termination_date'].notna().sum() if 'termination_date' in start_df.columns else current_df['termination_date'].notna().sum()
            # Adjust logic if apply_rule_based_turnover modifies df differently 
            print(f"Applying deterministic termination based on rule-based scores.") # Message might need update


        # --- Post-Termination Processing ---
        # Employees who terminated before the current year
        terminated_prior = current_df[(current_df['termination_date'].notna()) & (current_df['termination_date'] < year_start_date)].copy()
        # Employees who are active at year-end or terminated during this year
        active_or_terminated_this_year = current_df[
            (current_df['termination_date'].isna()) |
            ((current_df['termination_date'] >= year_start_date) & (current_df['termination_date'] <= year_end_date))
        ].copy()
        term_count_actual = ((current_df['termination_date'].notna()) & (current_df['termination_date'] >= year_start_date) & (current_df['termination_date'] <= year_end_date)).sum()
        print(f"Identified {term_count_actual} terminations during {current_sim_year}. Continuing: {active_or_terminated_this_year['termination_date'].isna().sum()}")
        print("DEBUG: Status counts after termination logic (before filtering):")
        print(active_or_terminated_this_year['status'].value_counts())
        print("DEBUG: Number of 'Terminated' employees (post-termination):", (active_or_terminated_this_year['status'] == 'Terminated').sum())
        # Use this DataFrame for plan rules and contributions
        current_df = active_or_terminated_this_year.copy()

        # 3. Generate New Hires
        target_headcount = len(start_df) # Aim to return to original size (simple assumption)
        if scenario_config.get('maintain_headcount', True):
            current_headcount = len(current_df)
            needed_hires = max(0, target_headcount - current_headcount)
            print(f"Target New Hires: {needed_hires} (Target: {target_headcount}, Current: {current_headcount})")
        else:
            # Hire based on fixed rate of *starting* population for the year
            starting_headcount_this_year = len(original_index) # Headcount before terms/hires
            needed_hires = int(starting_headcount_this_year * hire_rate)
            print(f"Target New Hires: {needed_hires} (Based on {hire_rate:.2%} of {starting_headcount_this_year})")


        if needed_hires > 0:
            new_hires_df = generate_new_hires(
                num_hires=needed_hires,
                hire_year=current_sim_year,
                role_distribution=role_distribution,
                role_compensation_params=role_compensation_params,
                age_mean=age_mean,
                age_std_dev=age_std_dev,
                min_working_age=min_working_age,
                max_working_age=max_working_age,
                scenario_config=scenario_config # Pass full config for flexibility
            )
            # Ensure columns match before concatenating
            new_hires_df = new_hires_df.reindex(columns=current_df.columns, fill_value=pd.NA)
            current_df = pd.concat([current_df, new_hires_df], ignore_index=True)
            print(f"Generated {len(new_hires_df)} new hires.")
            # Recalculate age and tenure for all employees (including new hires)
            if 'birth_date' in current_df.columns:
                current_df['age'] = calculate_age(current_df['birth_date'], year_end_date)
            if 'hire_date' in current_df.columns:
                current_df['tenure'] = calculate_tenure(current_df['hire_date'], year_end_date)
        
        print("Applying Plan Rule Engine...")
        # --- Apply Plan Rules --- 
        # Replace the single call with the sequence of rule functions
        current_df = determine_eligibility(current_df, scenario_config, year_end_date)
        
        # Apply Auto-Enrollment only if enabled
        if scenario_config.get('plan_rules', {}).get('auto_enrollment', {}).get('enabled', False):
            current_df = apply_auto_enrollment(
                current_df,
                scenario_config['plan_rules'],
                year_start_date,
                year_end_date
            )

        # Apply Deferral Response to Plan Changes (Optional Feature) - Check if enabled/configured
        # Assuming logic for this function exists and config keys are defined if used.
        # if scenario_config.get('plan_features', {}).get('plan_change_deferral_response_enabled', False):
        #    current_df = apply_plan_change_deferral_response(current_df, scenario_config, baseline_scenario_config, year_end_date)

        # Apply Auto-Increase only if enabled
        ai_cfg = scenario_config.get('plan_rules', {}).get('auto_increase', {})
        if ai_cfg.get('enabled', False):
            # ensure ai_enrolled column exists
            if 'ai_enrolled' not in current_df.columns:
                current_df['ai_enrolled'] = False
            # initial AI enrollment
            if ai_cfg.get('apply_to_new_hires_only', False):
                # flag employees hired this year
                mask_new = current_df['hire_date'].dt.year == year_end_date.year
                current_df.loc[mask_new, 'ai_enrolled'] = True
            else:
                # flag all eligible employees
                current_df.loc[current_df['is_eligible'], 'ai_enrolled'] = True
            # Apply auto-increase per scenario config
            current_df = apply_auto_increase(
                current_df,
                scenario_config['plan_rules'],
                year_end_date.year
            )
            # Debug AI: report flagged count and sample rates
            ai_count = current_df['ai_enrolled'].sum() if 'ai_enrolled' in current_df.columns else 0
            print(f"  DEBUG AI: total ai_enrolled: {ai_count}")
            if ai_count > 0:
                # Ensure ai_enrolled mask has no NA values
                sample_rates = current_df.loc[current_df['ai_enrolled'].fillna(False), 'deferral_rate'].head(5).tolist()
                print(f"  DEBUG AI: sample ai_enrolled deferral rates: {sample_rates}")

        print(f"DEBUG: Before calculate_contributions - Type: {type(current_df)}")
        if current_df is not None:
            print("DEBUG: Before calculate_contributions - Head:\n", current_df.head())
        else:
            print("DEBUG: Before calculate_contributions - current_df is None!")
        assert current_df is not None, "current_df became None before calculate_contributions call!"
        # Calculate contributions (deferrals, match, NEC)
        current_df = calculate_contributions(current_df, scenario_config, year_end_date.year, year_start_date, year_end_date)

        # --- Snapshot Yearly Results with Terminations Included ---
        # Define terminated_employees as those who terminated during this year
        terminated_employees = current_df[(current_df['termination_date'].notna()) & (current_df['termination_date'] >= year_start_date) & (current_df['termination_date'] <= year_end_date)].copy()
        # Align terminated to same columns
        terminated_aligned = terminated_employees.reindex(columns=current_df.columns, fill_value=pd.NA)
        year_snapshot = pd.concat([terminated_aligned, current_df], ignore_index=True)
        # Classify status into five categories per year
        conditions = [
            # Previously terminated (before this period)
            year_snapshot['termination_date'].notna() & (year_snapshot['termination_date'] < year_start_date),
            # Experienced employees terminating this period
            year_snapshot['termination_date'].notna() & (year_snapshot['termination_date'] >= year_start_date) & (year_snapshot['hire_date'] < year_start_date),
            # New hires terminating this period
            year_snapshot['termination_date'].notna() & (year_snapshot['termination_date'] >= year_start_date) & (year_snapshot['hire_date'] >= year_start_date),
            # Continuous active (hired before period, not terminated)
            year_snapshot['termination_date'].isna() & (year_snapshot['hire_date'] < year_start_date),
            # New hire active (hired this period, not terminated)
            year_snapshot['termination_date'].isna() & (year_snapshot['hire_date'] >= year_start_date)
        ]
        choices = [
            'Previously Terminated',
            'Experienced Terminated',
            'New Hire Terminated',
            'Continuous Active',
            'New Hire Active'
        ]
        year_snapshot['status_category'] = np.select(conditions, choices, default='Unknown')
        projected_data[year_num] = year_snapshot.copy()
        active_count = len(current_df)
        print(f"End of Year {current_sim_year} Active Headcount: {active_count}")

    print("\n--- Projection Complete ---")
    return projected_data
