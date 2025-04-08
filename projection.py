"""
Core projection logic for the census data.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import math

from utils import calculate_age, calculate_tenure, generate_new_ssn
from ml_logic import prepare_features_for_model, calculate_turnover_score_rule_based, ML_LIBS_AVAILABLE


def project_census(
    start_df,           # Starting census DataFrame (e.g., latest historical data)
    projection_model,   # Trained ML model (Pipeline) or None
    model_feature_names,# List of feature names model requires
    projection_years,   # Integer number of years to project
    growth_rate,        # Float annual growth rate
    turnover_rate,      # Float target overall annual turnover rate
    comp_increase_rate  # Float annual compensation increase rate
    ):
    """
    Projects census forward year by year.

    Args:
        start_df (pd.DataFrame): The cleaned census data for the starting year.
        projection_model (Pipeline or None): The trained ML model pipeline, or None to use rule-based.
        model_feature_names (list): Exact list of feature column names model needs.
        projection_years (int): Number of years to project forward.
        growth_rate (float): Annual growth rate.
        turnover_rate (float): Target overall annual turnover rate (guides N terms).
        comp_increase_rate (float): Annual compensation increase rate.

    Returns:
        dict: Dictionary of projected census DataFrames keyed by projection year number (1, 2, ...).
    """
    projected_data = {}
    current_df = start_df.copy()
    last_plan_year_end_date = current_df['plan_year_end_date'].iloc[0]

    print(f"\n--- Starting Projection from {last_plan_year_end_date.year} ---")
    print(f"Initial Headcount: {len(current_df)}")

    for year in range(1, projection_years + 1):
        projection_date = last_plan_year_end_date + timedelta(days=365 * year)
        print(f"\n--- Projecting Year {year} ({projection_date.year}) ---")

        # 1. Apply Comp Increase to continuing employees
        current_df['gross_compensation'] *= (1 + comp_increase_rate)

        # 2. Predict Turnover / Assign Turnover Score
        # Prepare features based on state *before* terminations/hires for the year
        features_for_prediction = prepare_features_for_model(current_df, model_feature_names, projection_date)

        turnover_scores = pd.Series(index=current_df.index, dtype=float)
        if projection_model and ML_LIBS_AVAILABLE and not features_for_prediction.empty:
            try:
                print("Predicting turnover using ML model...")
                # Ensure features are in the same order as during training
                features_ordered = features_for_prediction[model_feature_names] # Excludes SSN
                turnover_scores = projection_model.predict_proba(features_ordered)[:, 1]
                print(f"ML prediction successful. Average predicted probability: {turnover_scores.mean():.4f}")
            except Exception as e:
                print(f"Error during ML prediction: {e}. Falling back to rule-based scoring.")
                 # Fallback inside the loop if prediction fails for this year
                comp_p25 = current_df['gross_compensation'].quantile(0.25)
                turnover_scores = current_df.apply(
                    lambda row: calculate_turnover_score_rule_based(
                        calculate_age(row['birth_date'], projection_date),
                        calculate_tenure(row['hire_date'], projection_date),
                        row['gross_compensation'],
                        comp_p25
                    ), axis=1)
        else:
            print("Using rule-based turnover scoring...")
            comp_p25 = current_df['gross_compensation'].quantile(0.25)
            turnover_scores = current_df.apply(
                lambda row: calculate_turnover_score_rule_based(
                    calculate_age(row['birth_date'], projection_date),
                    calculate_tenure(row['hire_date'], projection_date),
                    row['gross_compensation'],
                    comp_p25
                ), axis=1)

        current_df['turnover_score'] = turnover_scores

        # 3. Determine Terminations
        # Calculate target number of terminations based on turnover rate
        # Apply rate to the *start* of year headcount for stability
        target_terminations = math.ceil(len(current_df) * turnover_rate)

        # Select employees to terminate based on score (highest scores first)
        terminating_indices = current_df.nlargest(target_terminations, 'turnover_score').index
        terminated_df = current_df.loc[terminating_indices].copy()
        terminated_df['termination_date'] = projection_date # Assign term date
        terminated_df['status'] = 'Terminated'

        # Update main df: Keep those NOT terminated
        continuing_df = current_df.drop(terminating_indices).copy()
        continuing_df['status'] = 'Active'
        print(f"Identified {len(terminated_df)} terminations (Target: {target_terminations}).")


        # 4. Determine New Hires
        target_growth = math.ceil(len(current_df) * growth_rate)
        num_new_hires = target_terminations + target_growth # Replace terms + add growth
        print(f"Target Growth: {target_growth}, New Hires Needed: {num_new_hires}")

        if num_new_hires > 0:
            # --- Create New Hire Data ---
            # Use characteristics of the *continuing* population to model new hires
            # More sophisticated: sample from recent hires in historical data
            new_hire_base = continuing_df if not continuing_df.empty else start_df # Fallback if everyone terminated

            # Assign compensation based on distribution of existing active employees at START of year
            # Apply the annual increase to the sampled base salary for new hires.
            base_salaries_for_sampling = new_hire_base['gross_compensation'] * (1 + comp_increase_rate)

            # Sample compensation
            new_hires_df = pd.DataFrame({
                'gross_compensation': np.random.choice(
                    # Sample from the base salaries AFTER applying the annual increase rate
                    base_salaries_for_sampling,
                    size=num_new_hires,
                    replace=True
                ) * (1 + np.random.uniform(-0.05, 0.05)), # Add small variance
                'ssn': generate_new_ssn(set(start_df['ssn']).union(set(continuing_df['ssn'])), num_new_hires),
                'birth_date': pd.to_datetime([
                    f"{projection_date.year - 30}-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}"
                     for _ in range(num_new_hires)
                ]),
                'hire_date': projection_date, # Assume hired on projection date
                'termination_date': pd.NaT,
                'status': 'Active'
                 # Add other relevant columns (e.g., employee contributions, department) - initialize as NaN or default
                 # 'employee_contribution': 0,
                 # 'employer_contribution': 0,
            })

            # --- Combine continuing and new hires ---
            next_year_df = pd.concat([continuing_df, new_hires_df], ignore_index=True)
            print(f"Added {len(new_hires_df)} new hires.")
        else:
            next_year_df = continuing_df
            print("No new hires needed this year.")

        # 5. Update Plan Year End Date for the new DataFrame
        next_year_df['plan_year_end_date'] = projection_date

        # 6. Store the projected year's data (before cleaning up temporary columns)
        projected_data[year] = next_year_df.copy()

        # 7. Prepare for next iteration
        current_df = next_year_df.drop(columns=['turnover_score'], errors='ignore') # Clean up score
        print(f"End of Year {year} Headcount: {len(current_df)}")


    print("\n--- Projection Complete ---")
    return projected_data
