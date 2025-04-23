"""
Script to run retirement plan projection simulations for multiple scenarios.
"""

import pandas as pd
from projection_utils import project_census
from datetime import datetime # Added for IRS limits
import argparse # Keep argparse for input/output files
import joblib # Import joblib for loading the model
import numpy as np # Import numpy for np.where
import logging

# --- Define Scenario Configurations (Based on updates.md) ---

# Example IRS Limits (Expand as needed for simulation years)
DEFAULT_IRS_LIMITS = {
    2024: {'comp_limit': 345000, 'deferral_limit': 23000, 'catch_up': 7500},
    2025: {'comp_limit': 360000, 'deferral_limit': 24000, 'catch_up': 8000}, # Placeholder
    2026: {'comp_limit': 375000, 'deferral_limit': 25000, 'catch_up': 8000}, # Placeholder
    2027: {'comp_limit': 390000, 'deferral_limit': 26000, 'catch_up': 8500}, # Placeholder
    2028: {'comp_limit': 405000, 'deferral_limit': 27000, 'catch_up': 8500}, # Placeholder
    2029: {'comp_limit': 420000, 'deferral_limit': 28000, 'catch_up': 9000}, # Placeholder
}

# Baseline: AE on new hires only (35-day window), no AI
baseline_scenario = {
    'scenario_name': 'Baseline',
    'start_year': 2025,
    'projection_years': 5,
    'comp_increase_rate': 0.03,
    'hire_rate': 0.10,
    'termination_rate': 0.08,
    'maintain_headcount': False,
    'plan_rules': {
        'eligibility': {'min_age':21, 'min_service_months':0},
        'auto_enrollment': {
            'enabled': True,
            'proactive_enrollment_probability': 0.05,  # enable proactive AE
            'proactive_rate_range': (0.01, 0.05),      # variable deferral range
            'default_rate': 0.02,
            'ae_outcome_distribution': {'stay_default':0.90, 'opt_out':0.10},
            'window_days': 35
        },
        'auto_increase': {'enabled': False},
        'employer_match_formula': '50% up to 6%',
        'employer_non_elective_formula': '0%',
        'min_hours_worked': 1000,
        'last_day_work_rule': True,
        'match_change_response': {
            'enabled': True,
            'increase_probability': 1.0,
            'increase_target': 'optimal'
        }
    },
    'irs_limits': DEFAULT_IRS_LIMITS,
    'use_ml_turnover': True,
    'ml_model_path': 'termination_model_pipeline.joblib',
    'model_features_path': 'termination_model_features.joblib'
}

# Scenario: AIP for new hires only
aip_new_hires = {
    **baseline_scenario,
    'scenario_name': 'AIP_New_Hires',
    'plan_rules': {**baseline_scenario['plan_rules'], 
                   'auto_enrollment': {**baseline_scenario['plan_rules']['auto_enrollment'], 
                                       'proactive_enrollment_probability': 0.05},  # enable proactive AE
                   'auto_increase': {'enabled': True, 'increase_rate':0.01, 'cap_rate':0.06, 'apply_to_new_hires_only':True}}
}

# Scenario: AIP for all eligible
aip_all_eligible = {
    **baseline_scenario,
    'scenario_name': 'AIP_All_Eligible',
    'plan_rules': {
        **baseline_scenario['plan_rules'],
        'auto_enrollment': {
            **baseline_scenario['plan_rules']['auto_enrollment'],
            'proactive_enrollment_probability': 0.05,
            're_enroll_existing': True  # enroll existing eligibles at default rate
        },
        'auto_increase': {'enabled': True, 'increase_rate': 0.01, 'cap_rate': 0.06}
    }
}

# List of scenarios to run
SCENARIOS_TO_RUN = [baseline_scenario, aip_new_hires, aip_all_eligible]

# --- Helper Functions ---

def load_and_initialize_data(csv_path):
    """
    Loads the initial census CSV, parses dates, and adds required columns if missing.
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded initial census: {csv_path}")
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {csv_path}")
        return None
    except Exception as e:
        print(f"Error loading CSV {csv_path}: {e}")
        return None

    # Basic Date Parsing (add more robust error handling as needed)
    date_cols = ['birth_date', 'hire_date', 'termination_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        else:
            print(f"Warning: Expected date column '{col}' not found.")
            # Decide handling: Add empty? Error out? For now, just warn.
            if col == 'termination_date': # Termination date can be optional
                 df[col] = pd.NaT

    # --- Initialization of Status Columns (as per updates.md) ---
    # These will be properly calculated within the simulation loop later,
    # but initializing them helps structure.
    if 'is_eligible' not in df.columns: df['is_eligible'] = False
    if 'is_participating' not in df.columns: df['is_participating'] = False # Placeholder
    if 'ae_opted_out' not in df.columns: df['ae_opted_out'] = False
    if 'ai_opted_out' not in df.columns: df['ai_opted_out'] = False
    # make sure we always have an AI‑enrollment flag
    if 'ai_enrolled' not in df.columns:
        df['ai_enrolled'] = False
    # make sure we always have an AE‑enrollment flag
    if 'ae_enrolled' not in df.columns:
        df['ae_enrolled'] = False
    # Initial deferral rate from percentage column
    if 'pre_tax_deferral_percentage' in df.columns:
        # Convert percentage or fraction to rate (<=1 treated as fraction, >1 as percent)
        if 'deferral_rate' not in df.columns:
            raw_pct = pd.to_numeric(df['pre_tax_deferral_percentage'], errors='coerce')
            df['deferral_rate'] = raw_pct.where(raw_pct <= 1, raw_pct / 100.0).fillna(0.0)
    elif 'deferral_rate' not in df.columns:
        df['deferral_rate'] = 0.0 # Default if no source column
        print("Warning: 'pre_tax_deferral_percentage' or 'deferral_rate' not found. Initializing deferral rate to 0.")

    # Ensure essential columns exist for projection logic (add more as identified)
    required_cols = ['ssn', 'birth_date', 'hire_date', 'gross_compensation']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' is missing from the input CSV.")
            return None

    # Add initial plan_year_end_date based on the LATEST date found in the input
    # This is a simple heuristic; might need refinement.
    # For simulation, the 'start_year' from config will be more important.
    latest_date = df[date_cols].max().max()
    if pd.isna(latest_date) and 'plan_year_end_date' in df.columns:
         latest_date = pd.to_datetime(df['plan_year_end_date']).max()
    if pd.isna(latest_date):
        print("Warning: Could not determine latest date from data. Using default end date assumption.")
        # A reasonable default might be end of previous year if start_year is known
        # For now, just add a placeholder column if needed by downstream funcs.
        if 'plan_year_end_date' not in df.columns: df['plan_year_end_date'] = pd.NaT
    elif 'plan_year_end_date' not in df.columns:
        df['plan_year_end_date'] = pd.Timestamp(f"{latest_date.year}-12-31")
        print(f"Added 'plan_year_end_date' based on latest data: {df['plan_year_end_date'].iloc[0].date()}")
    else:
         df['plan_year_end_date'] = pd.to_datetime(df['plan_year_end_date'])

    # --- Set initial 'status' based on termination_date --- NEW --- #
    if 'status' not in df.columns:
        if 'termination_date' in df.columns:
            df['status'] = np.where(pd.isna(df['termination_date']), 'Active', 'Terminated')
            print("Initialized 'status' column based on 'termination_date'.")
        else:
            df['status'] = 'Active' # Assume active if no termination info
            print("Warning: 'termination_date' not found. Initializing 'status' column to 'Active' for all.")
    else:
        print("Column 'status' already exists in input file.")
    # --- END NEW --- #

    return df

def aggregate_scenario_results(yearly_data, scenario_config):
    """
    Aggregates key metrics from the detailed yearly projection dataframes.

    Args:
        yearly_data (dict): Dictionary where keys are years and values are projected DFs.
        scenario_config (dict): The configuration for the scenario.

    Returns:
        pd.DataFrame: A DataFrame summarizing key metrics per year.
    """
    summary_list = []
    start_year = scenario_config['start_year']

    for year, df in yearly_data.items():
        # Ensure year is integer for calculations if needed
        current_sim_year = int(year)

        # Filter for active employees at year-end for most metrics
        # Align status check with other functions
        active_df = df[df['status'].isin(['Active', 'Unknown'])].copy()
        
        if active_df.empty:
             print(f"  Warning: No active employees found for year {current_sim_year}. Skipping summary.")
             continue

        # Calculate Metrics
        headcount = len(active_df)
        
        # Ensure calculation columns exist
        has_contributions = 'total_contributions' in active_df.columns
        has_participation = 'is_participating' in active_df.columns
        has_eligibility = 'is_eligible' in active_df.columns
        has_deferral = 'deferral_rate' in active_df.columns
        has_pre_tax = 'pre_tax_contributions' in active_df.columns
        has_match = 'employer_match_contribution' in active_df.columns
        has_nec = 'employer_non_elective_contribution' in active_df.columns
        has_plan_comp = 'plan_year_compensation' in active_df.columns
        has_capped_comp = 'capped_compensation' in active_df.columns

        num_eligible = active_df['is_eligible'].sum() if has_eligibility else 0
        num_participating = active_df['is_participating'].sum() if has_participation else 0
        
        participation_rate_eligible = (num_participating / num_eligible) if num_eligible > 0 else 0
        participation_rate_total = (num_participating / headcount) if headcount > 0 else 0
        
        avg_deferral_rate_participants = active_df.loc[active_df['is_participating'] == True, 'deferral_rate'].mean() if has_deferral and num_participating > 0 else 0
        avg_deferral_rate_total = active_df['deferral_rate'].mean() if has_deferral else 0

        total_pre_tax = active_df['pre_tax_contributions'].sum() if has_pre_tax else 0
        total_match = active_df['employer_match_contribution'].sum() if has_match else 0
        total_nec = active_df['employer_non_elective_contribution'].sum() if has_nec else 0
        total_employer_cost = total_match + total_nec
        total_contributions_all = active_df['total_contributions'].sum() if has_contributions else 0 # Should equal pre_tax + match + nec

        # Compensation Totals
        total_plan_year_compensation = 0
        if has_plan_comp:
            if pd.api.types.is_numeric_dtype(active_df['plan_year_compensation']):
                total_plan_year_compensation = active_df['plan_year_compensation'].sum()
                if pd.isna(total_plan_year_compensation):
                    print(f"  Warning: Sum of 'plan_year_compensation' is NaN for year {current_sim_year}. Treating as 0.")
                    total_plan_year_compensation = 0
            else:
                print(f"  Warning: 'plan_year_compensation' column is not numeric for year {current_sim_year}. Type: {active_df['plan_year_compensation'].dtype}. Treating as 0.")
                total_plan_year_compensation = 0
        else:
            print(f"  Warning: 'plan_year_compensation' column not found for year {current_sim_year}. Cannot calculate compensation-based metrics accurately.")

        total_capped_compensation = 0
        if has_capped_comp:
             if pd.api.types.is_numeric_dtype(active_df['capped_compensation']):
                 total_capped_compensation = active_df['capped_compensation'].sum()
                 if pd.isna(total_capped_compensation):
                    print(f"  Warning: Sum of 'capped_compensation' is NaN for year {current_sim_year}. Treating as 0.")
                    total_capped_compensation = 0
             else:
                print(f"  Warning: 'capped_compensation' column is not numeric for year {current_sim_year}. Type: {active_df['capped_compensation'].dtype}. Treating as 0.")
                total_capped_compensation = 0
        else:
            print(f"  Warning: 'capped_compensation' column not found for year {current_sim_year}. Cannot calculate capped compensation-based metrics accurately.")
            
        # Employer Cost Percentages
        employer_cost_pct_plan_comp = (total_employer_cost / total_plan_year_compensation) if total_plan_year_compensation > 0 else 0
        employer_cost_pct_capped_comp = (total_employer_cost / total_capped_compensation) if total_capped_compensation > 0 else 0

        summary_list.append({
            'Year': current_sim_year,
            'Headcount': headcount,
            'Eligible': num_eligible,
            'Participating': num_participating,
            'Participation Rate (Eligible)': participation_rate_eligible,
            'Participation Rate (Total)': participation_rate_total,
            'Avg Deferral Rate (Participants)': avg_deferral_rate_participants,
            'Avg Deferral Rate (Total)': avg_deferral_rate_total,
            'Total Employee Pre-Tax': total_pre_tax,
            'Total Employer Match': total_match,
            'Total Employer NEC': total_nec,
            'Total Employer Cost': total_employer_cost,
            'Total Contributions': total_contributions_all,
            'Total Plan Year Compensation': total_plan_year_compensation,
            'Total Capped Compensation': total_capped_compensation,
            'Employer Cost % Plan Comp': employer_cost_pct_plan_comp,
            'Employer Cost % Capped Comp': employer_cost_pct_capped_comp
        })

    summary_df = pd.DataFrame(summary_list)
    # Set year as index
    if not summary_df.empty:
        summary_df.set_index('Year', inplace=True)
        
    return summary_df

def run_scenario_simulation(scenario_name, scenario_config, start_census_df, baseline_scenario_config):
    """Runs the projection simulation for a single scenario."""
    print(f"Running simulation for scenario: {scenario_name}...")
    
    # Make a deep copy to avoid modifying the original df across scenarios
    start_df = start_census_df.copy(deep=True)

    # *** Crucial Adaptation Point ***
    # The existing `project_census` needs significant modification (as per updates.md)
    # to accept the full scenario_config and integrate the plan rule engine calls.
    # For now, we pass the config, assuming `project_census` will be updated
    # to use parameters like 'comp_increase_rate', 'hire_rate', 'termination_rate',
    # 'projection_years', 'start_year', and eventually the plan rule params.
    
    # TODO: Modify projection.project_census function extensively.
    projected_data = project_census(
        start_df=start_df,
        scenario_config=scenario_config,
        baseline_scenario_config=baseline_scenario_config
    )

    aggregated_results = aggregate_scenario_results(projected_data, scenario_config)
    return projected_data, aggregated_results

# --- Main Execution --- 

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description='Run retirement plan projection simulations for multiple scenarios.')
    parser.add_argument('input_csv', help='Path to the initial census CSV file.')
    parser.add_argument('--output', help='Optional base path/name for output Excel file (e.g., projection_results). Scenario name and .xlsx will be appended.')
    parser.add_argument('--raw-output', action='store_true', help='Save raw agent-level results to Excel')
    # Removed args for individual rates, years are now in config

    args = parser.parse_args()

    # 1. Load and Initialize Data ONCE
    initial_census_df = load_and_initialize_data(args.input_csv)

    if initial_census_df is not None:
        all_summary_results = {}
        all_raw_results = {}

        # 2. Loop Through Scenarios and Run Simulation
        for config in SCENARIOS_TO_RUN:
            baseline_scenario = next((s for s in SCENARIOS_TO_RUN if s['scenario_name'] == 'Baseline'), None)
            if not baseline_scenario:
                print("Error: 'Baseline' scenario configuration not found. Cannot proceed.")
                exit()
            raw_data, summary_data = run_scenario_simulation(config['scenario_name'], config, initial_census_df, baseline_scenario)
            
            all_summary_results[config['scenario_name']] = summary_data
            all_raw_results[config['scenario_name']] = raw_data

        # 3. Process and Save Results
        if args.output:
            base_output_path = args.output
            # Save summary metrics for each scenario
            for scenario_name, yearly_data in all_summary_results.items():
                output_file = f"{base_output_path}_{scenario_name}.xlsx"
                try:
                    with pd.ExcelWriter(output_file) as writer:
                        yearly_data.to_excel(writer, index=True)
                    print(f"Saved results for scenario '{scenario_name}' to: {output_file}")
                except Exception as e:
                    print(f"Error saving results for scenario '{scenario_name}' to {output_file}: {e}")
            # NEW: save combined summary sheet for all scenarios
            try:
                combined_df = pd.concat(all_summary_results, names=['Scenario','Year']).reset_index()
                combined_file = f"{base_output_path}_all_summaries.xlsx"
                with pd.ExcelWriter(combined_file) as writer:
                    combined_df.to_excel(writer, index=False, sheet_name='Comparison')
                print(f"Saved combined summary to: {combined_file}")
            except Exception as e:
                print(f"Error saving combined summary: {e}")
        if args.raw_output:
            # Save raw agent-level data for each scenario
            for scenario_name, data_dict in all_raw_results.items():
                raw_file = f"{base_output_path}_{scenario_name}_raw.xlsx"
                try:
                    with pd.ExcelWriter(raw_file) as writer:
                        # Write each year to its own sheet
                        for year, df in data_dict.items():
                            df.to_excel(writer, sheet_name=f'Year_{year}', index=False)
                        # Combine all years with a Year column into one sheet
                        combined_df = pd.concat([
                            df.assign(Year=year) for year, df in data_dict.items()
                        ], ignore_index=True)
                        combined_df.to_excel(writer, sheet_name='Combined_Raw', index=False)
                except Exception as e:
                    print(f"Error saving raw data for '{scenario_name}' to {raw_file}: {e}")
    else:
        print("Projection run failed due to data loading/initialization errors.")
