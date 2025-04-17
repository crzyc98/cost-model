# run_abm_simulation.py

import pandas as pd
import yaml
import argparse
import logging
import os
from decimal import Decimal
from model.retirement_model import RetirementPlanModel

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

def run_simulation(config_path, census_path, output_prefix):
    """Loads config and data, runs the simulation, and saves results."""

    print("--- Loading Configuration ---")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from: {config_path}")
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file {config_path}: {e}")
        return

    print("\n--- Loading Census Data ---")
    try:
        # Specify date columns and how to handle potential errors
        date_cols = ['birth_date', 'hire_date', 'termination_date']
        census_df = pd.read_csv(
            census_path,
            parse_dates=date_cols,
            date_parser=lambda x: pd.to_datetime(x, errors='coerce') # Coerce invalid dates to NaT
        )
        # Convert monetary values and rates to Decimal for consistency
        # Ensure the columns exist before trying to convert
        if 'gross_compensation' in census_df.columns:
             census_df['gross_compensation'] = census_df['gross_compensation'].apply(lambda x: Decimal(str(x)) if pd.notna(x) else Decimal('0.00'))
        if 'deferral_rate' in census_df.columns:
            census_df['deferral_rate'] = census_df['deferral_rate'].apply(lambda x: Decimal(str(x)) if pd.notna(x) else Decimal('0.00'))
            
        # Make sure the ID column is treated as string initially if it might have non-numeric values
        if 'ssn' in census_df.columns:
             census_df['ssn'] = census_df['ssn'].astype(str)

        print(f"Loaded census data from: {census_path}")
        print(f"Initial census size: {len(census_df)} records")
        # print("Census data sample:\n", census_df.head()) # Optional: print head
        # print("Data types:\n", census_df.dtypes) # Optional: check types

    except FileNotFoundError:
        print(f"Error: Census data file not found at {census_path}")
        return
    except Exception as e:
        print(f"Error reading or processing census data file {census_path}: {e}")
        return


    print("\n--- Initializing Model ---")
    try:
        model = RetirementPlanModel(initial_census_df=census_df, scenario_config=config)
    except Exception as e:
        print(f"Error initializing RetirementPlanModel: {e}")
        # Potentially print more details or traceback here
        return

    print("\n--- Running Simulation ---")
    projection_years = config.get('projection_years', 5)
    start_year = config.get('start_year', pd.Timestamp.now().year)
    print(f"Running simulation from {start_year} for {projection_years} years...")
    
    for i in range(projection_years):
        try:
            model.step()
        except Exception as e:
            print(f"Error during model step for year {model.current_year}: {e}")
            # Decide whether to stop or continue simulation
            break 

    print("\n--- Simulation Complete ---")

    print("\n--- Collecting and Saving Results ---")
    try:
        # Get data from the collector
        model_data = model.datacollector.get_model_vars_dataframe()
        agent_data = model.datacollector.get_agent_vars_dataframe()

        # Process agent data (multi-index: Step, AgentID)
        agent_data.reset_index(inplace=True)
        # Index is now columns named 'Step' and 'AgentID' (or original level names)
        # Rename 'Step' column if necessary (Mesa usually names it 'Step')
        if 'Step' in agent_data.columns:
            # Adjust Year to match simulation years (Year = Step + Start Year - 1)
            agent_data['Year'] = agent_data['Step'] + start_year - 1
            agent_data.drop(columns=['Step'], inplace=True) # Remove original step column
        else:
            print("Warning: 'Step' column not found after resetting index in agent data.")
            
        # Process model data (index: Step)
        model_data.index.names = ['YearStep'] # Use YearStep temporarily
        model_data.reset_index(inplace=True)
        # Adjust Year to match simulation years (Year = Step + Start Year - 1)
        model_data['Year'] = model_data['YearStep'] + start_year - 1
        model_data.drop(columns=['YearStep'], inplace=True) # Remove temporary step column
        
        # Print employment status counts for verification
        print("\n--- Employment Status Counts in Agent Data ---")
        status_counts = agent_data['EmploymentStatus'].value_counts()
        print(status_counts)
        
        # Check if 'New Hire Active' is present
        if 'New Hire Active' not in status_counts.index:
            print("WARNING: 'New Hire Active' status is missing from the results!")
            # Look for new hires in the model's population
            new_hire_count = sum(1 for a in model.population.values() if a.employment_status == "New Hire Active")
            print(f"Found {new_hire_count} agents with 'New Hire Active' status in the model population")

        # Ensure output directory exists
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Use only the basename of the prefix to avoid nested dirs
        base_prefix = os.path.basename(output_prefix)
        model_output_path = os.path.join(output_dir, f"{base_prefix}_model_results.csv")
        agent_output_path = os.path.join(output_dir, f"{base_prefix}_agent_results.csv")

        model_data.to_csv(model_output_path, index=False)
        agent_data.to_csv(agent_output_path, index=False)

        print(f"Model-level results saved to: {model_output_path}")
        print(f"Agent-level results saved to: {agent_output_path}")

    except Exception as e:
        print(f"Error collecting or saving results: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Agent-Based Retirement Plan Simulation")
    parser.add_argument(
        "--config", 
        type=str, 
        default="data/config.yaml", 
        help="Path to the scenario configuration YAML file."
    )
    parser.add_argument(
        "--census", 
        type=str, 
        default="data/census_data.csv", 
        help="Path to the initial census data CSV file."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="abm_simulation", 
        help="Prefix for the output result files (e.g., 'results/scenario_A')."
    )
    
    args = parser.parse_args()

    run_simulation(args.config, args.census, args.output)
    print("\n--- Script Finished ---")
