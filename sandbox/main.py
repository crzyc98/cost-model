import pandas as pd
import os
import traceback
from datetime import datetime

# Import from newly created modules
from config import PROJECTION_YEARS, ANNUAL_GROWTH_RATE, ANNUAL_TURNOVER_RATE, ANNUAL_COMP_INCREASE_RATE
from data_processing import load_and_clean_census
from ml_logic import build_training_data, train_turnover_model, ML_LIBS_AVAILABLE
from projection import project_census
from dummy_data import create_dummy_census_files

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        # 1. Define Historical Files (Use dummy data or provide actual paths)
        # Option A: Create Dummy Files
        historical_files = create_dummy_census_files(num_years=3, base_year=2022, n_employees=10000)

        # Option B: Use Actual Files (replace list with your paths)
        # historical_files = [
        #     '/path/to/your/census_2022.csv',
        #     '/path/to/your/census_2023.csv',
        #     '/path/to/your/census_2024.csv'
        # ]
        # historical_files.sort() # Ensure chronological order if using actual files

        if not historical_files:
            raise ValueError("No historical census files specified or generated.")

        # 2. Define Model Features
        # These are the features the ML model will be trained on and use for prediction.
        # Make sure these names match columns calculated in build_training_data
        # and prepared by prepare_features_for_model.
        model_features = ['age_at_period_start', 'tenure_at_period_start', 'comp_at_period_start']

        # 3. Build Training Data (if enough historical data)
        ml_model = None
        if ML_LIBS_AVAILABLE and len(historical_files) >= 2:
            X_train, y_train = build_training_data(historical_files, model_features)
            if X_train is not None and y_train is not None:
                ml_model = train_turnover_model(X_train, y_train)
                if ml_model is None:
                    print("ML Model training failed, will use rule-based fallback.")
            else:
                print("Could not build training data, will use rule-based fallback.")
        else:
            if not ML_LIBS_AVAILABLE:
                 print("ML libraries not available. Using rule-based scoring for projection.")
            else:
                 print("Not enough historical data for training (< 2 files). Using rule-based scoring.")


        # 4. Load the latest census data as the starting point for projection
        latest_census_file = historical_files[-1]
        required_cols_for_proj = ['ssn', 'birth_date', 'hire_date', 'termination_date', 'gross_compensation']
        start_df = load_and_clean_census(latest_census_file, {'required': required_cols_for_proj})

        if start_df is None:
            raise ValueError(f"Failed to load the starting census file: {latest_census_file}")

        # Filter to active employees at the start of the projection period
        last_plan_year_end_date = start_df['plan_year_end_date'].iloc[0]
        start_df = start_df[
            start_df['termination_date'].isna() | (start_df['termination_date'] > last_plan_year_end_date)
        ].copy()
        print(f"Starting projection with {len(start_df)} active employees from {last_plan_year_end_date.date()}")


        # 5. Run the projection
        projected_results = project_census(
            start_df=start_df,
            projection_model=ml_model, # Pass the trained model or None
            model_feature_names=model_features, # Use the consistent feature names
            projection_years=PROJECTION_YEARS,
            growth_rate=ANNUAL_GROWTH_RATE,
            turnover_rate=ANNUAL_TURNOVER_RATE, # Use config value
            comp_increase_rate=ANNUAL_COMP_INCREASE_RATE
        )

        # 6. Access and Process Results
        if projected_results:
            print("\n--- Accessing Projected Data Example (Year 1) ---")
            year_1_df = projected_results[1] # Access projection for year 1

            # Recalculate age/tenure at the *end* of the projected year for reporting
            proj_end_date_y1 = last_plan_year_end_date + pd.DateOffset(years=1)
            # Use utils functions if needed, or do it directly if simple
            year_1_df['age_at_year_end'] = year_1_df['birth_date'].apply(lambda x: (proj_end_date_y1.year - pd.to_datetime(x).year - ((proj_end_date_y1.month, proj_end_date_y1.day) < (pd.to_datetime(x).month, pd.to_datetime(x).day))))
            year_1_df['tenure_at_year_end'] = year_1_df.apply(lambda row: (proj_end_date_y1 - pd.to_datetime(row['hire_date'])).days / 365.25 if row['status'] == 'Active' else None, axis=1)


            print(year_1_df[['ssn', 'status', 'birth_date', 'hire_date', 'termination_date', 'gross_compensation',
                             'age_at_year_end', 'tenure_at_year_end']].head())
            print(f"\nYear 1 Projected Shape: {year_1_df.shape}")

            # Example: Save all projections to Excel
            output_filename = "projected_census_all_years.xlsx"
            with pd.ExcelWriter(output_filename) as writer:
                 for year, df in projected_results.items():
                     df.to_excel(writer, sheet_name=f'Year_{year}', index=False)
            print(f"\nSaved all projected years to {output_filename}")

            # --- Generate Summary Output ---
            print("\n--- Projection Summary ---")
            print(" Assumptions:")
            print(f"   - Projection Years: {PROJECTION_YEARS}")
            print(f"   - Annual Growth Rate: {ANNUAL_GROWTH_RATE:.1%}")
            print(f"   - Target Annual Turnover Rate: {ANNUAL_TURNOVER_RATE:.1%}")
            print(f"   - Annual Comp Increase Rate: {ANNUAL_COMP_INCREASE_RATE:.1%}")

            print("\n Results:")
            summary_data = []
            try:
                excel_file = pd.ExcelFile(output_filename)
                projection_start_year = int(os.path.basename(historical_files[-1]).split('_')[-1].split('.')[0]) + 1
                for i, sheet_name in enumerate(excel_file.sheet_names):
                    year_df = excel_file.parse(sheet_name)
                    year = projection_start_year + i
                    headcount = len(year_df)
                    total_comp = year_df['gross_compensation'].sum()
                    summary_data.append({
                        'Year': year,
                        'Participants': headcount, 
                        'Total Compensation': total_comp
                    })

                summary_df = pd.DataFrame(summary_data)
                summary_df['Total Compensation'] = summary_df['Total Compensation'].map('{:,.2f}'.format)
                print(summary_df.to_string(index=False))

            except FileNotFoundError:
                print(f"Error: Output file {output_filename} not found.")
            except Exception as e:
                print(f"Error generating summary: {e}")

    except ValueError as ve:
         print(f"\nConfiguration or Data Error: {ve}")
    except ImportError as ie:
         print(f"\nImport Error: {ie}. Make sure all required libraries are installed.")
         print("Try: pip install pandas numpy scikit-learn lightgbm")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the main process: {e}")
        traceback.print_exc()