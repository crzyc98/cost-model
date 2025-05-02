#!/usr/bin/env python3
"""
Generate historical dummy census files for retirement plan modeling.

This script orchestrates the generation process year by year, calling helper
functions from cost_model.utils.census_generation_helpers to handle the details of
employee record creation, termination selection, and derived field calculation.
"""
import sys
from pathlib import Path
import os

# --- Add project root to Python path FIRST ---
# This ensures utils can be found
try:
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    # Optional: print(f"DEBUG: Added project root to sys.path: {project_root}")
except Exception as e:
    print(f"Error determining project root or modifying sys.path: {e}")
    sys.exit(1) # Exit if we can't setup the path

# --- Now perform imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
import logging
import argparse
from numpy.random import default_rng, Generator
from typing import Dict, List, Optional, Set

# --- Import from utils AFTER path modification ---
try:
    # Import the refactored helper functions
    from cost_model.utils.census_generation_helpers import (
        generate_employee_record,
        calculate_derived_fields,
        select_weighted_terminations
    )
    # Import column constants
    from cost_model.utils.columns import (
        EMP_SSN, EMP_ROLE, EMP_BIRTH_DATE, EMP_HIRE_DATE, EMP_TERM_DATE,
        EMP_GROSS_COMP, EMP_PLAN_YEAR_COMP, EMP_CAPPED_COMP,
        EMP_DEFERRAL_RATE, EMP_CONTR, EMPLOYER_CORE, EMPLOYER_MATCH
    )
except ImportError as e:
    print(f"Error importing from cost_model.utils: {e}. Make sure cost_model/utils directory exists and contains census_generation_helpers.py and columns.py relative to {project_root}.")
    # Define fallbacks if running standalone or utils is missing
    print("Warning: Using fallback column names and potentially missing helper functions.")
    EMP_SSN, EMP_ROLE, EMP_BIRTH_DATE, EMP_HIRE_DATE, EMP_TERM_DATE = 'employee_ssn', 'employee_role', 'employee_birth_date', 'employee_hire_date', 'employee_termination_date'
    EMP_GROSS_COMP, EMP_PLAN_YEAR_COMP, EMP_CAPPED_COMP = 'employee_gross_compensation', 'employee_plan_year_compensation', 'employee_capped_compensation'
    EMP_DEFERRAL_RATE, EMP_CONTR = 'employee_deferral_rate', 'employee_contribution'
    EMPLOYER_CORE, EMPLOYER_MATCH = 'employer_core_contribution', 'employer_match_contribution'
    # Define dummy functions if helpers are missing to avoid NameError later
    def generate_employee_record(*args, **kwargs): raise NotImplementedError("generate_employee_record missing")
    def calculate_derived_fields(*args, **kwargs): raise NotImplementedError("calculate_derived_fields missing")
    def select_weighted_terminations(*args, **kwargs): raise NotImplementedError("select_weighted_terminations missing")


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s")
logger = logging.getLogger(Path(__file__).stem) # Use script name for logger

# Suppress specific Pandas warnings if needed (optional)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# --- Main Census Generation Function ---
def create_dummy_census_files(
    # --- File & Population Settings ---
    num_years: int = 5,
    base_year: int = 2024,
    total_population: int = 1000, # Default population size
    termination_rate: float = 0.10,
    # --- Age Distribution ---
    age_mean: float = 42,
    age_std_dev: float = 12,
    min_working_age: int = 18,
    max_working_age: int = 70,
    # --- Tenure Distribution (For Initial Population) ---
    tenure_mean_years: float = 7,
    tenure_std_dev_years: float = 5,
    max_tenure_years: float = 40,
    # --- Role Configuration ---
    role_distribution: Dict[str, float] = {
        'Staff': 0.75, 'Manager': 0.20, 'Executive': 0.05
    },
    role_compensation_params: Dict[str, Dict] = {
        'Staff': {'comp_base_salary': 50000, 'comp_increase_per_age_year': 300, 'comp_increase_per_tenure_year': 500, 'comp_log_mean_factor': 1.0, 'comp_spread_sigma': 0.20, 'comp_min_salary': 28000},
        'Manager': {'comp_base_salary': 150000, 'comp_increase_per_age_year': 600, 'comp_increase_per_tenure_year': 1000, 'comp_log_mean_factor': 1.0, 'comp_spread_sigma': 0.25, 'comp_min_salary': 60000},
        'Executive': {'comp_base_salary': 250000, 'comp_increase_per_age_year': 1500, 'comp_increase_per_tenure_year': 3000, 'comp_log_mean_factor': 1.1, 'comp_spread_sigma': 0.35, 'comp_min_salary': 120000}
    },
    # --- Deferral Distribution ---
    deferral_distribution: Dict[float, float] = {
        0: 0.10, 1: 0.05, 2: 0.04, 3: 0.40, 4: 0.10,
        5: 0.10, 6: 0.06, 7: 0.03, 8: 0.01, 9: 0.01, 10: 0.10
    },
    # --- Compensation Increase for Survivors ---
    annual_comp_increase_mean: float = 0.03,
    annual_comp_increase_std: float = 0.015,
    # --- Employer Contributions ---
    employer_nec_rate: float = 0.02,
    employer_match_rate: float = 1.00,
    employer_match_cap_deferral_perc: float = 0.03,
    # --- Output Options ---
    output_dir: str = ".",
    file_prefix: str = "dummy_census_",
    seed: Optional[int] = None,
) -> List[str]:
    """
    Creates configurable dummy historical census CSV files.

    Orchestrates the year-by-year generation process, calling helper functions
    for detailed logic. Each file (census_YYYY.csv) contains records relevant
    for plan contributions in year YYYY (actives + those terminated in YYYY).
    """
    logger.info("Creating %d dummy census files from %d to %d", num_years, base_year - num_years + 1, base_year)
    rng = default_rng(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
    all_ssns = set() # Keep track of all generated SSNs for uniqueness
    generated_files = []
    # Initialize active population DF for the loop
    current_active_population_df = pd.DataFrame()
    employee_counter = 0 # For unique IDs in generate_employee_record

    # --- IRS Limits ---
    # Define limits within the function or load from external source
    IRS_LIMITS = {
        2020: {'comp_limit': 285000, 'deferral_limit': 19500, 'catch_up': 6500},
        2021: {'comp_limit': 290000, 'deferral_limit': 19500, 'catch_up': 6500},
        2022: {'comp_limit': 305000, 'deferral_limit': 20500, 'catch_up': 6500},
        2023: {'comp_limit': 330000, 'deferral_limit': 22500, 'catch_up': 7500},
        2024: {'comp_limit': 345000, 'deferral_limit': 23000, 'catch_up': 7500},
        2025: {'comp_limit': 355000, 'deferral_limit': 23500, 'catch_up': 7500}, # Example projection
    }
    latest_known_year = max(IRS_LIMITS.keys())
    for y in range(latest_known_year + 1, base_year + 2): # Ensure coverage for base_year
         if y not in IRS_LIMITS:
             IRS_LIMITS[y] = IRS_LIMITS[latest_known_year]

    # --- Normalize Distributions ---
    def _normalize_dist(dist: dict, name: str) -> dict:
        """Normalizes probability distribution if it doesn't sum to 1."""
        if not dist: return {}
        total_prob = sum(dist.values())
        if total_prob <= 0:
            logger.error(f"{name} distribution probabilities sum to zero or less. Check config.")
            return {k: 1.0/len(dist) for k in dist} # Fallback to uniform
        if not np.isclose(total_prob, 1.0):
            logger.warning(f"{name} distribution probabilities sum to {total_prob:.4f}. Normalizing.")
            factor = 1.0 / total_prob
            return {k: v * factor for k, v in dist.items()}
        return dist

    role_distribution = _normalize_dist(role_distribution, "Role")
    # Ensure keys are floats for deferral rates before normalizing
    deferral_distribution_float_keys = {float(k): v for k, v in deferral_distribution.items()}
    deferral_distribution = _normalize_dist(deferral_distribution_float_keys, "Deferral")


    start_year = base_year - num_years + 1

    # --- Generate Population Year by Year ---
    for year in range(start_year, base_year + 1):
        plan_end_date = datetime(year, 12, 31)
        logger.info("--- Processing Year %d ---", year)

        # Define current_census_population_df at the start of the loop scope
        current_census_population_df = pd.DataFrame()

        if year == start_year:
            # --- Generate Initial Population ---
            logger.info("Generating initial population of %d...", total_population)
            employee_records = []
            for i in range(total_population):
                employee_counter += 1
                # Use the imported helper function
                record = generate_employee_record(
                    year=year, plan_end_date=plan_end_date, is_new_hire=False, existing_ssns=all_ssns,
                    min_working_age=min_working_age, max_working_age=max_working_age,
                    age_mean=age_mean, age_std_dev=age_std_dev,
                    tenure_mean_years=tenure_mean_years, tenure_std_dev_years=tenure_std_dev_years,
                    max_tenure_years=max_tenure_years,
                    role_distribution=role_distribution, role_compensation_params=role_compensation_params,
                    deferral_distribution=deferral_distribution, unique_id=employee_counter, rng=rng
                )
                employee_records.append(record)
            current_active_population_df = pd.DataFrame(employee_records)
            current_active_population_df[EMP_TERM_DATE] = pd.NaT # No terms yet
            logger.info("Initial population generated (%d rows).", len(current_active_population_df))
            # For the first year, the census population is just the active population
            current_census_population_df = current_active_population_df.copy()

        else: # Subsequent years: Simulate flow from previous year
            # previous_population_df now holds ONLY the active from end of last year
            previous_population_df = current_active_population_df.copy()
            num_previous = len(previous_population_df)
            logger.info("Starting year %d with %d active employees from %d.", year, num_previous, year-1)

            # --- Simulate Terminations from Previous Year's Active Pop ---
            num_to_terminate = int(round(num_previous * termination_rate))
            terminated_records_this_year = pd.DataFrame() # Reset for this iteration
            survivor_df = previous_population_df.copy()

            if num_to_terminate > 0 and num_previous > 0:
                if num_to_terminate >= num_previous:
                    logger.warning(f"Termination rate ({termination_rate:.1%}) results in terminating all {num_previous} employees.")
                    indices_to_terminate = previous_population_df.index
                else:
                    logger.info("Simulating %d terminations from prior year pop (weighted by low tenure/comp)...", num_to_terminate)
                    # Use the imported helper function
                    indices_to_terminate = select_weighted_terminations(previous_population_df, num_to_terminate, rng)

                # Assign termination date *within the current year* (YYYY)
                terminated_records_this_year = previous_population_df.loc[indices_to_terminate].copy()
                year_start_dt = datetime(year, 1, 1)
                days_range = (plan_end_date - year_start_dt).days + 1
                term_offsets = rng.integers(0, days_range, size=len(terminated_records_this_year))
                terminated_records_this_year[EMP_TERM_DATE] = year_start_dt + pd.to_timedelta(term_offsets, unit='D')

                survivor_df = previous_population_df.drop(indices_to_terminate)
                logger.info("%d survivors remaining.", len(survivor_df))
            else:
                logger.info("No terminations simulated this cycle.")


            # --- Update Survivors for the Current Year End ---
            logger.info("Updating %d survivors for end of %d...", len(survivor_df), year)
            if not survivor_df.empty:
                # Increment tenure approx 1 year
                # Ensure 'tenure' column exists and is numeric before incrementing
                if 'tenure' not in survivor_df.columns: survivor_df['tenure'] = 0.0 # Initialize if missing
                survivor_df['tenure'] = pd.to_numeric(survivor_df['tenure'], errors='coerce').fillna(0.0) + 1

                # Apply random compensation increase
                increase_factor = rng.normal(loc=1 + annual_comp_increase_mean, scale=annual_comp_increase_std, size=len(survivor_df)).clip(min=0.5)
                survivor_df[EMP_GROSS_COMP] = (survivor_df[EMP_GROSS_COMP] * increase_factor).round(2)


            # --- Add New Hires for the Current Year ---
            num_survivors = len(survivor_df)
            num_new_hires_needed = max(0, total_population - num_survivors)
            logger.info("Adding %d new hires for %d...", num_new_hires_needed, year)
            new_hire_records = []
            if num_new_hires_needed > 0:
                for i in range(num_new_hires_needed):
                    employee_counter += 1
                    # Use the imported helper function
                    record = generate_employee_record(
                        year=year, plan_end_date=plan_end_date, is_new_hire=True, existing_ssns=all_ssns,
                        min_working_age=min_working_age, max_working_age=max_working_age,
                        age_mean=age_mean, age_std_dev=age_std_dev,
                        tenure_mean_years=tenure_mean_years, tenure_std_dev_years=tenure_std_dev_years,
                        max_tenure_years=max_tenure_years,
                        role_distribution=role_distribution, role_compensation_params=role_compensation_params,
                        deferral_distribution=deferral_distribution, unique_id=employee_counter, rng=rng
                    )
                    new_hire_records.append(record)
            new_hires_df = pd.DataFrame(new_hire_records)


            # --- Define the population relevant for THIS year's census ---
            # Combine survivors and new hires (active at year end)
            current_active_end_of_year_df = pd.concat([survivor_df, new_hires_df], ignore_index=True)

            # Add back those terminated *during this year*
            if not terminated_records_this_year.empty:
                 logger.info("Including %d terminated records in census for %d", len(terminated_records_this_year), year)
                 # Ensure columns align before concat
                 cols_active = set(current_active_end_of_year_df.columns)
                 cols_term = set(terminated_records_this_year.columns)
                 # Add missing columns to term records with default nulls if needed
                 for col in cols_active:
                     if col not in terminated_records_this_year.columns:
                         dtype = current_active_end_of_year_df[col].dtype
                         na_val = pd.NA
                         if pd.api.types.is_datetime64_any_dtype(dtype): na_val = pd.NaT
                         elif pd.api.types.is_numeric_dtype(dtype): na_val = np.nan
                         terminated_records_this_year[col] = na_val

                 # Ensure term records have same columns as active before concat
                 # Use list comprehension to handle potential missing columns gracefully
                 cols_to_align = [col for col in current_active_end_of_year_df.columns if col in terminated_records_this_year.columns]
                 terminated_records_aligned = terminated_records_this_year[cols_to_align]

                 current_census_population_df = pd.concat(
                     [current_active_end_of_year_df, terminated_records_aligned],
                     ignore_index=True, sort=False
                 )
            else:
                 # If no terminations, the census pop is just the active pop
                 current_census_population_df = current_active_end_of_year_df.copy()

            logger.info("Total relevant population for %d census: %d employees.", year, len(current_census_population_df))


        # --- Calculate Derived Fields & Save ---
        # This check now correctly uses the variable assigned in both branches
        if not current_census_population_df.empty:
             # Ensure required date columns are datetime objects before calculation
             for col in [EMP_BIRTH_DATE, EMP_HIRE_DATE, EMP_TERM_DATE]:
                 if col in current_census_population_df.columns:
                    current_census_population_df[col] = pd.to_datetime(current_census_population_df[col], errors='coerce')

             # Calculate derived fields on the COMBINED population using imported helper
             df_final_year = calculate_derived_fields(
                 current_census_population_df, year, IRS_LIMITS,
                 employer_nec_rate, employer_match_rate, employer_match_cap_deferral_perc
             )

             # Format dates for CSV output
             for col in [EMP_BIRTH_DATE, EMP_HIRE_DATE, EMP_TERM_DATE]:
                 if col in df_final_year.columns:
                     if pd.api.types.is_datetime64_any_dtype(df_final_year[col]):
                         # Format valid dates, leave NaT as empty string
                         df_final_year[col] = df_final_year[col].dt.strftime('%Y-%m-%d').fillna('')
                     else: # Handle cases where coercion might have failed
                         df_final_year[col] = ''


             # Save to CSV
             output_filename = output_path / f"{file_prefix}{year}.csv"
             df_final_year.to_csv(output_filename, index=False, date_format='%Y-%m-%d') # date_format might be redundant now
             generated_files.append(str(output_filename))
             logger.info("Saved Combined Census: %s (%d rows)", output_filename, len(df_final_year))

             # Prepare for NEXT year: only carry forward the ACTIVE population
             # Active are those in the final DF with no term date OR term date after current year end
             # Use the original datetime column for comparison, not the formatted string
             active_mask = current_census_population_df[EMP_TERM_DATE].isna() | \
                           (current_census_population_df[EMP_TERM_DATE] > plan_end_date)

             # Select active rows using the boolean mask from the *original* combined DF
             current_active_population_df = current_census_population_df[active_mask].copy()
             logger.debug("Carrying forward %d active employees to year %d", len(current_active_population_df), year + 1)


        else: # Handle case where combined pop is empty
            logger.info("Skipping save for %d due to empty population.", year)
            current_active_population_df = pd.DataFrame() # Reset for next year


    logger.info("Dummy file creation complete.")
    return generated_files

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Generate dummy census files.")
    # Add arguments with defaults from the function signature
    p.add_argument('--years', type=int, default=5, help="Number of years ending at base-year")
    p.add_argument('--base-year', type=int, default=2024, help="Final year of census data to generate")
    p.add_argument('--pop', type=int, default=1000, help="Target total population size for each year")
    p.add_argument('--term-rate', type=float, default=0.10, help="Approx annual termination rate")
    p.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility")
    p.add_argument('--outdir', type=Path, default=Path('./data/generated_census'), help="Output directory")
    p.add_argument('--prefix', type=str, default="census_", help="Prefix for output filenames")
    # Could add arguments for other parameters like age/tenure means, etc. if desired

    args = p.parse_args()

    # Create the output directory if it doesn't exist
    args.outdir.mkdir(parents=True, exist_ok=True)

    files = create_dummy_census_files(
        num_years=args.years,
        base_year=args.base_year,
        total_population=args.pop,
        termination_rate=args.term_rate,
        seed=args.seed,
        output_dir=str(args.outdir),
        file_prefix=args.prefix
        # Pass other defaults or load from a config YAML here
    )
    logger.info("Generated files: %s", files)
