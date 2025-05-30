#!/usr/bin/env python3
"""
Generate synthetic employee census for a single calendar year.

This script generates a synthetic employee census for the start_year specified in dev_tiny.yaml,
targeting approximately 100 active employees at year-end. The output contains all individuals
who were part of the workforce at any point during the year, allowing identification of:

1. Full-Year Actives: Employees active at start and end of year
2. New Hire Actives: Employees hired during year and still active at end
3. New Hire Terms: Employees hired during year and terminated before end
4. Experienced Terms: Employees active at start but terminated before end

Usage:
    python generate_census_seed.py

Output:
    Parquet file at the path specified by census_template_path in dev_tiny.yaml
"""

import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import logging

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent  # Go up one level from scripts/ to project root
sys.path.insert(0, str(project_root))

# Import project modules
from cost_model.state.schema import (
    EMP_ID, EMP_HIRE_DATE, EMP_BIRTH_DATE, EMP_GROSS_COMP, EMP_TERM_DATE,
    EMP_ACTIVE, EMP_TENURE, EMP_TENURE_BAND, EMP_LEVEL, EMP_LEVEL_SOURCE,
    EMP_STATUS_EOY, EMP_EXITED, SIMULATION_YEAR, EMP_DEFERRAL_RATE,
    EMP_CONTR, EMPLOYER_CORE, EMPLOYER_MATCH, IS_ELIGIBLE,
    SNAPSHOT_COLS, SNAPSHOT_DTYPES
)
from cost_model.state.tenure import assign_tenure_band, TENURE_BAND_CATEGORICAL_DTYPE
from cost_model.utils.id_generation import _generate_sequential_ids

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config_files() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load configuration from dev_tiny.yaml and hazard_defaults.yaml."""
    # Get the project root (one level up from the scripts directory)
    project_root = Path(__file__).parent.parent
    config_dir = project_root / "config"

    # Load dev_tiny.yaml
    dev_tiny_path = config_dir / "dev_tiny.yaml"
    with open(dev_tiny_path, 'r') as f:
        dev_tiny = yaml.safe_load(f)

    # Load hazard_defaults.yaml
    hazard_defaults_path = config_dir / "hazard_defaults.yaml"
    with open(hazard_defaults_path, 'r') as f:
        hazard_defaults = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {dev_tiny_path} and {hazard_defaults_path}")
    return dev_tiny, hazard_defaults


def extract_parameters(dev_tiny: Dict[str, Any], hazard_defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Extract relevant parameters from configuration files."""
    global_params = dev_tiny['global_parameters']

    params = {
        'start_year': global_params['start_year'],
        'new_hire_rate': global_params['new_hires']['new_hire_rate'],
        'new_hire_term_rate': hazard_defaults['termination']['base_rate_for_new_hire'],
        'avg_term_rate_existing': global_params['attrition']['annual_termination_rate'],
        'target_year_end_actives': 100,
        'min_working_age': global_params['min_working_age'],
        'max_working_age': global_params['max_working_age'],
        'new_hire_average_age': global_params['new_hire_average_age'],
        'new_hire_age_std_dev': global_params['new_hire_age_std_dev'],
        'census_template_path': global_params['census_template_path'],
        'job_levels': dev_tiny['job_levels'],
        'random_seed': global_params.get('random_seed', 79)
    }

    logger.info(f"Extracted parameters: start_year={params['start_year']}, "
                f"new_hire_rate={params['new_hire_rate']}, "
                f"new_hire_term_rate={params['new_hire_term_rate']}, "
                f"avg_term_rate_existing={params['avg_term_rate_existing']}")

    return params


def calculate_opening_headcount(params: Dict[str, Any]) -> Tuple[int, int]:
    """
    Calculate opening headcount and new hires using algebraic solution.

    Given:
    - expected_active_end = opening_active * (1 - avg_term_rate_existing) + new_hires * (1 - new_hire_term_rate)
    - new_hires = opening_active * new_hire_rate
    - expected_active_end = 100

    Solve for opening_active and new_hires.
    """
    target_end = params['target_year_end_actives']
    new_hire_rate = params['new_hire_rate']
    existing_survival_rate = 1 - params['avg_term_rate_existing']
    new_hire_survival_rate = 1 - params['new_hire_term_rate']

    # Substitute new_hires = opening_active * new_hire_rate into the first equation:
    # target_end = opening_active * existing_survival_rate + (opening_active * new_hire_rate) * new_hire_survival_rate
    # target_end = opening_active * (existing_survival_rate + new_hire_rate * new_hire_survival_rate)

    combined_factor = existing_survival_rate + new_hire_rate * new_hire_survival_rate
    opening_active = int(round(target_end / combined_factor))
    new_hires = int(round(opening_active * new_hire_rate))

    logger.info(f"Calculated opening_active={opening_active}, new_hires={new_hires}")
    logger.info(f"Expected year-end actives: {opening_active * existing_survival_rate + new_hires * new_hire_survival_rate:.1f}")

    return opening_active, new_hires


def generate_level_distribution(job_levels: List[Dict[str, Any]], num_employees: int,
                               is_new_hire: bool = False, rng: np.random.Generator = None) -> np.ndarray:
    """Generate job level distribution for employees."""
    if is_new_hire:
        # New hires are typically entry-level (levels 0-1)
        levels = [0, 1]
        weights = [0.7, 0.3]  # 70% level 0, 30% level 1
    else:
        # Existing employees follow a more distributed pattern based on promotion probabilities
        levels = [jl['level_id'] for jl in job_levels]
        # Use inverse of promotion probability as a proxy for steady-state distribution
        weights = [1.0 / max(jl['promotion_probability'], 0.01) for jl in job_levels]
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

    return rng.choice(levels, size=num_employees, p=weights)


def calculate_compensation(level_id: int, age: int, job_levels: List[Dict[str, Any]],
                         rng: np.random.Generator) -> float:
    """Calculate compensation based on job level and age."""
    level_config = next((jl for jl in job_levels if jl['level_id'] == level_id), None)
    if not level_config:
        logger.warning(f"No configuration found for level {level_id}, using default")
        return 50000.0

    base_salary = level_config['comp_base_salary']
    age_factor = level_config['comp_age_factor']
    std_dev = level_config['comp_stochastic_std_dev']
    min_comp = level_config['min_compensation']
    max_comp = level_config['max_compensation']

    # Calculate base compensation with age factor
    age_adjustment = (age - 25) * age_factor * base_salary  # Assume 25 as baseline age
    target_comp = base_salary + age_adjustment

    # Add stochastic variation
    if std_dev > 0:
        log_mean = np.log(max(target_comp, 1000)) - (std_dev ** 2) / 2
        comp = np.exp(rng.normal(log_mean, std_dev))
    else:
        comp = target_comp

    # Clamp to min/max bounds
    comp = max(min_comp, min(comp, max_comp))

    return round(comp, 2)


def generate_tenure_for_existing(num_employees: int, rng: np.random.Generator) -> np.ndarray:
    """Generate tenure distribution for existing employees (survivorship bias)."""
    # Use exponential distribution with higher mean to simulate survivorship bias
    # Most employees have been around for a while (otherwise they would have left)
    tenure_years = rng.exponential(scale=6.0, size=num_employees)

    # Clamp to reasonable bounds (0.5 to 25 years)
    tenure_years = np.clip(tenure_years, 0.5, 35.0)

    return tenure_years


def generate_ages(num_employees: int, mean_age: float, std_age: float,
                 min_age: int, max_age: int, rng: np.random.Generator) -> np.ndarray:
    """Generate age distribution."""
    ages = rng.normal(mean_age, std_age, num_employees)
    ages = np.clip(ages, min_age, max_age).astype(int)
    return ages


def generate_opening_snapshot(opening_active: int, params: Dict[str, Any],
                            rng: np.random.Generator) -> pd.DataFrame:
    """Generate the opening snapshot of existing employees."""
    logger.info(f"Generating opening snapshot with {opening_active} employees")

    start_year = params['start_year']
    job_levels = params['job_levels']

    # Generate employee IDs
    employee_ids = _generate_sequential_ids([], opening_active)

    # Generate levels for existing employees
    levels = generate_level_distribution(job_levels, opening_active, is_new_hire=False, rng=rng)

    # Generate tenure for existing employees (survivorship bias)
    tenure_years = generate_tenure_for_existing(opening_active, rng)

    # Generate ages based on tenure (older employees tend to have more tenure)
    base_ages = generate_ages(opening_active, 40, 10,
                             params['min_working_age'], params['max_working_age'], rng)
    # Adjust ages based on tenure to make it realistic
    ages = np.clip(base_ages + tenure_years * 0.5, params['min_working_age'], params['max_working_age']).astype(int)

    # Calculate hire dates based on tenure
    start_date = datetime(start_year, 1, 1)
    hire_dates = [start_date - timedelta(days=int(tenure * 365.25)) for tenure in tenure_years]

    # Calculate birth dates based on ages
    birth_dates = []
    for age, hire_date in zip(ages, hire_dates):
        birth_year = hire_date.year - age
        # Random month and day for birth
        birth_month = int(rng.integers(1, 13))
        birth_day = int(rng.integers(1, 29))  # Avoid leap year issues
        birth_dates.append(datetime(birth_year, birth_month, birth_day))

    # Calculate compensation
    compensations = [calculate_compensation(level, age, job_levels, rng)
                    for level, age in zip(levels, ages)]

    # Generate tenure bands
    tenure_bands = [assign_tenure_band(tenure) for tenure in tenure_years]

    # Create DataFrame
    data = {
        EMP_ID: employee_ids,
        EMP_HIRE_DATE: hire_dates,
        EMP_BIRTH_DATE: birth_dates,
        EMP_GROSS_COMP: compensations,
        EMP_TERM_DATE: [pd.NaT] * opening_active,  # Initially no terminations
        EMP_ACTIVE: [True] * opening_active,
        EMP_TENURE: tenure_years,
        EMP_TENURE_BAND: tenure_bands,
        EMP_LEVEL: levels,
        EMP_LEVEL_SOURCE: ['hire'] * opening_active,  # All existing employees were hired
        EMP_EXITED: [False] * opening_active,
        EMP_STATUS_EOY: ['Active'] * opening_active,  # Will be updated after terminations
        SIMULATION_YEAR: [start_year] * opening_active,
        EMP_DEFERRAL_RATE: rng.choice([0.0, 0.03, 0.05, 0.06], size=opening_active,
                                     p=[0.4, 0.3, 0.2, 0.1]),  # Simple deferral distribution
        EMP_CONTR: [0.0] * opening_active,  # Will be calculated later
        EMPLOYER_CORE: [0.0] * opening_active,  # Will be calculated later
        EMPLOYER_MATCH: [0.0] * opening_active,  # Will be calculated later
        IS_ELIGIBLE: [True] * opening_active,  # Assume all existing employees are eligible
    }

    df = pd.DataFrame(data)

    # Apply proper data types
    for col, dtype in SNAPSHOT_DTYPES.items():
        if col in df.columns:
            if col == EMP_TENURE_BAND:
                df[col] = df[col].astype(TENURE_BAND_CATEGORICAL_DTYPE)
            else:
                df[col] = df[col].astype(dtype)

    logger.info(f"Generated opening snapshot with {len(df)} employees")
    return df


def apply_terminations_to_existing(df: pd.DataFrame, term_rate: float,
                                  start_year: int, rng: np.random.Generator) -> pd.DataFrame:
    """Apply terminations to existing employees."""
    num_to_terminate = int(round(len(df) * term_rate))
    logger.info(f"Applying {num_to_terminate} terminations to existing employees (rate: {term_rate:.3f})")

    if num_to_terminate == 0:
        return df

    # Randomly select employees to terminate
    terminate_indices = rng.choice(len(df), size=num_to_terminate, replace=False)

    # Generate termination dates throughout the year
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(start_year, 12, 31)

    for idx in terminate_indices:
        # Random termination date during the year
        days_in_year = (end_date - start_date).days
        random_day = int(rng.integers(0, days_in_year))
        term_date = start_date + timedelta(days=random_day)

        df.loc[idx, EMP_TERM_DATE] = term_date
        df.loc[idx, EMP_ACTIVE] = False
        df.loc[idx, EMP_EXITED] = True
        df.loc[idx, EMP_STATUS_EOY] = 'Terminated'

    logger.info(f"Applied terminations to {num_to_terminate} existing employees")
    return df


def generate_new_hires(num_new_hires: int, params: Dict[str, Any],
                      existing_ids: List[str], rng: np.random.Generator) -> pd.DataFrame:
    """Generate new hire employees."""
    logger.info(f"Generating {num_new_hires} new hires")

    if num_new_hires == 0:
        return pd.DataFrame(columns=SNAPSHOT_COLS)

    start_year = params['start_year']
    job_levels = params['job_levels']

    # Generate employee IDs (continue from existing)
    employee_ids = _generate_sequential_ids(existing_ids, num_new_hires)

    # Generate levels for new hires (entry-level)
    levels = generate_level_distribution(job_levels, num_new_hires, is_new_hire=True, rng=rng)

    # Generate ages for new hires
    ages = generate_ages(num_new_hires, params['new_hire_average_age'],
                        params['new_hire_age_std_dev'],
                        params['min_working_age'], params['max_working_age'], rng)

    # Generate hire dates throughout the year
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(start_year, 12, 31)
    days_in_year = (end_date - start_date).days

    hire_dates = []
    for _ in range(num_new_hires):
        random_day = int(rng.integers(0, days_in_year))
        hire_date = start_date + timedelta(days=random_day)
        hire_dates.append(hire_date)

    # Calculate birth dates based on ages and hire dates
    birth_dates = []
    for age, hire_date in zip(ages, hire_dates):
        birth_year = hire_date.year - age
        birth_month = int(rng.integers(1, 13))
        birth_day = int(rng.integers(1, 29))
        birth_dates.append(datetime(birth_year, birth_month, birth_day))

    # Calculate tenure (less than 1 year)
    tenure_years = [(datetime(start_year, 12, 31) - hire_date).days / 365.25
                   for hire_date in hire_dates]

    # Calculate compensation
    compensations = [calculate_compensation(level, age, job_levels, rng)
                    for level, age in zip(levels, ages)]

    # Generate tenure bands (all should be "<1")
    tenure_bands = [assign_tenure_band(tenure) for tenure in tenure_years]

    # Create DataFrame
    data = {
        EMP_ID: employee_ids,
        EMP_HIRE_DATE: hire_dates,
        EMP_BIRTH_DATE: birth_dates,
        EMP_GROSS_COMP: compensations,
        EMP_TERM_DATE: [pd.NaT] * num_new_hires,  # Initially no terminations
        EMP_ACTIVE: [True] * num_new_hires,
        EMP_TENURE: tenure_years,
        EMP_TENURE_BAND: tenure_bands,
        EMP_LEVEL: levels,
        EMP_LEVEL_SOURCE: ['hire'] * num_new_hires,
        EMP_EXITED: [False] * num_new_hires,
        EMP_STATUS_EOY: ['Active'] * num_new_hires,  # Will be updated after terminations
        SIMULATION_YEAR: [start_year] * num_new_hires,
        EMP_DEFERRAL_RATE: rng.choice([0.0, 0.03, 0.05], size=num_new_hires,
                                     p=[0.6, 0.3, 0.1]),  # Lower participation for new hires
        EMP_CONTR: [0.0] * num_new_hires,
        EMPLOYER_CORE: [0.0] * num_new_hires,
        EMPLOYER_MATCH: [0.0] * num_new_hires,
        IS_ELIGIBLE: [True] * num_new_hires,  # Assume immediate eligibility
    }

    df = pd.DataFrame(data)

    # Apply proper data types
    for col, dtype in SNAPSHOT_DTYPES.items():
        if col in df.columns:
            if col == EMP_TENURE_BAND:
                df[col] = df[col].astype(TENURE_BAND_CATEGORICAL_DTYPE)
            else:
                df[col] = df[col].astype(dtype)

    logger.info(f"Generated {len(df)} new hires")
    return df


def apply_terminations_to_new_hires(df: pd.DataFrame, term_rate: float,
                                   start_year: int, rng: np.random.Generator) -> pd.DataFrame:
    """Apply terminations to new hire employees."""
    num_to_terminate = int(round(len(df) * term_rate))
    logger.info(f"Applying {num_to_terminate} terminations to new hires (rate: {term_rate:.3f})")

    if num_to_terminate == 0:
        return df

    # Randomly select new hires to terminate
    terminate_indices = rng.choice(len(df), size=num_to_terminate, replace=False)

    for idx in terminate_indices:
        hire_date = df.loc[idx, EMP_HIRE_DATE]

        # Termination date must be after hire date but before year end
        year_end = datetime(start_year, 12, 31)
        days_employed = (year_end - hire_date).days

        if days_employed > 0:
            # Random termination date between hire date and year end
            random_day = int(rng.integers(1, days_employed + 1))
            term_date = hire_date + timedelta(days=random_day)
        else:
            # If hired very late in year, terminate at year end
            term_date = year_end

        df.loc[idx, EMP_TERM_DATE] = term_date
        df.loc[idx, EMP_ACTIVE] = False
        df.loc[idx, EMP_EXITED] = True
        df.loc[idx, EMP_STATUS_EOY] = 'Terminated'

        # Update tenure to reflect actual time worked
        actual_tenure = (term_date - hire_date).days / 365.25
        df.loc[idx, EMP_TENURE] = actual_tenure
        df.loc[idx, EMP_TENURE_BAND] = assign_tenure_band(actual_tenure)

    logger.info(f"Applied terminations to {num_to_terminate} new hires")
    return df


def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived fields to help identify the four employee categories."""
    # Add flag to identify new hires vs existing employees
    df['WasNewHireInSimYear'] = df[EMP_HIRE_DATE].dt.year == df[SIMULATION_YEAR]

    # Add original hire date (same as hire date for this single-year simulation)
    df['OriginalHireDate'] = df[EMP_HIRE_DATE]

    # Add simulation year start date
    df['SimYearStartDate'] = df.apply(
        lambda row: row[EMP_HIRE_DATE] if row['WasNewHireInSimYear']
        else datetime(row[SIMULATION_YEAR], 1, 1), axis=1
    )

    # Add status and level fields for year start and end
    df['StatusAtYearEnd'] = df[EMP_STATUS_EOY]
    df['JobLevelAtYearStart'] = df[EMP_LEVEL]  # Simplified - no promotions in this version
    df['JobLevelAtYearEnd'] = df[EMP_LEVEL]

    # Add tenure at year start
    df['TenureAtYearStart'] = df.apply(
        lambda row: 0.0 if row['WasNewHireInSimYear']
        else row[EMP_TENURE] - (datetime(row[SIMULATION_YEAR], 12, 31) - datetime(row[SIMULATION_YEAR], 1, 1)).days / 365.25,
        axis=1
    )

    # Add age at year start
    df['AgeAtYearStart'] = df.apply(
        lambda row: (datetime(row[SIMULATION_YEAR], 1, 1) - row[EMP_BIRTH_DATE]).days / 365.25
        if not row['WasNewHireInSimYear']
        else (row[EMP_HIRE_DATE] - row[EMP_BIRTH_DATE]).days / 365.25,
        axis=1
    ).round().astype(int)

    # Add compensation fields
    df['CompensationAtYearStart'] = df[EMP_GROSS_COMP]
    df['CompensationAtYearEnd'] = df[EMP_GROSS_COMP]  # No raises in this simplified version

    return df


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics about the generated census."""
    logger.info("=== CENSUS SUMMARY STATISTICS ===")

    total_employees = len(df)
    logger.info(f"Total employees in census: {total_employees}")

    # Four categories
    full_year_actives = df[(~df['WasNewHireInSimYear']) & (df['StatusAtYearEnd'] == 'Active')]
    new_hire_actives = df[(df['WasNewHireInSimYear']) & (df['StatusAtYearEnd'] == 'Active')]
    new_hire_terms = df[(df['WasNewHireInSimYear']) & (df['StatusAtYearEnd'] == 'Terminated')]
    experienced_terms = df[(~df['WasNewHireInSimYear']) & (df['StatusAtYearEnd'] == 'Terminated')]

    logger.info(f"Full-Year Actives: {len(full_year_actives)}")
    logger.info(f"New Hire Actives: {len(new_hire_actives)}")
    logger.info(f"New Hire Terms: {len(new_hire_terms)}")
    logger.info(f"Experienced Terms: {len(experienced_terms)}")
    logger.info(f"Total Year-End Actives: {len(full_year_actives) + len(new_hire_actives)}")

    # Verification
    assert len(full_year_actives) + len(new_hire_actives) + len(new_hire_terms) + len(experienced_terms) == total_employees

    # Level distribution
    logger.info("\nLevel distribution:")
    level_counts = df[EMP_LEVEL].value_counts().sort_index()
    for level, count in level_counts.items():
        logger.info(f"  Level {level}: {count} employees")

    # Tenure band distribution
    logger.info("\nTenure band distribution:")
    tenure_counts = df[EMP_TENURE_BAND].value_counts()
    for band, count in tenure_counts.items():
        logger.info(f"  {band}: {count} employees")


def main():
    """Main execution function."""
    logger.info("Starting census seed generation")

    try:
        # Load configuration
        logger.info("Loading configuration files...")
        dev_tiny, hazard_defaults = load_config_files()
        logger.info("Extracting parameters...")
        params = extract_parameters(dev_tiny, hazard_defaults)

        # Set random seed for reproducibility
        rng = np.random.default_rng(params['random_seed'])
        logger.info(f"Using random seed: {params['random_seed']}")

        # Calculate opening headcount and new hires
        opening_active, num_new_hires = calculate_opening_headcount(params)

        # Generate opening snapshot
        opening_df = generate_opening_snapshot(opening_active, params, rng)

        # Apply terminations to existing employees
        opening_df = apply_terminations_to_existing(
            opening_df, params['avg_term_rate_existing'], params['start_year'], rng
        )

        # Generate new hires
        new_hires_df = generate_new_hires(num_new_hires, params, opening_df[EMP_ID].tolist(), rng)

        # Apply terminations to new hires
        new_hires_df = apply_terminations_to_new_hires(
            new_hires_df, params['new_hire_term_rate'], params['start_year'], rng
        )

        # Combine all employees
        all_employees_df = pd.concat([opening_df, new_hires_df], ignore_index=True)

        # Add derived fields for analysis
        all_employees_df = add_derived_fields(all_employees_df)

        # Print summary statistics
        print_summary_statistics(all_employees_df)

        # Save to parquet file
        output_path = Path(params['census_template_path'])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Select only the core schema columns for output
        output_df = all_employees_df[SNAPSHOT_COLS + ['WasNewHireInSimYear', 'OriginalHireDate',
                                                     'SimYearStartDate', 'StatusAtYearEnd',
                                                     'JobLevelAtYearStart', 'JobLevelAtYearEnd',
                                                     'TenureAtYearStart', 'AgeAtYearStart',
                                                     'CompensationAtYearStart', 'CompensationAtYearEnd']]

        output_df.to_parquet(output_path, index=False)
        logger.info(f"Census saved to: {output_path}")
        logger.info(f"Output file contains {len(output_df)} employee records")

        logger.info("Census seed generation completed successfully")

    except Exception as e:
        logger.error(f"Error during census generation: {e}")
        raise


if __name__ == "__main__":
    main()
