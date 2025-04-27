# utils/projection_utils.py

"""
Core projection logic for the census data, orchestrating six clear steps:
1. Compensation bump
2. Termination sampling
3. Filter out early terminations
4. New-hire generation & compensation sampling
5. ML- or rule-based turnover
6. Plan rules & contributions
"""

import time
import logging

import pandas as pd
import numpy as np

from utils.date_utils import calculate_age, calculate_tenure
from utils.sandbox_utils import generate_new_hires
from utils.sampling.terminations import sample_terminations
from utils.sampling.new_hires import sample_new_hire_compensation
from utils.compensation.bump import apply_comp_increase
from utils.sampling.salary import SalarySampler, DefaultSalarySampler
from utils.ml.ml_utils import try_load_ml_model, predict_turnover
from utils.plan_rules import (
    determine_eligibility,
    apply_auto_enrollment,
    apply_auto_increase,
    calculate_contributions
)

logger = logging.getLogger(__name__)


def project_census(
    start_df: pd.DataFrame,
    scenario_config: dict,
    baseline_scenario_config: dict
) -> dict[int, pd.DataFrame]:
    projection_years     = scenario_config['projection_years']
    start_year           = scenario_config['start_year']
    comp_increase_rate   = scenario_config.get('comp_increase_rate', 0.0)
    hire_rate            = scenario_config.get('hire_rate', 0.0)
    termination_rate     = scenario_config.get('termination_rate', 0.0)
    seed                 = scenario_config.get('random_seed')
    # Initialize pluggable salary sampler
    sampler: SalarySampler = scenario_config.get('salary_sampler', DefaultSalarySampler())

    # Load ML model and features if configured
    model_path = scenario_config.get("ml_model_path", "")
    features_path = scenario_config.get("model_features_path", "")
    if model_path and features_path:
        ml_pair = try_load_ml_model(model_path, features_path)
        projection_model, feature_cols = (None, []) if ml_pair is None else ml_pair
    else:
        projection_model, feature_cols = None, []

    # Prepare snapshots
    projected_data = {}
    current_df     = start_df.copy()
    prev_hire_salaries = start_df['gross_compensation'].dropna()
    prev_term_salaries = start_df['gross_compensation'].dropna()

    logger.info("Starting projection for '%s' from %s", scenario_config['scenario_name'], start_year)
    logger.info("Initial headcount: %d", len(current_df))

    for year_num in range(1, projection_years + 1):
        sim_year       = start_year + year_num - 1
        start_date     = pd.Timestamp(f"{sim_year}-01-01")
        end_date       = pd.Timestamp(f"{sim_year}-12-31")

        logger.info("Year %d (%s) ▶ headcount %d", year_num, sim_year, len(current_df))

        # 1. Second-year bump (pluggable) + all-others via comp_increase_rate
        if 'gross_compensation' in current_df:
            # Apply second-year bump via sampler
            current_df['gross_compensation'] = sampler.sample_second_year(
                current_df,
                comp_col='gross_compensation',
                dist=scenario_config.get('second_year_compensation_dist', {}),
                rate=comp_increase_rate,
                seed=seed
            )
            # Apply global comp increase to others
            if 'tenure' in current_df.columns:
                mask_2nd = current_df['tenure'] == 1
                mask_rest = ~mask_2nd
            else:
                mask_rest = pd.Series(False, index=current_df.index)
            current_df.loc[mask_rest, 'gross_compensation'] *= (1 + comp_increase_rate)
        else:
            logger.warning("Missing 'gross_compensation' column, skipping bump")

        # 2. Sample terminations (assign dates)
        current_df = sample_terminations(
            current_df,
            'hire_date',
            termination_rate,
            end_date,
            seed
        )
        # 2b. Sample termination salaries via sampler
        term_here = current_df['termination_date'].between(start_date, end_date)
        n_term = term_here.sum()
        if n_term:
            draws = sampler.sample_terminations(prev_term_salaries, size=n_term, seed=seed)
            current_df.loc[term_here, 'gross_compensation'] = draws.values

        # 3. Drop those who terminated before this period
        current_df = current_df[
            current_df['termination_date'].isna() |
            (current_df['termination_date'] >= start_date)
        ].copy()

        # 4. Generate new hires & sample their compensation
        if scenario_config.get('maintain_headcount', True):
            needed = max(0, len(start_df) - len(current_df))
        else:
            needed = int(len(current_df) * hire_rate)

        if needed > 0:
            nh_df = generate_new_hires(
                num_hires=needed,
                hire_year=sim_year,
                role_distribution=scenario_config.get('role_distribution'),
                role_compensation_params=scenario_config.get('role_compensation_params'),
                age_mean=scenario_config.get('age_mean'),
                age_std_dev=scenario_config.get('age_std_dev'),
                min_working_age=scenario_config.get('min_working_age'),
                max_working_age=scenario_config.get('max_working_age'),
                scenario_config=scenario_config
            )
            nh_df = sample_new_hire_compensation(
                nh_df,                          # DataFrame of new hires
                'gross_compensation',           # column to fill
                prev_hire_salaries.values,      # historical salary array
                seed
            )
            current_df = pd.concat([current_df, nh_df], ignore_index=True)
            logger.info("Generated %d new hires", needed)

        # 5. ML- or rule-based turnover
        if projection_model:
            probs = predict_turnover(
                current_df,
                projection_model,
                feature_cols,
                end_date,
                seed
            )
            current_df = sample_terminations(
                current_df,
                'hire_date',
                probs,
                end_date,
                None
            )
            # Sample termination salaries for ML-based terminations
            term_here = current_df['termination_date'].between(start_date, end_date)
            n_term = term_here.sum()
            if n_term:
                draws = sampler.sample_terminations(prev_term_salaries, size=n_term, seed=seed)
                current_df.loc[term_here, 'gross_compensation'] = draws.values
        else:
            current_df = sample_terminations(
                current_df,
                'hire_date',
                termination_rate,
                end_date,
                seed
            )
            # Sample termination salaries for rule-based terminations
            term_here = current_df['termination_date'].between(start_date, end_date)
            n_term = term_here.sum()
            if n_term:
                draws = sampler.sample_terminations(prev_term_salaries, size=n_term, seed=seed)
                current_df.loc[term_here, 'gross_compensation'] = draws.values

        # 6. Apply plan rules & calculate contributions
        current_df = determine_eligibility(current_df, scenario_config, end_date)
        ae_cfg     = scenario_config.get('plan_rules', {}).get('auto_enrollment', {})
        if ae_cfg.get('enabled', False):
            current_df = apply_auto_enrollment(current_df, scenario_config['plan_rules'], start_date, end_date)
        ai_cfg = scenario_config.get('plan_rules', {}).get('auto_increase', {})
        if ai_cfg.get('enabled', False):
            current_df = apply_auto_increase(current_df, scenario_config['plan_rules'], sim_year)
        current_df = calculate_contributions(current_df, scenario_config, sim_year, start_date, end_date)

        # Snapshot & refresh
        projected_data[year_num]     = current_df.copy()
        prev_hire_salaries = projected_data[year_num].loc[
            projected_data[year_num]['hire_date'].between(start_date, end_date),
            'gross_compensation'
        ].dropna()
        prev_term_salaries = projected_data[year_num].loc[
            projected_data[year_num]['termination_date'].between(start_date, end_date),
            'gross_compensation'
        ].dropna()

    logger.info("Projection complete for '%s'", scenario_config['scenario_name'])
    return projected_data