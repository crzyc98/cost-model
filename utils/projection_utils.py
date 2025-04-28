# utils/projection_utils.py

"""
Core projection logic for the census data, orchestrating six clear steps:
1. Compensation bump
2. Termination sampling
3. Filter out early terminations
4. New-hire generation & compensation sampling
5. ML- or rule-based turnover
6. Apply plan-rule facades (eligibility, auto-enrollment, auto-increase, contributions)

Args:
    start_df (pd.DataFrame): initial employee census.
    scenario_config (dict): projection parameters and plan rules.
    random_seed (int, optional): master seed for reproducibility.

Returns:
    dict[int, pd.DataFrame]: yearly projected snapshots.
"""

import logging

import pandas as pd
import numpy as np
from typing import Union, Optional, Mapping, Any

from utils.date_utils import calculate_age, calculate_tenure
from utils.sandbox_utils import generate_new_hires
from utils.sampling.terminations import sample_terminations
from utils.sampling.new_hires import sample_new_hire_compensation
from utils.sampling.salary import SalarySampler, DefaultSalarySampler
from utils.ml.ml_utils import try_load_ml_model, predict_turnover
from utils.rules.eligibility import apply as apply_eligibility
from utils.rules.auto_enrollment import apply as apply_auto_enrollment
from utils.rules.auto_increase import apply as apply_auto_increase
from utils.rules.contributions import apply as apply_contributions

logger = logging.getLogger(__name__)


def _apply_turnover(
    df: pd.DataFrame,
    hire_col: str,
    probs_or_rate: Union[float, pd.Series],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    rng: np.random.Generator,
    sampler: SalarySampler,
    prev_term_salaries: np.ndarray
) -> pd.DataFrame:
    df2 = sample_terminations(df, hire_col, probs_or_rate, end_date, rng)
    term_here = df2['termination_date'].between(start_date, end_date)
    n_term = term_here.sum()
    if n_term:
        draws = sampler.sample_terminations(prev_term_salaries, size=n_term, rng=rng)
        df2.loc[term_here, 'gross_compensation'] = draws.values
        logger.debug("_apply_turnover: sampled %d terminations", n_term)
    return df2


def _apply_comp_bump(
    df: pd.DataFrame,
    comp_col: str,
    dist: Mapping[str, Any],
    rate: float,
    rng: np.random.Generator,
    sampler: SalarySampler
) -> pd.DataFrame:
    df2 = df.copy()
    if comp_col in df2:
        df2[comp_col] = sampler.sample_second_year(
            df2, comp_col=comp_col, dist=dist, rate=rate, rng=rng
        )
        if 'tenure' in df2.columns:
            mask_2nd = df2['tenure'] == 1
            mask_rest = ~mask_2nd
        else:
            # no tenure info: no second-year bumps or rest bumps
            mask_2nd = pd.Series(False, index=df2.index)
            mask_rest = pd.Series(False, index=df2.index)
        # count bumps
        n_second = mask_2nd.sum()
        n_rest   = mask_rest.sum()
        df2.loc[mask_rest, comp_col] *= (1 + rate)
        logger.debug("_apply_comp_bump: bumped %d second-year and %d rest", n_second, n_rest)
    else:
        logger.warning("Missing '%s' column, skipping comp bump", comp_col)
    return df2


def project_census(
    start_df: pd.DataFrame,
    scenario_config: dict,
    random_seed: int = None
) -> dict[int, pd.DataFrame]:
    projection_years     = scenario_config['projection_years']
    start_year           = scenario_config['start_year']
    comp_increase_rate   = scenario_config.get('comp_increase_rate', 0.0)
    hire_rate            = scenario_config.get('hire_rate', 0.0)
    termination_rate     = scenario_config.get('termination_rate', 0.0)
    # Initialize pluggable salary sampler
    sampler_cfg = scenario_config.get('salary_sampler', DefaultSalarySampler())
    if isinstance(sampler_cfg, type):
        sampler: SalarySampler = sampler_cfg()
    elif isinstance(sampler_cfg, SalarySampler):
        sampler = sampler_cfg
    else:
        sampler = DefaultSalarySampler()

    seed = random_seed if random_seed is not None else scenario_config.get('random_seed')
    # Randomness fully handled by numpy RNG below
    # Create a local RNG for reproducibility
    rng = np.random.default_rng(seed)

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
    # Compute and freeze baseline hire salary distribution
    baseline_hire_salaries = start_df.loc[
        start_df['hire_date'].dt.year == start_year - 1,
        'gross_compensation'
    ].dropna()
    if baseline_hire_salaries.empty:
        baseline_hire_salaries = start_df['gross_compensation'].dropna()

    current_df     = start_df.copy()
    prev_term_salaries = start_df['gross_compensation'].dropna()

    logger.info("Starting projection for '%s' from %s", scenario_config['scenario_name'], start_year)
    logger.info("Initial headcount: %d", len(current_df))

    for year_num in range(1, projection_years + 1):
        sim_year       = start_year + year_num - 1
        start_date     = pd.Timestamp(f"{sim_year}-01-01")
        end_date       = pd.Timestamp(f"{sim_year}-12-31")

        logger.info("Year %d (%s) ▶ headcount %d", year_num, sim_year, len(current_df))

        # 1. Compensation bump
        current_df = _apply_comp_bump(
            current_df,
            'gross_compensation',
            scenario_config.get('second_year_compensation_dist', {}),
            comp_increase_rate,
            rng,
            sampler
        )

        # 2. Apply turnover (termination dates + salary draws)
        current_df = _apply_turnover(
            current_df, 'hire_date', termination_rate,
            start_date, end_date, rng,
            sampler, prev_term_salaries
        )

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
                nh_df,
                'gross_compensation',
                baseline_hire_salaries.values,
                rng=rng
            )
            current_df = pd.concat([current_df, nh_df], ignore_index=True)
            logger.info("Generated %d new hires", needed)
            logger.debug("_step4 new-hire generation: added %d rows", len(nh_df))

        # 5. ML- or rule-based turnover
        logger.debug("_step5 before ML turnover: %d rows", len(current_df))
        # Recompute features
        if 'birth_date' in current_df and 'hire_date' in current_df:
            current_df['age'] = calculate_age(current_df['birth_date'], end_date)
            current_df['tenure'] = calculate_tenure(current_df['hire_date'], end_date)
        probs = (predict_turnover(current_df, projection_model, feature_cols, end_date, rng)
                 if projection_model else termination_rate)
        current_df = _apply_turnover(
            current_df, 'hire_date', probs,
            start_date, end_date, rng,
            sampler, prev_term_salaries
        )

        # 6. Apply plan rules & calculate contributions
        rules = scenario_config.get('plan_rules') or {}
        # Eligibility
        current_df = apply_eligibility(current_df, rules, end_date)
        # Auto-enrollment & increase
        ae_cfg = rules.get('auto_enrollment', {})
        if ae_cfg.get('enabled', False):
            current_df = apply_auto_enrollment(current_df, rules, start_date, end_date)
        ai_cfg = rules.get('auto_increase', {})
        if ai_cfg.get('enabled', False):
            current_df = apply_auto_increase(current_df, rules, sim_year)
        # Contributions
        current_df = apply_contributions(current_df, rules, sim_year, start_date, end_date)

        # Snapshot
        projected_data[year_num] = current_df.copy()
        # Update prev_term_salaries for next iteration
        prev_term_salaries = projected_data[year_num].loc[
            projected_data[year_num]['termination_date'].between(start_date, end_date),
            'gross_compensation'
        ].dropna().values

    logger.info("Projection complete for '%s'", scenario_config['scenario_name'])
    return projected_data