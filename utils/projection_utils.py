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
import math

import pandas as pd
import numpy as np
from typing import Union, Optional, Mapping, Any, Dict

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
        # assign sampled termination draws; draws may be numpy array or pandas Series
        df2.loc[term_here, 'gross_compensation'] = draws
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
    if comp_col not in df2:
        logger.warning("Missing '%s' column, skipping comp bump", comp_col)
        return df2

    # define groups by tenure (new hires omitted)
    mask_second = df2['tenure'] == 1
    mask_exp = df2['tenure'] >= 2

    # 1) second-year employees get sampler bump
    if mask_second.any():
        df2.loc[mask_second, comp_col] = sampler.sample_second_year(
            df2.loc[mask_second], comp_col=comp_col, dist=dist, rate=rate, rng=rng
        )
    # 2) experienced employees get flat increase
    if mask_exp.any():
        df2.loc[mask_exp, comp_col] *= (1 + rate)

    logger.debug(
        "Comp bump: %d second-year sampled, %d experienced bumped by %.1f%%",
        mask_second.sum(), mask_exp.sum(), rate * 100
    )
    return df2


def apply_onboarding_bump(
    df: pd.DataFrame,
    comp_col: str,
    ob_cfg: Mapping[str, Any],
    baseline_hire_salaries: np.ndarray,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Apply onboarding bump for new hires.
    Methods: 'flat_rate' or 'sample_plus_rate'.
    flat_rate: apply a flat percentage bump on existing comp_col.
    sample_plus_rate: sample from baseline_hire_salaries then apply rate.
    """
    df2 = df.copy()
    if not ob_cfg.get('enabled', False):
        return df2
    method = ob_cfg.get('method', '')
    rate = ob_cfg.get('rate', 0.0)
    if method == 'flat_rate':
        df2[comp_col] = df2[comp_col] * (1 + rate)
    elif method == 'sample_plus_rate':
        draws = rng.choice(baseline_hire_salaries, size=len(df2))
        df2[comp_col] = draws * (1 + rate)
    return df2


# Phase I: HR-only projection (census-only snapshots)
def project_hr(
    start_df: pd.DataFrame,
    scenario_config: Mapping[str, Any],
    random_seed: Optional[int] = None
) -> Dict[int, pd.DataFrame]:
    """
    Phase I: compensation bumps, terminations (rule + ML), headcount resets & new hires.
    Returns a dict mapping year→census-only DataFrame (no plan rules applied).
    """
    # Initialize RNGs same as project_census
    seed = random_seed if random_seed is not None else scenario_config.get('random_seed')
    master_ss = np.random.SeedSequence(seed)
    ss_bump, ss_term, ss_nh, ss_ml, ss_rules = master_ss.spawn(5)
    rng_bump = np.random.default_rng(ss_bump)
    rng_term = np.random.default_rng(ss_term)
    rng_nh = np.random.default_rng(ss_nh)
    rng_ml = np.random.default_rng(ss_ml)

    # Prep parameters
    projection_years = scenario_config['projection_years']
    start_year = scenario_config['start_year']
    # HR parameter fallbacks (top-level or under plan_rules)
    pr_cfg = scenario_config.get('plan_rules', {})
    comp_increase_rate = scenario_config.get('annual_compensation_increase_rate') or pr_cfg.get('annual_compensation_increase_rate', 0.0)
    termination_rate = scenario_config.get('annual_termination_rate') or pr_cfg.get('annual_termination_rate', 0.0)
    # headcount growth
    growth_rate = scenario_config.get('annual_growth_rate') or pr_cfg.get('annual_growth_rate', 0.0)
    maintain_hc = scenario_config.get('maintain_headcount', None)
    maintain_headcount = maintain_hc if maintain_hc is not None else pr_cfg.get('maintain_headcount', True)
    # new-hire termination rate
    nh_term_rate = scenario_config.get('new_hire_termination_rate') or pr_cfg.get('new_hire_termination_rate', 0.0)
    # ML model
    model_path = scenario_config.get('ml_model_path', '')
    features_path = scenario_config.get('model_features_path', '')
    if model_path and features_path:
        ml_pair = try_load_ml_model(model_path, features_path)
        projection_model, feature_cols = (None, []) if ml_pair is None else ml_pair
    else:
        projection_model, feature_cols = None, []
    use_ml = scenario_config.get('use_ml_turnover', False) and projection_model is not None

    # Salaries for new-hire sampling
    baseline_hire_salaries = start_df.loc[
        start_df['hire_date'].dt.year == start_year - 1,
        'gross_compensation'
    ].dropna()
    if baseline_hire_salaries.empty:
        baseline_hire_salaries = start_df['gross_compensation'].dropna()

    current_df = start_df.copy()
    prev_term_salaries = start_df['gross_compensation'].dropna().values
    base_count = len(start_df)
    hr_snapshots: Dict[int, pd.DataFrame] = {}

    for year_num in range(1, projection_years + 1):
        sim_year = start_year + year_num - 1
        start_date = pd.Timestamp(f"{sim_year}-01-01")
        end_date = pd.Timestamp(f"{sim_year}-12-31")

        # Recalculate tenure
        current_df['tenure'] = calculate_tenure(current_df['hire_date'], start_date)
        # 1. Comp bump
        current_df = _apply_comp_bump(
            current_df, 'gross_compensation',
            scenario_config.get('second_year_compensation_dist', {}),
            comp_increase_rate, rng_bump, DefaultSalarySampler()
        )
        # 2. Early turnover
        current_df = _apply_turnover(
            current_df, 'hire_date', termination_rate,
            start_date, end_date, rng_term,
            DefaultSalarySampler(), prev_term_salaries
        )
        # 3. Drop pre-period terminations
        current_df = current_df[
            current_df['termination_date'].isna() |
            (current_df['termination_date'] >= start_date)
        ].copy()
        # 4. New hires & comp sampling
        # determine hires needed
        survivors = len(current_df)
        if maintain_headcount:
            needed = max(0, base_count - survivors)
        else:
            target = int(base_count * (1 + growth_rate) ** year_num)
            net_needed = max(0, target - survivors)
            if nh_term_rate < 1:
                needed = math.ceil(net_needed / (1 - nh_term_rate))
            else:
                needed = net_needed
        if needed > 0:
            # derive new-hire age parameters: top-level, plan_rules, or generic age_mean
            age_mean = scenario_config.get('new_hire_average_age') or pr_cfg.get('new_hire_average_age') or scenario_config.get('age_mean') or 30
            # safe std dev fallback: new_hire_age_std_dev -> age_std_dev -> default 5.0
            age_std = scenario_config.get('new_hire_age_std_dev') or scenario_config.get('age_std_dev') or pr_cfg.get('age_std_dev', 5.0) or 5.0
            # working age bounds fall back to defaults
            min_age = scenario_config.get('min_working_age', 18)
            max_age = scenario_config.get('max_working_age', 65)
            nh_df = generate_new_hires(
                num_hires=needed,
                hire_year=sim_year,
                role_distribution=scenario_config.get('role_distribution'),
                role_compensation_params=scenario_config.get('role_compensation_params'),
                age_mean=age_mean,
                age_std_dev=age_std,
                min_working_age=min_age,
                max_working_age=max_age,
                scenario_config=scenario_config
            )
            nh_df = sample_new_hire_compensation(
                nh_df, 'gross_compensation',
                baseline_hire_salaries.values, rng_nh
            )
            ob_cfg = scenario_config.get('plan_rules', {}).get('onboarding_bump', {})
            nh_df = apply_onboarding_bump(
                nh_df, 'gross_compensation', ob_cfg,
                baseline_hire_salaries.values, rng_nh
            )
            current_df = pd.concat([current_df, nh_df], ignore_index=True)
        # 5. ML-based or rule-based turnover
        if use_ml:
            probs = predict_turnover(current_df, projection_model, feature_cols, end_date, rng_ml)
        else:
            probs = termination_rate
        current_df = _apply_turnover(
            current_df, 'hire_date', probs,
            start_date, end_date, rng_term,
            DefaultSalarySampler(), prev_term_salaries
        )
        # update prev_term_salaries for sampled terminations
        prev_term_salaries = current_df.loc[
            current_df['termination_date'].between(start_date, end_date),
            'gross_compensation'
        ].dropna().values
        # Snapshot
        hr_snapshots[year_num] = current_df.copy()
    return hr_snapshots


# Phase II: Apply plan rules (eligibility, auto_enroll, auto_increase, contributions)
def apply_plan_rules(
    df: pd.DataFrame,
    scenario_config: Mapping[str, Any],
    year_num: int
) -> pd.DataFrame:
    sim_year = scenario_config['start_year'] + year_num - 1
    start_date = pd.Timestamp(f"{sim_year}-01-01")
    end_date = pd.Timestamp(f"{sim_year}-12-31")

    # ensure deferral_rate exists
    if 'deferral_rate' not in df and 'pre_tax_deferral_percentage' in df:
        df['deferral_rate'] = df['pre_tax_deferral_percentage']

    # normalize participation and enrollment flags using pandas nullable BooleanDtype
    for col in ['is_participating', 'ae_opted_out', 'ai_opted_out']:
        # get existing col or create empty BooleanDtype Series
        series = df[col] if col in df.columns else pd.Series(index=df.index, dtype="boolean")
        # cast to BooleanDtype and fill missing as False
        df[col] = series.astype("boolean").fillna(False)

    df = apply_eligibility(df, scenario_config['plan_rules'], end_date)
    ae_cfg = scenario_config['plan_rules'].get('auto_enrollment', {})
    if ae_cfg.get('enabled', False):
        df = apply_auto_enrollment(df, scenario_config['plan_rules'], start_date, end_date)
    ai_cfg = scenario_config['plan_rules'].get('auto_increase', {})
    if ai_cfg.get('enabled', False):
        df = apply_auto_increase(df, scenario_config['plan_rules'], sim_year)
    df = apply_contributions(df, scenario_config['plan_rules'], sim_year, start_date, end_date)
    return df


# Legacy: Full census projection (all steps in one pass, unchanged)
def project_census(
    start_df: pd.DataFrame,
    scenario_config: Mapping[str, Any],
    random_seed: Optional[int] = None,
) -> Dict[int, pd.DataFrame]:
    projection_years     = scenario_config['projection_years']
    start_year           = scenario_config['start_year']
    # HR parameter fallbacks (top-level or under plan_rules)
    pr_cfg = scenario_config.get('plan_rules', {})
    comp_increase_rate = scenario_config.get('annual_compensation_increase_rate') or pr_cfg.get('annual_compensation_increase_rate', 0.0)
    termination_rate = scenario_config.get('annual_termination_rate') or pr_cfg.get('annual_termination_rate', 0.0)
    # headcount growth
    growth_rate = scenario_config.get('annual_growth_rate') or pr_cfg.get('annual_growth_rate', 0.0)
    maintain_hc = scenario_config.get('maintain_headcount', None)
    maintain_headcount = maintain_hc if maintain_hc is not None else pr_cfg.get('maintain_headcount', True)
    # new-hire termination rate
    nh_term_rate = scenario_config.get('new_hire_termination_rate') or pr_cfg.get('new_hire_termination_rate', 0.0)
    # Initialize pluggable salary sampler
    sampler_cfg = scenario_config.get('salary_sampler', DefaultSalarySampler())
    if isinstance(sampler_cfg, type):
        sampler: SalarySampler = sampler_cfg()
    elif isinstance(sampler_cfg, SalarySampler):
        sampler = sampler_cfg
    else:
        sampler = DefaultSalarySampler()

    seed = random_seed if random_seed is not None else scenario_config.get('random_seed')
    # Master SeedSequence for dedicated streams
    master_ss = np.random.SeedSequence(seed)
    bump_ss, term_ss, nh_ss, ml_ss, contrib_ss = master_ss.spawn(5)
    rng_bump = np.random.default_rng(bump_ss)
    rng_term = np.random.default_rng(term_ss)
    rng_nh = np.random.default_rng(nh_ss)
    rng_ml = np.random.default_rng(ml_ss)
    rng_contrib = np.random.default_rng(contrib_ss)

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
    base_count = len(start_df)
    growth_rate = growth_rate

    logger.info("Starting projection for '%s' from %s", scenario_config['scenario_name'], start_year)
    logger.info("Initial headcount: %d", len(current_df))

    for year_num in range(1, projection_years + 1):
        sim_year       = start_year + year_num - 1
        start_date     = pd.Timestamp(f"{sim_year}-01-01")
        end_date       = pd.Timestamp(f"{sim_year}-12-31")

        # Year 1 Start: log before any changes
        if year_num == 1:
            init_hc = len(current_df)
            init_comp = current_df['gross_compensation'].sum()
            logger.info(f"[Year 1 Start] headcount={init_hc}, total_gross_comp={init_comp:.2f}")

        logger.info("Year %d (%s) ▶ headcount %d", year_num, sim_year, len(current_df))
        # Recalculate tenure each year before compensation bump
        current_df['tenure'] = calculate_tenure(current_df['hire_date'], start_date)

        # 1. Compensation bump with dedicated RNG
        current_df = _apply_comp_bump(
            current_df,
            'gross_compensation',
            scenario_config.get('second_year_compensation_dist', {}),
            comp_increase_rate,
            rng_bump,
            sampler
        )

        # Yearly After Bump diagnostics
        post_bump = current_df['gross_compensation'].sum()
        mask_second = current_df['tenure'] == 1
        mask_exp = current_df['tenure'] >= 2
        logger.debug(
            f"[Year {year_num} After Bump] "
            f"second_year={mask_second.sum()}, "
            f"experienced={mask_exp.sum()}, "
            f"total_comp={post_bump:.2f}"
        )

        # 2. Apply turnover with dedicated RNG
        current_df = _apply_turnover(
            current_df, 'hire_date', termination_rate,
            start_date, end_date, rng_term,
            sampler, prev_term_salaries
        )

        # 3. Drop those who terminated before this period
        current_df = current_df[
            current_df['termination_date'].isna() |
            (current_df['termination_date'] >= start_date)
        ].copy()

        # 4. Generate new hires & sample their compensation
        # Compute year-end target headcount and survivors
        target_count = int(base_count * (1 + growth_rate) ** year_num)
        active_incumbents = current_df[
            current_df['termination_date'].isna() |
            (current_df['termination_date'] > end_date)
        ]
        survivors_count = len(active_incumbents)
        # Determine hires needed
        if scenario_config.get('maintain_headcount', True):
            net_needed = max(0, base_count - survivors_count)
            needed = net_needed
        else:
            target = int(base_count * (1 + growth_rate) ** year_num)
            net_needed = max(0, target - survivors_count)
            if nh_term_rate < 1:
                needed = math.ceil(net_needed / (1 - nh_term_rate))
            else:
                needed = net_needed
        logger.debug(
            "Year %d: target=%d, survivors=%d, net_needed=%d, hires=%d",
            year_num, target_count, survivors_count, net_needed, needed
        )

        if needed > 0:
            # derive new-hire age parameters with sensible default 30
            age_mean = (
                scenario_config.get('new_hire_average_age') or
                pr_cfg.get('new_hire_average_age') or
                scenario_config.get('age_mean') or
                30
            )
            age_std = scenario_config.get('new_hire_age_std_dev') or scenario_config.get('age_std_dev') or pr_cfg.get('age_std_dev', 5.0) or 5.0
            # working age bounds fall back to defaults
            min_age = scenario_config.get('min_working_age', 18)
            max_age = scenario_config.get('max_working_age', 65)
            nh_df = generate_new_hires(
                num_hires=needed,
                hire_year=sim_year,
                role_distribution=scenario_config.get('role_distribution'),
                role_compensation_params=scenario_config.get('role_compensation_params'),
                age_mean=age_mean,
                age_std_dev=age_std,
                min_working_age=min_age,
                max_working_age=max_age,
                scenario_config=scenario_config
            )
            # 3. New-hire compensation sampling with dedicated RNG
            nh_df = sample_new_hire_compensation(
                nh_df,
                'gross_compensation',
                baseline_hire_salaries.values,
                rng=rng_nh
            )
            # Year 1 New-Hire Salary Distribution (unchanged)
            if year_num == 1:
                logger.debug(
                    f"[Year {year_num} New-hire Salaries] "
                    f"count={len(nh_df)}, "
                    f"total=${nh_df['gross_compensation'].sum():,.0f}, "
                    f"mean=${nh_df['gross_compensation'].mean():,.0f}, "
                    f"min=${nh_df['gross_compensation'].min():,.0f}, "
                    f"max=${nh_df['gross_compensation'].max():,.0f}"
                )
            # 4. Onboarding bump with dedicated RNG
            ob_cfg = scenario_config.get('plan_rules', {}).get('onboarding_bump', {})
            nh_df = apply_onboarding_bump(
                nh_df,
                'gross_compensation',
                ob_cfg,
                baseline_hire_salaries.values,
                rng_nh
            )
            after_onb = nh_df['gross_compensation'].sum()
            logger.debug(
                f"[Year {year_num} Onboarding] "
                f"enabled={ob_cfg.get('enabled')}, "
                f"rate={ob_cfg.get('rate')}, "
                f"post={after_onb:.2f}"
            )
            # append all hires and recalc tenure
            current_df = pd.concat([current_df, nh_df], ignore_index=True)
            current_df['tenure'] = calculate_tenure(current_df['hire_date'], start_date)
            logger.info("Generated %d new hires", needed)
            logger.debug("_step4 new-hire generation: added %d rows", len(nh_df))

            # Year 1 New Hires
            if year_num == 1:
                hires_comp = nh_df['gross_compensation'].sum()
                logger.info(f"[Year 1 New Hires] count={len(nh_df)}, total_gross_comp={hires_comp:.2f}")

        # 5. ML-based turnover with dedicated RNG
        probs = (predict_turnover(current_df, projection_model, feature_cols, end_date, rng_ml)
                 if scenario_config.get('use_ml_turnover', False) and projection_model is not None else 0.0)
        current_df = _apply_turnover(
            current_df, 'hire_date', probs,
            start_date, end_date, rng_term,
            sampler, prev_term_salaries
        )

        # Year 1 After Turnover
        if year_num == 1:
            final_hc = len(current_df)
            final_comp = current_df['gross_compensation'].sum()
            logger.info(f"[Year 1 After Turnover] headcount={final_hc}, total_gross_comp={final_comp:.2f}")

        # 6. Apply plan rules & calculate contributions
        rules = scenario_config.get('plan_rules') or {}
        # Ensure deferral_rate exists (alias pre_tax_deferral_percentage)
        if 'deferral_rate' not in current_df.columns and 'pre_tax_deferral_percentage' in current_df.columns:
            current_df['deferral_rate'] = current_df['pre_tax_deferral_percentage']
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