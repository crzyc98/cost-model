# cost_model/dynamics/sampling/salary.py
"""
Functions related to sampling salary changes during workforce simulations.
QuickStart: see docs/cost_model/dynamics/sampling/salary.md
"""

from __future__ import annotations
import pandas as pd
from numpy.random import Generator, default_rng
from typing import Protocol, runtime_checkable, Optional, Dict


@runtime_checkable
class SalarySampler(Protocol):
    def sample_second_year(
        self,
        df: pd.DataFrame,
        comp_col: str,
        *,
        rate: float = 0.0,
        dist: Optional[Dict[str, float]] = None,
        rng: Optional[Generator] = None,
    ) -> pd.Series: ...

    def sample_terminations(
        self, prev: pd.Series, size: int, rng: Optional[Generator] = None
    ) -> pd.Series: ...


class DefaultSalarySampler:
    """
    Bump second-year only by a fixed rate (or normal dist if provided),
    and sample terminations and new hires with config-driven, age-adjusted, and clamped logic.
    """

    def __init__(self, rng: Optional[Generator] = None):
        self.rng = rng or default_rng()

    def sample_second_year(
        self,
        df: pd.DataFrame,
        comp_col: str,
        *,
        rate: float = 0.0,
        dist: Optional[Dict[str, float]] = None,
        rng: Optional[Generator] = None,
    ) -> pd.Series:
        rng = rng or self.rng
        comp = df[comp_col].astype(float)
        mask = df.get("tenure", 0) == 1
        bumped = comp.copy()

        if dist and dist.get("type") == "normal":
            mean = dist.get("mean", 0.0)
            std = dist.get("std", 0.0)
            bumps = rng.normal(loc=mean, scale=std, size=mask.sum())
            bumped.loc[mask] = (comp[mask] * (1 + bumps)).round(2)
        elif rate:
            bumped.loc[mask] = (comp[mask] * (1 + rate)).round(2)

        return pd.Series(bumped, index=df.index)

    def sample_terminations(
        self, prev: pd.Series, size: int, rng: Optional[Generator] = None
    ) -> pd.Series:
        rng = rng or self.rng
        data = prev.dropna().to_numpy()
        logger.info(f"[SALARY.SAMPLE] Compensation pool stats (N={len(data)}):")
        logger.info(f"[SALARY.SAMPLE]   Min: ${np.min(data):,.0f}, Max: ${np.max(data):,.0f}")
        logger.info(f"[SALARY.SAMPLE]   Mean: ${np.mean(data):,.0f}, Median: ${np.median(data):,.0f}")
        logger.info(f"[SALARY.SAMPLE]   25th: ${np.percentile(data, 25):,.0f}, 75th: ${np.percentile(data, 75):,.0f}")
        if size <= 0 or data.size == 0:
            logger.warning(f"[SALARY.SAMPLE] No compensations to sample from (size={size}, pool_size={len(data)})")
            return pd.Series([], dtype=prev.dtype, index=[])
        draws = rng.choice(data, size=size, replace=True)
        min_bound = max(40000, np.percentile(data, 10))
        max_bound = min(150000, np.percentile(data, 90))
        clamped_draws = np.clip(draws, min_bound, max_bound)
        logger.info(f"[SALARY.SAMPLE] Sampled {size} compensations:")
        logger.info(f"[SALARY.SAMPLE]   Raw Min: ${np.min(draws):,.0f}, Max: ${np.max(draws):,.0f}")
        logger.info(f"[SALARY.SAMPLE]   Clamped Min: ${np.min(clamped_draws):,.0f}, Max: ${np.max(clamped_draws):,.0f}")
        logger.info(f"[SALARY.SAMPLE]   Sampled Mean: ${np.mean(clamped_draws):,.0f}, Median: ${np.median(clamped_draws):,.0f}")
        logger.info(f"[SALARY.SAMPLE]   Clamping bounds: Min=${min_bound:,.0f}, Max=${max_bound:,.0f}")
        return pd.Series(clamped_draws, index=range(size), dtype=draws.dtype)

    def sample_new_hires(
        self, size: int, params: Dict[str, float], ages: Optional[np.ndarray] = None, 
        rng: Optional[Generator] = None
    ) -> pd.Series:
        """
        Generate realistic new hire salaries based on configuration parameters and ages.
        Uses the same parameters as in the config:
        - comp_base_salary: Base salary amount
        - comp_min_salary: Minimum allowable salary
        - comp_max_salary: Maximum allowable salary
        - comp_age_factor: Salary increase factor per year of age
        - comp_stochastic_std_dev: Standard deviation for stochastic component
        """
        rng = rng or self.rng
        if size <= 0:
            logger.warning(f"[SALARY.SAMPLE] No new hire salaries to generate (size={size})")
            return pd.Series([], dtype=float, index=[])
        base_salary = params.get('comp_base_salary', 100000)
        min_salary = params.get('comp_min_salary', 40000)
        max_salary = params.get('comp_max_salary', 450000)
        age_factor = params.get('comp_age_factor', 0.05)
        std_dev_factor = params.get('comp_stochastic_std_dev', 0.08)
        std_dev = base_salary * std_dev_factor
        base_salaries = rng.normal(loc=base_salary, scale=std_dev, size=size)
        if ages is not None and len(ages) == size:
            reference_age = 30
            age_adjustments = (ages - reference_age) * age_factor * base_salary
            salaries = base_salaries + age_adjustments
        else:
            salaries = base_salaries
        clamped_salaries = np.clip(salaries, min_salary, max_salary)
        clamped_salaries = np.round(clamped_salaries, 2)
        logger.info(f"[SALARY.SAMPLE] Generated {size} new hire salaries:")
        logger.info(f"[SALARY.SAMPLE]   Raw Min: ${np.min(salaries):,.2f}, Max: ${np.max(salaries):,.2f}")
        logger.info(f"[SALARY.SAMPLE]   Clamped Min: ${np.min(clamped_salaries):,.2f}, Max: ${np.max(clamped_salaries):,.2f}")
        logger.info(f"[SALARY.SAMPLE]   Sampled Mean: ${np.mean(clamped_salaries):,.2f}, Median: ${np.median(clamped_salaries):,.2f}")
        logger.info(f"[SALARY.SAMPLE]   Parameters: Base=${base_salary:,.2f}, StdDev=${std_dev:,.2f}")
        logger.info(f"[SALARY.SAMPLE]   Age factor: ${age_factor*base_salary:,.2f} per year from reference")
        logger.info(f"[SALARY.SAMPLE]   Clamping bounds: Min=${min_salary:,.2f}, Max=${max_salary:,.2f}")
        return pd.Series(clamped_salaries, index=range(size), dtype=float)
