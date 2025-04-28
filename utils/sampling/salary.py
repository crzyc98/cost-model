import numpy as np
import pandas as pd
from typing import Protocol, runtime_checkable
from numpy.random import Generator


@runtime_checkable
class SalarySampler(Protocol):
    """Protocol defining salary sampling behavior."""
    # dist: dict of distribution parameters (e.g. {'mean': float, 'std': float}) for sample_second_year
    def sample_second_year(
        self,
        df: pd.DataFrame,
        comp_col: str,
        dist: dict,
        rate: float,
        rng: Generator = None,
        seed: int = None,
        **kwargs
    ) -> pd.Series:
        """
        Apply a rate-based bump or sample distribution-based bump for second-year employees.

        dist: parameters for custom sampler (e.g. {'mean':..., 'std':...}).
        """
        ...

    def sample_terminations(
        self,
        prev: pd.Series,
        size: int,
        rng: Generator = None,
        seed: int = None,
        **kwargs
    ) -> pd.Series:
        ...


class DefaultSalarySampler:
    """Default implementation of SalarySampler.
    Ignores `dist` parameter and applies a fixed rate bump to second-year only.
    Custom samplers can override to use dist for distribution-based bumps."""

    def sample_second_year(
        self,
        df: pd.DataFrame,
        comp_col: str,
        dist: dict,
        rate: float,
        rng: Generator = None,
        seed: int = None,
        **kwargs
    ) -> pd.Series:
        """
        Apply a fixed rate bump to second-year employees only (dist ignored).

        Args:
            df: DataFrame containing a 'tenure' column.
            comp_col: Name of the compensation column to bump.
            dist: Distribution params for custom samplers (ignored here).
            rate: Bump rate (e.g., 0.1 for a 10% increase).
            rng: Random number generator.
            seed: Seed for random number generator (for test compatibility).

        Returns:
            Series of updated compensation values.
        """
        # support seed-based rng
        if rng is None:
            rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        comp = df[comp_col].astype(float)
        # Determine mask for second-year employees (tenure == 1)
        if 'tenure' in df.columns:
            mask = df['tenure'] == 1
        else:
            mask = pd.Series(False, index=df.index)
        result = comp.copy()
        # Bump only second-year (tenure == 1)
        result.loc[mask] = (comp[mask] * (1 + rate)).round(2)
        return result

    def sample_terminations(
        self,
        prev: pd.Series,
        size: int,
        rng: Generator = None,
        seed: int = None,
        **kwargs
    ) -> pd.Series:
        """
        Draw termination compensations by sampling with replacement from prior-year values.

        Args:
            prev: Series of prior-year compensation values.
            size: Number of draws to sample.
            rng: Random number generator.
            seed: Seed for random number generator.

        Returns:
            Series of sampled compensation values (length == size).
        """
        # initialize rng with seed if provided
        if rng is None:
            rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        # support numpy array or Series for prev data
        data = np.asarray(prev)
        # Edge cases: no draws or no source data
        if size <= 0 or data.size == 0:
            return pd.Series([], dtype=prev.dtype if hasattr(prev, 'dtype') else float)
        draws = rng.choice(data, size=size, replace=True)
        return pd.Series(draws, dtype=prev.dtype if hasattr(prev, 'dtype') else float)
