import numpy as np
import pandas as pd
from typing import Optional, Protocol


class SalarySampler(Protocol):
    """Protocol defining salary sampling behavior."""
    def sample_second_year(
        self,
        df: pd.DataFrame,
        comp_col: str,
        dist: dict,
        rate: float,
        seed: Optional[int] = None
    ) -> pd.Series:
        ...

    def sample_terminations(
        self,
        prev: pd.Series,
        size: int,
        seed: Optional[int] = None
    ) -> pd.Series:
        ...


class DefaultSalarySampler:
    """Default implementation of SalarySampler using simple bump and uniform sampling."""

    def sample_second_year(
        self,
        df: pd.DataFrame,
        comp_col: str,
        dist: dict,
        rate: float,
        seed: Optional[int] = None
    ) -> pd.Series:
        """
        Apply a rate-based bump to second-year employees only.

        Args:
            df: DataFrame containing a 'tenure' column.
            comp_col: Name of the compensation column to bump.
            dist: Unused in default sampler (kept for signature compatibility).
            rate: Bump rate (e.g., 0.1 for a 10% increase).
            seed: Optional random seed (ignored in default sampler).

        Returns:
            Series of updated compensation values.
        """
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
        seed: Optional[int] = None
    ) -> pd.Series:
        """
        Draw termination compensations by sampling with replacement from prior-year values.

        Args:
            prev: Series of prior-year compensation values.
            size: Number of draws to sample.
            seed: Optional random seed for reproducibility.

        Returns:
            Series of sampled compensation values (length == size).
        """
        # Edge cases: no draws or no source data
        if size <= 0 or prev.empty:
            return pd.Series([], dtype=prev.dtype)
        if seed is not None:
            np.random.seed(seed)
        draws = np.random.choice(prev.values, size=size, replace=True)
        return pd.Series(draws, dtype=prev.dtype)
