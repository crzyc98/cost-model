# utils/sampling/salary.py

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
    """Bump second-year only by a fixed rate (or normal dist if provided),
    and sample terminations by drawing from prior compensation."""

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
        """
        For tenure == 1, bump comp_col by either:
          - flat `rate`  (e.g. 0.10 → +10%), or
          - normal distribution if `dist={'mean':…, 'std':…}` is passed.
        """
        rng = rng or self.rng
        comp = df[comp_col].astype(float)
        mask = df.get("tenure", 0) == 1
        bumped = comp.copy()

        if dist and dist.get("type") == "normal":
            mean = dist.get("mean", 0.0)
            std = dist.get("std", 0.0)
            # draw one bump per second-year row
            bumps = rng.normal(loc=mean, scale=std, size=mask.sum())
            bumped.loc[mask] = (comp[mask] * (1 + bumps)).round(2)
        elif rate:
            bumped.loc[mask] = (comp[mask] * (1 + rate)).round(2)

        return pd.Series(bumped, index=df.index)

    def sample_terminations(
        self, prev: pd.Series, size: int, rng: Optional[Generator] = None
    ) -> pd.Series:
        """
        Draw `size` termination‐year compensations by sampling with replacement
        from `prev` (prior‐year comp) values.
        """
        rng = rng or self.rng
        data = prev.dropna().to_numpy()
        if size <= 0 or data.size == 0:
            # preserve dtype if possible
            return pd.Series([], dtype=prev.dtype, index=[])
        draws = rng.choice(data, size=size, replace=True)
        return pd.Series(draws, index=range(size), dtype=draws.dtype)
