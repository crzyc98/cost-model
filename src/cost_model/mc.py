"""
Monte Carlo driver for retirement plan cost model.
"""
import numpy as np
from typing import Dict, List, Any, Callable
from .model import RetirementPlanModel

__all__ = ['MonteCarloRunner']

class MonteCarloRunner:
    """
    Drive Monte Carlo simulations for multiple scenarios.

    Attributes:
        scenarios: mapping of scenario names to config dicts
        sampler: callable that applies randomness to scenario configs
        runs: number of Monte Carlo iterations
        rng_seed: base seed for reproducible RNG
    """
    def __init__(
        self,
        scenarios: Dict[str, dict],
        sampler: Callable[[Dict[str, dict], np.random.RandomState], Dict[str, dict]],
        runs: int,
        rng_seed: int
    ):
        """
        Initialize the Monte Carlo runner.

        Args:
            scenarios: dict of scenario_name to config dict
            sampler: function(scenarios, rng) -> sampled scenarios
            runs: number of Monte Carlo runs
            rng_seed: base seed for RNG
        """
        self.scenarios = scenarios
        self.sampler = sampler
        self.runs = runs
        self.rng_seed = rng_seed

    def run(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Execute Monte Carlo simulations.

        Returns:
            Dict mapping each scenario name to a list of KPI dictionaries from each run.
        """
        # prepare results container
        results: Dict[str, List[Dict[str, Any]]] = {name: [] for name in self.scenarios}

        for i in range(self.runs):
            # new RNG per iteration for reproducibility
            rng = np.random.RandomState(self.rng_seed + i)
            # sample scenario configurations
            sampled = self.sampler(self.scenarios, rng)
            # run each sampled scenario
            for name, cfg in sampled.items():
                model = RetirementPlanModel(**cfg, rng=rng)
                model.run()
                results[name].append(model.kpi)

        return results
