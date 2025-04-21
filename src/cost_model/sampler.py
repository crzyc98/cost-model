import json
import random
from typing import Optional, Dict

class ParameterSampler:
    """
    Load MC package definitions from a JSON file and sample parameters by package.
    """
    def __init__(self, mc_packages_path: str, seed: Optional[int] = None):
        with open(mc_packages_path) as f:
            self.packages = json.load(f)
        # deterministic RNG
        self.random = random.Random(seed)

    def sample(self, package_key: str) -> Dict[str, float]:
        """
        Return a dict of sampled parameters for the given package key.
        """
        if package_key not in self.packages:
            raise KeyError(f"Package '{package_key}' not found in MC packages")
        spec_pkg = self.packages[package_key]
        samples: Dict[str, float] = {}
        for name, spec in spec_pkg.items():
            dist = spec.get("dist")
            if dist == "constant":
                samples[name] = spec["value"]
            elif dist == "normal":
                mu = spec["mean"]
                sd = spec["sd"]
                samples[name] = self.random.gauss(mu, sd)
            elif dist == "uniform":
                low = spec["low"]
                high = spec["high"]
                samples[name] = self.random.uniform(low, high)
            else:
                raise ValueError(f"Unknown distribution '{dist}' for parameter '{name}'")
        return samples
