import json
import numpy as np

class ParameterSampler:
    """
    Loads Monte Carlo parameter packages from a JSON config and samples parameters reproducibly.
    """
    def __init__(self, path: str, seed: int = None):
        with open(path, 'r') as f:
            self.packages = json.load(f)
        # Initialize RNG for reproducibility
        self.rng = np.random.RandomState(seed)

    def sample(self, package_name: str) -> dict:
        if package_name not in self.packages:
            raise KeyError(f"Package '{package_name}' not found")
        specs = self.packages[package_name]
        result = {}
        for param, spec in specs.items():
            # Constant specification
            if not isinstance(spec, dict):
                result[param] = spec
                continue
            dist = spec.get('dist')
            if dist == 'constant':
                result[param] = spec['value']
            elif dist == 'uniform':
                low = spec['low']
                high = spec['high']
                result[param] = self.rng.uniform(low, high)
            elif dist == 'normal':
                mean = spec['mean']
                sd = spec['sd']
                result[param] = self.rng.normal(mean, sd)
            else:
                raise ValueError(f"Unknown distribution: {dist}")
        return result
