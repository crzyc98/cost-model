import json
import os
import numpy as np

class McParams:
    """
    Monte Carlo parameter sampler for T-Shirt stress packages.
    """
    def __init__(self, package_name: str, packages: dict):
        if package_name not in packages:
            raise ValueError(f"MC package '{package_name}' not found in config")
        self.name = package_name
        self.variants = packages[package_name]  # e.g., {'XS': {...}, 'S': {...}, ...}
        self.variant_names = list(self.variants.keys())

    def sample(self, rng) -> dict:
        """
        Draw a random T-Shirt variant and sample its parameters.
        Returns a dict of {param_name: value, ..., 'mc_package': variant_name}
        """
        variant = rng.choice(self.variant_names)
        specs = self.variants[variant]
        draw = { 'mc_package': variant }
        for param, spec in specs.items():
            # constant value
            if not isinstance(spec, dict):
                draw[param] = spec
                continue
            dist = spec.get('dist')
            if dist == 'constant':
                draw[param] = spec['value']
            elif dist == 'normal':
                mean = spec['mean']
                sd = spec['sd']
                scale = spec.get('scale', 1.0)
                draw[param] = float(rng.normal(mean, sd) * scale)
            else:
                raise NotImplementedError(f"Distribution '{dist}' not supported")
        return draw


def load_mc_package(package_name: str, path: str = None) -> McParams:
    """
    Load MC package definitions from JSON and return a sampler.
    """
    if path is None:
        here = os.path.dirname(__file__)
        path = os.path.abspath(os.path.join(here, os.pardir, 'configs', 'mc_packages.json'))
    with open(path, 'r') as f:
        data = json.load(f)
    return McParams(package_name, data)
