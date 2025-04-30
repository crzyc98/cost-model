"""Monte Carlo package loader and sampler."""
import json
from pathlib import Path
import numpy as np
import logging
from typing import Optional, Union, Dict, Any, Tuple
from numpy.random import Generator

logger = logging.getLogger(__name__)
_PACKAGE_CACHE: Dict[Tuple[str, str], 'McParams'] = {}

class McParams:
    """
    Monte Carlo parameter sampler for T-Shirt stress packages.
    """
    def __init__(self, package_name: str, packages: Dict[str, Any]):
        if package_name not in packages:
            raise ValueError(f"MC package '{package_name}' not found in config")
        self.name = package_name
        self.variants = packages[package_name]
        self.variant_names = list(self.variants.keys())
        # Validate variant specs
        for variant, specs in self.variants.items():
            for param, spec in specs.items():
                if isinstance(spec, dict):
                    dist = spec.get('dist')
                    if dist == 'constant' and 'value' not in spec:
                        raise ValueError(f"MC package {package_name}: variant {variant} missing 'value' for constant param {param}")
                    elif dist == 'normal' and not all(k in spec for k in ('mean','sd')):
                        raise ValueError(f"MC package {package_name}: variant {variant} missing 'mean' or 'sd' for normal param {param}")
                    elif dist == 'uniform' and not all(k in spec for k in ('low','high')):
                        raise ValueError(f"MC package {package_name}: variant {variant} missing 'low' or 'high' for uniform param {param}")
                    elif dist not in ('constant','normal','uniform'):
                        raise ValueError(f"MC package {package_name}: variant {variant} unsupported dist '{dist}' for param {param}")

    def sample(self, rng: Generator) -> Dict[str, Any]:
        """
        Draw a random T-Shirt variant and sample its parameters.
        Returns a dict of {param_name: value, ..., 'mc_package': variant_name}
        """
        variant = rng.choice(self.variant_names)
        logger.debug("MC[%s] -> variant %s", self.name, variant)
        specs = self.variants[variant]
        draw: Dict[str, Any] = {'mc_package': variant}
        # Distribution functions
        _DIST_FNS = {
            'constant': lambda rng, spec: spec['value'],
            'normal':   lambda rng, spec: float(rng.normal(spec['mean'], spec['sd']) * spec.get('scale',1.0)),
            'uniform':  lambda rng, spec: float(rng.uniform(spec['low'], spec['high'])),
        }
        for param, spec in specs.items():
            if not isinstance(spec, dict):
                draw[param] = spec
            else:
                dist = spec.get('dist')
                fn = _DIST_FNS.get(dist)
                if fn is None:
                    raise NotImplementedError(f"Distribution '{dist}' not supported")
                draw[param] = fn(rng, spec)
        return draw

def load_mc_package(
    package_name: str,
    path: Optional[Union[str, Path]] = None
) -> McParams:
    """
    Load MC package definitions from JSON and return a sampler.
    """
    # Resolve path
    pkg_path = Path(path) if path else Path(__file__).parent.parent / 'configs' / 'mc_packages.json'
    pkg_path = pkg_path.resolve()
    key = (str(pkg_path), package_name)
    if key in _PACKAGE_CACHE:
        return _PACKAGE_CACHE[key]
    data = json.loads(pkg_path.read_text())
    sampler = McParams(package_name, data)
    _PACKAGE_CACHE[key] = sampler
    return sampler
