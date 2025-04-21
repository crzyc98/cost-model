import os
import glob
import yaml
from copy import deepcopy

def deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge override into base and return the result.
    """
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            base[key] = deep_merge(base[key], val)
        else:
            base[key] = deepcopy(val)
    return base


def load_yaml_dir(dir_path: str) -> dict:
    """
    Load all .yaml files in dir_path into a dict of configurations,
    resolving 'extends' inheritance.
    Returns:
        Dict[str, dict]: mapping scenario name to merged config dict.
    """
    raw_configs = {}
    # Load raw YAMLs
    for path in sorted(glob.glob(os.path.join(dir_path, '*.yaml'))):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, 'r') as f:
            raw_configs[name] = yaml.safe_load(f) or {}

    # Recursive resolver with cycle detection
    def resolve(name: str, seen=None) -> dict:
        if seen is None:
            seen = set()
        if name in seen:
            raise ValueError(f"Circular extends detected in '{name}'")
        seen.add(name)
        cfg = raw_configs.get(name)
        if cfg is None:
            raise ValueError(f"Scenario '{name}' not found in {dir_path}")
        parent = cfg.get('extends')
        base_cfg = {}
        if parent:
            base_cfg = resolve(parent, seen)
        # Remove 'extends' before merging
        overrides = {k: v for k, v in cfg.items() if k != 'extends'}
        return deep_merge(deepcopy(base_cfg), overrides)

    # Build final scenarios
    scenarios = {}
    for scenario in raw_configs:
        scenarios[scenario] = resolve(scenario)
    return scenarios
