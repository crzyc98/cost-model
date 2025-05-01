from typing import Dict, Any
import os
import glob
import yaml
from copy import deepcopy

__all__ = ['load']

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge override into base and return the result.
    """
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            base[key] = deep_merge(base[key], val)
        else:
            base[key] = deepcopy(val)
    return base


def load(path: str) -> Any:
    """
    Load a scenario from a YAML file or directory of YAMLs.
    If path is a directory, returns Dict[str, Dict].
    If path is a file, returns a single config dict, resolving 'extends'.
    """
    if os.path.isdir(path):
        raw = {}
        for ext in ('*.yaml', '*.yml'):
            for fp in sorted(glob.glob(os.path.join(path, ext))):
                name = os.path.splitext(os.path.basename(fp))[0]
                with open(fp) as f:
                    raw[name] = yaml.safe_load(f) or {}
        def resolve(name: str, seen=None):
            if seen is None:
                seen = set()
            if name in seen:
                raise ValueError(f"Circular extends detected in '{name}'")
            seen.add(name)
            cfg = raw.get(name)
            if cfg is None:
                raise ValueError(f"Scenario '{name}' not found in {path}")
            parent = cfg.get('extends')
            base = {}
            if parent:
                parent_name = os.path.splitext(parent)[0]
                base = resolve(parent_name, seen)
            overrides = {k: v for k, v in cfg.items() if k != 'extends'}
            return deep_merge(base, overrides)
        return {name: resolve(name) for name in raw}
    else:
        dirpath = os.path.dirname(path)
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        parent = cfg.get('extends')
        if not parent:
            return cfg
        parent_fp = os.path.join(dirpath, parent)
        if not os.path.exists(parent_fp):
            raise FileNotFoundError(f"Parent config '{parent}' not found for {path}")
        parent_cfg = load(parent_fp)
        overrides = {k: v for k, v in cfg.items() if k != 'extends'}
        return deep_merge(parent_cfg, overrides)
