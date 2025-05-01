# loaders/scenario_loader.py
from pathlib import Path
from typing import Dict, Any, Union, Set
import yaml
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)
# Cache for directory loads: key is resolved dir path
_SCENARIO_CACHE: Dict[str, Dict[str, Dict[str, Any]]] = {}

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base, returning a new dict."""
    merged = deepcopy(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = deepcopy(v)
    return merged


def load_yaml_dir(dir_path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """Load all .yaml/.yml files in a directory, resolve 'extends', return name→config mapping."""
    base = Path(dir_path)
    key = str(base.resolve())
    if key in _SCENARIO_CACHE:
        return _SCENARIO_CACHE[key]
    # Read raw YAML files
    raw: Dict[str, Dict[str, Any]] = {}
    for pattern in ('*.yaml', '*.yml'):
        for fp in sorted(base.glob(pattern)):
            raw[fp.stem] = yaml.safe_load(fp.read_text()) or {}
    # Resolver
    def _resolve(name: str, seen: Set[str] = None) -> Dict[str, Any]:
        if seen is None:
            seen = set()
        if name in seen:
            raise ValueError(f"Circular extends detected in '{name}'")
        seen.add(name)
        cfg = raw.get(name)
        if cfg is None:
            raise KeyError(f"No scenario '{name}' in {base}")
        parent = cfg.get('extends')
        base_cfg = _resolve(parent, seen) if parent else {}
        overrides = {k: v for k, v in cfg.items() if k != 'extends'}
        return deep_merge(base_cfg, overrides)
    # Build and cache
    result = {name: _resolve(name) for name in raw}
    _SCENARIO_CACHE[key] = result
    return result


def load(path: Union[str, Path]) -> Union[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Load scenario(s) from a directory or single YAML file."""
    p = Path(path)
    if p.is_dir():
        return load_yaml_dir(p)
    # Single file mode
    cfg = yaml.safe_load(p.read_text()) or {}
    parent = cfg.pop('extends', None)
    if not parent:
        return cfg
    # Resolve parent file
    parent_fp = (p.parent / parent)
    if not parent_fp.exists():
        for ext in ('.yaml', '.yml'):
            if parent_fp.with_suffix(ext).exists():
                parent_fp = parent_fp.with_suffix(ext)
                break
        else:
            raise FileNotFoundError(f"Cannot find parent '{parent}' for {p}")
    base_cfg = load(parent_fp)
    merged = deep_merge(base_cfg, cfg)  # type: ignore
    return merged
