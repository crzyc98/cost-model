# cost_model/projections/config.py
"""
Configuration loading and conversion utilities.
QuickStart: see docs/cost_model/projections/config.md
"""

# YAML loading + namespace conversion
import yaml
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Union


def dict_to_simplenamespace(d: Any) -> Any:
    """Recursively convert a dictionary to a SimpleNamespace."""
    if isinstance(d, dict):
        kwargs = {str(k): dict_to_simplenamespace(v) for k, v in d.items()}
        return SimpleNamespace(**kwargs)
    elif isinstance(d, list):
        return [dict_to_simplenamespace(item) for item in d]
    return d


def load_config_to_namespace(config_path: Union[Path, str]) -> SimpleNamespace:
    """Load a YAML configuration file and convert it to a SimpleNamespace."""
    if isinstance(config_path, str):
        config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return dict_to_simplenamespace(config_dict)
