import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Set

import yaml
from cerberus import Validator

# Keys whose values should remain as dicts rather than namespaces
# These can be nested paths like 'global_parameters.new_hire_compensation_params'
DICT_KEYS: Set[str] = {
    "role_compensation_params",
    "new_hire_compensation_params",
    "global_parameters.role_compensation_params",
    "global_parameters.new_hire_compensation_params",
    # add any other mapping-style sections here
}


def should_remain_dict(key_path: str) -> bool:
    """
    Check if a key path should remain a dict based on DICT_KEYS.
    """
    return key_path in DICT_KEYS or any(key_path.endswith(f".{k}") for k in DICT_KEYS)


def dict_to_namespace(obj: Any, path: str = "") -> Any:
    """
    Recursively convert dicts to SimpleNamespace, except for paths matching DICT_KEYS.
    Lists will be converted elementwise.
    """
    if isinstance(obj, dict):
        if should_remain_dict(path):
            return obj
        converted: Dict[str, Any] = {}
        for k, v in obj.items():
            str_k = str(k)  # Ensure all keys are strings
            child_path = f"{path}.{str_k}" if path else str_k
            converted[str_k] = dict_to_namespace(v, child_path)
        return SimpleNamespace(**converted)
    elif isinstance(obj, list):
        return [dict_to_namespace(v, path) for v in obj]
    else:
        return obj


# Configure logger for this module
logger = logging.getLogger(__name__)


class ConfigLoadError(Exception):
    """Custom exception for errors during config loading."""

    pass


def load_yaml_config(config_path: Path) -> Optional[Dict[str, Any]]:
    """
    Loads configuration data from a YAML file.

    Args:
        config_path: Path object pointing to the YAML configuration file.

    Returns:
        A dictionary containing the loaded configuration, or None if loading fails.

    Raises:
        ConfigLoadError: If the file cannot be found or parsed.
    """
    if not isinstance(config_path, Path):
        config_path = Path(config_path)

    logger.info(f"Attempting to load configuration from: {config_path}")

    try:
        if not config_path.is_file():
            logger.error(f"Configuration file not found at path: {config_path}")
            raise ConfigLoadError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        if not isinstance(config_data, dict):
            logger.error(f"Configuration file {config_path} did not parse into a dictionary.")
            raise ConfigLoadError(
                f"Invalid configuration format in {config_path}: Expected a dictionary."
            )

        logger.info(f"Successfully loaded configuration from {config_path}")
        return config_data

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found (FileNotFoundError): {config_path}")
        raise ConfigLoadError(f"Configuration file not found: {config_path}") from e
    except yaml.YAMLError as e:
        logger.exception(f"Error parsing YAML configuration file {config_path}: {e}")
        raise ConfigLoadError(f"Error parsing YAML file {config_path}") from e
    except Exception as e:
        logger.exception(f"An unexpected error occurred while loading config {config_path}: {e}")
        raise ConfigLoadError(f"Unexpected error loading config {config_path}") from e


def load_config_to_namespace(config_path: Path) -> SimpleNamespace:
    """
    Loads YAML, validates its schema, merges role defaults, and converts to a SimpleNamespace.
    - Mapping-style sections in DICT_KEYS remain dicts (including nested paths).
    - role_compensation_params is merged with new_hire_compensation_params defaults for each role.
    - Raises ConfigLoadError on validation errors.
    """
    config_data = load_yaml_config(config_path)
    if config_data is None:
        raise ConfigLoadError(f"No config at {config_path}")

    # 1. Schema validation
    schema = {
        "global_parameters": {"type": "dict", "required": False},
        "plan_rules": {"type": "dict", "required": False},
        "scenarios": {"type": "dict", "required": False},
        "new_hire_compensation_params": {"type": "dict", "required": False},
        "role_compensation_params": {"type": "dict", "required": False},
        "job_levels": {
            "type": "list",
            "required": False,
            "schema": {
                "type": "dict",
                "schema": {
                    "level_id": {"type": "integer", "required": True},
                    "name": {"type": "string", "required": True},
                    "description": {"type": "string", "required": False},
                    "min_compensation": {"type": "number", "required": True},
                    "max_compensation": {"type": "number", "required": True},
                    "comp_base_salary": {"type": "number", "required": False},
                    "comp_age_factor": {"type": "number", "required": False},
                    "comp_stochastic_std_dev": {"type": "number", "required": False},
                    "avg_annual_merit_increase": {"type": "number", "required": False},
                    "promotion_probability": {"type": "number", "required": False},
                    "target_bonus_percent": {"type": "number", "required": False},
                    "job_families": {
                        "type": "list",
                        "required": False,
                        "schema": {"type": "string"},
                    },
                },
            },
        },
        # Add other top-level keys as needed
    }
    v = Validator(schema)
    if not v.validate(config_data):
        raise ConfigLoadError(f"Config validation failed: {v.errors}")

    # 2. Merge top-level defaults
    if "new_hire_compensation_params" in config_data and "role_compensation_params" in config_data:
        default_params = config_data["new_hire_compensation_params"] or {}
        raw_roles = config_data["role_compensation_params"] or {}
        merged = {role: {**default_params, **overrides} for role, overrides in raw_roles.items()}
        config_data["role_compensation_params"] = merged

    # 3. Merge under global_parameters if present
    gp = config_data.get("global_parameters")
    if isinstance(gp, dict):
        # --- Flatten attrition & new_hires into global_parameters ---
        # 1) Pull attrition sub-dict up
        attr = gp.pop("attrition", {}) or {}
        gp.update(attr)
        # 2) Pull new_hires sub-dict up
        nh = gp.pop("new_hires", {}) or {}
        gp.update(nh)
        # 3) Move job_levels from root to global_parameters if present
        if "job_levels" in config_data and "job_levels" not in gp:
            gp["job_levels"] = config_data["job_levels"]
        # Always update config_data with the flattened gp
        config_data["global_parameters"] = gp
        # 4) Process compensation params
        nhcp = gp.get("new_hire_compensation_params")
        rcp = gp.get("role_compensation_params")
        if isinstance(nhcp, dict) and isinstance(rcp, dict):
            merged_gp_roles = {role: {**nhcp, **overrides} for role, overrides in rcp.items()}
            gp["role_compensation_params"] = merged_gp_roles
            config_data["global_parameters"] = gp

    # 4. Convert to namespace, preserving mapping keys
    namespace = dict_to_namespace(config_data)
    logger.debug(f"Configuration loaded into namespace: {namespace}")
    return namespace


# Expose for import
__all__ = [
    "load_yaml_config",
    "load_config_to_namespace",
    "ConfigLoadError",
]
