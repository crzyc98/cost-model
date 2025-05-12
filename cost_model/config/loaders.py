# cost_model/config/loaders.py

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple
from types import SimpleNamespace
from cerberus import Validator

# Keys whose values should remain as dicts rather than namespaces
# These can be nested paths like 'global_parameters.new_hire_compensation_params'
DICT_KEYS = {
    'role_compensation_params',
    'new_hire_compensation_params',
    'global_parameters.role_compensation_params',
    'global_parameters.new_hire_compensation_params',
    # add any other mapping-style sections here
}

def should_remain_dict(key_path: str) -> bool:
    """
    Check if a key path should remain a dict based on DICT_KEYS.
    """
    return key_path in DICT_KEYS or any(key_path.endswith(f'.{k}') for k in DICT_KEYS)

def dict_to_namespace(obj: Any, path: str = '') -> Any:
    """
    Recursively convert dicts to SimpleNamespace, except for paths matching DICT_KEYS.
    Lists will be converted elementwise.
    """
    if isinstance(obj, dict):
        # If this path should remain a dict, return it as is
        if should_remain_dict(path):
            return obj
        
        # Otherwise convert to namespace, but check each child
        converted = {}
        for k, v in obj.items():
            child_path = f"{path}.{k}" if path else k
            converted[k] = dict_to_namespace(v, child_path)
        return SimpleNamespace(**converted)
    elif isinstance(obj, list):
        return [dict_to_namespace(v, path) for v in obj]
    else:
        return obj

# Configure logger for this module
logger = logging.getLogger(__name__)


# Define a custom exception for configuration loading errors
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
        config_path = Path(config_path)  # Ensure it's a Path object

    logger.info(f"Attempting to load configuration from: {config_path}")

    try:
        # Check if the file exists
        if not config_path.is_file():
            logger.error(f"Configuration file not found at path: {config_path}")
            raise ConfigLoadError(f"Configuration file not found: {config_path}")

        # Open and parse the YAML file
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)  # Use safe_load for security

        if not isinstance(config_data, dict):
            # Handle cases where YAML is valid but not a dictionary at the top level
            logger.error(
                f"Configuration file {config_path} did not parse into a dictionary."
            )
            raise ConfigLoadError(
                f"Invalid configuration format in {config_path}: Expected a dictionary."
            )

        logger.info(f"Successfully loaded configuration from {config_path}")
        return config_data

    except FileNotFoundError as e:
        # This case is technically covered by is_file(), but good practice to handle
        logger.error(f"Configuration file not found (FileNotFoundError): {config_path}")
        raise ConfigLoadError(f"Configuration file not found: {config_path}") from e
    except yaml.YAMLError as e:
        logger.exception(f"Error parsing YAML configuration file {config_path}: {e}")
        raise ConfigLoadError(f"Error parsing YAML file {config_path}") from e
    except Exception as e:
        # Catch any other unexpected errors during file handling or loading
        logger.exception(
            f"An unexpected error occurred while loading config {config_path}: {e}"
        )
        raise ConfigLoadError(f"Unexpected error loading config {config_path}") from e


def load_config_to_namespace(config_path: Path) -> SimpleNamespace:
    """
    Loads YAML, validates its schema, merges role defaults, and converts to a SimpleNamespace at the top level.
    - Mapping-style sections in DICT_KEYS remain dicts (including nested paths).
    - role_compensation_params is merged with new_hire_compensation_params defaults for each role.
    - Raises ConfigLoadError on schema or validation errors.
    """
    config_data = load_yaml_config(config_path)
    if config_data is None:
        raise ConfigLoadError(f"No config at {config_path}")

    # --- 1. Schema validation (expand as needed) ---
    schema = {
        'global_parameters': {'type': 'dict', 'required': False},
        'new_hire_compensation_params': {'type': 'dict', 'required': False},
        'role_compensation_params': {'type': 'dict', 'required': False},
        # Add other top-level keys as needed
    }
    v = Validator(schema)
    if not v.validate(config_data):
        raise ConfigLoadError(f"Config validation failed: {v.errors}")

    # --- 2. Handle default merging for role_compensation_params ---
    # Top-level params
    if 'new_hire_compensation_params' in config_data and 'role_compensation_params' in config_data:
        default_params = config_data['new_hire_compensation_params']
        raw_role = config_data['role_compensation_params']
        merged_role = {}
        for role, overrides in raw_role.items():
            merged_role[role] = {**default_params, **overrides}
        config_data['role_compensation_params'] = merged_role
    
    # Nested params under global_parameters
    if 'global_parameters' in config_data and isinstance(config_data['global_parameters'], dict):
        global_params = config_data['global_parameters']
        if 'new_hire_compensation_params' in global_params and 'role_compensation_params' in global_params:
            default_params = global_params['new_hire_compensation_params']
            raw_role = global_params['role_compensation_params']
            merged_role = {}
            for role, overrides in raw_role.items():
                merged_role[role] = {**default_params, **overrides}
            global_params['role_compensation_params'] = merged_role

    # Convert to namespace with path-based dict preservation
    return dict_to_namespace(config_data)

# Expose for import
__all__ = [
    'load_yaml_config',
    'load_config_to_namespace',
    'ConfigLoadError',
]

# Example Usage (could be in another module like config/accessors.py or simulation.py)
if __name__ == "__main__":
    # This block is for demonstration/testing purposes only
    # In a real scenario, you'd call load_yaml_config from elsewhere

    # Configure basic logging for the example
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s",
    )

    # Assume a dummy config file exists at ../configs/example_config.yaml relative to this file
    # You would replace this with the actual path logic needed in your application
    try:
        # Construct path relative to this file's location for example
        example_config_path = (
            Path(__file__).parent.parent.parent / "configs" / "dev_tiny.yaml"
        )
        print(f"Looking for example config at: {example_config_path}")

        loaded_config = load_yaml_config(example_config_path)

        if loaded_config:
            print("\n--- Example Loaded Config ---")
            # print(yaml.dump(loaded_config, default_flow_style=False)) # Pretty print YAML
            print("Global Parameters:", loaded_config.get("global_parameters", {}))
            print("Scenarios:", list(loaded_config.get("scenarios", {}).keys()))
        else:
            print("Config loading returned None (error should have been raised).")

    except ConfigLoadError as e:
        print("\n--- CONFIGURATION ERROR ---")
        print(e)
    except Exception as e:
        print("\n--- UNEXPECTED ERROR ---")
        print(e)
