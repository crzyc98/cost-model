# cost_model/config/loaders.py

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

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
