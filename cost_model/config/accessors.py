# cost_model/config/accessors.py
"""
Helper functions to access and process configuration data, primarily handling
the merging of global parameters and scenario-specific overrides.
"""

import logging
from copy import deepcopy
from typing import Dict, Any, Optional

# Import the Pydantic models
try:
    from .models import MainConfig, GlobalParameters, ScenarioDefinition, PlanRules, IRSYearLimits
except ImportError:
    # Fallback for standalone execution or if models haven't been created/moved yet
    print("Warning: Could not import config models. Using Any type hint.")
    from typing import Any as MainConfig # type: ignore
    from typing import Any as GlobalParameters # type: ignore
    from typing import Any as ScenarioDefinition # type: ignore
    from typing import Any as PlanRules # type: ignore
    from typing import Any as IRSYearLimits # type: ignore


logger = logging.getLogger(__name__)

# --- Helper for Deep Merging Dictionaries ---
# (Could be moved to a generic utils module if used elsewhere)
def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merges 'override' dict into 'base' dict.
    Creates new dictionaries for nested structures to avoid modifying originals.
    Simple values in override replace values in base.
    """
    merged = deepcopy(base) # Start with a copy of the base

    for key, override_value in override.items():
        base_value = merged.get(key)

        if isinstance(base_value, dict) and isinstance(override_value, dict):
            # If both values are dicts, recurse
            merged[key] = _deep_merge_dicts(base_value, override_value)
        elif override_value is not None: # Allow override with None if explicitly set? Decide policy.
            # Otherwise, override value takes precedence (if not None)
            # Use deepcopy for nested lists/dicts in the override value itself
            merged[key] = deepcopy(override_value)
        # If override_value is None and key exists in base, keep base value (or handle None override explicitly)

    return merged

# --- Main Accessor Function ---

def get_scenario_config(
    main_config: MainConfig,
    scenario_name: str
) -> GlobalParameters:
    """
    Retrieves the configuration for a specific scenario, merging global
    parameters with scenario-specific overrides.

    Args:
        main_config: The validated MainConfig object containing global params
                     and all scenario definitions.
        scenario_name: The name of the scenario to retrieve configuration for.

    Returns:
        A GlobalParameters object representing the fully resolved configuration
        for the requested scenario.

    Raises:
        KeyError: If the specified scenario_name does not exist in the config.
        TypeError: If main_config is not a valid MainConfig object (if type hint used).
    """
    logger.info(f"Resolving configuration for scenario: '{scenario_name}'")

    # 1. Get the specific scenario definition
    scenario_def = main_config.scenarios.get(scenario_name)
    if scenario_def is None:
        logger.error(f"Scenario '{scenario_name}' not found in the configuration.")
        raise KeyError(f"Scenario '{scenario_name}' not found.")

    # 2. Start with a deep copy of the global parameters dictionary
    # Use .dict() to work with dictionaries for merging, then re-validate
    global_dict = main_config.global_parameters.dict(exclude_unset=True)

    # 3. Get the scenario overrides dictionary, excluding None values and 'name'
    scenario_overrides = scenario_def.dict(exclude={'name'}, exclude_none=True)

    # 4. Perform the deep merge
    # The _deep_merge_dicts handles nested structures like plan_rules
    merged_config_dict = _deep_merge_dicts(global_dict, scenario_overrides)

    # 5. Re-validate the merged dictionary back into a GlobalParameters object
    # This ensures the merged result still conforms to the expected structure
    # and applies Pydantic defaults for any fields missing after merge.
    try:
        resolved_config = GlobalParameters(**merged_config_dict)
        logger.info(f"Successfully resolved configuration for scenario: '{scenario_name}'")
        return resolved_config
    except Exception as e: # Catch Pydantic validation errors or others
        logger.exception(f"Error validating merged configuration for scenario '{scenario_name}': {e}")
        raise ValueError(f"Merged configuration for scenario '{scenario_name}' failed validation.") from e


def get_irs_limit_for_year(
    config: Union[GlobalParameters, PlanRules],
    year: int,
    limit_type: str
) -> Optional[Any]:
    """
    Safely retrieves a specific IRS limit for a given year.

    Args:
        config: Either the resolved GlobalParameters object for a scenario
                or just the PlanRules object.
        year: The year for which to retrieve the limit.
        limit_type: The name of the limit to retrieve (e.g.,
                    'compensation_limit', 'deferral_limit', 'catchup_limit').

    Returns:
        The value of the requested limit for the year, or None if not found.
    """
    plan_rules = config if isinstance(config, PlanRules) else config.plan_rules
    irs_limits_dict = plan_rules.irs_limits

    if not irs_limits_dict:
        logger.warning("IRS limits dictionary is empty in the configuration.")
        return None

    year_limits = irs_limits_dict.get(year)

    if year_limits is None:
        # Fallback to the latest available year if the specific year is missing
        latest_year = max(irs_limits_dict.keys())
        logger.warning(f"IRS limits for year {year} not found. Using limits from latest available year: {latest_year}")
        year_limits = irs_limits_dict.get(latest_year)
        if year_limits is None: # Should not happen if dict not empty, but check anyway
             logger.error(f"Could not find any IRS limits, even for fallback year {latest_year}.")
             return None

    # Access the specific limit using getattr, returning None if the attribute doesn't exist
    limit_value = getattr(year_limits, limit_type, None)

    if limit_value is None:
        logger.warning(f"Limit type '{limit_type}' not found for year {year} (or fallback year).")

    return limit_value


# --- Example Usage ---
if __name__ == '__main__':
    # This block is for demonstration/testing purposes only
    import yaml
    from pathlib import Path
    # Assuming models.py is in the same directory for this example run
    try:
        from models import MainConfig
        from loaders import load_yaml_config, ConfigLoadError
    except ImportError:
         print("Run this example from the 'cost_model/config' directory or adjust paths.")
         sys.exit(1)


    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s")

    # Load the main config file (adjust path as needed)
    try:
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'config.yaml'
        print(f"Loading main config from: {config_path}")
        raw_config_data = load_yaml_config(config_path)
        if not raw_config_data:
            print("Failed to load raw config data.")
            sys.exit(1)

        # Validate the main config
        main_config_obj = MainConfig(**raw_config_data)
        print("Main config loaded and validated.")

    except (ConfigLoadError, FileNotFoundError, Exception) as e:
        print(f"Error loading/validating main config: {e}")
        sys.exit(1)

    # --- Example 1: Get Baseline Config ---
    print("\n--- Getting Baseline Scenario Config ---")
    try:
        baseline_config = get_scenario_config(main_config_obj, 'baseline')
        print(f"Scenario Name (from original def): {main_config_obj.scenarios['baseline'].name}")
        print(f"Resolved Start Year: {baseline_config.start_year}")
        print(f"Resolved Eligibility Min Age: {baseline_config.plan_rules.eligibility.min_age}")
        print(f"Resolved Match Rate (Tier 0): {baseline_config.plan_rules.employer_match.tiers[0].match_rate}")
        print(f"Resolved NEC Rate: {baseline_config.plan_rules.employer_nec.rate}")

        # Example: Get IRS limit using helper
        comp_limit_2026 = get_irs_limit_for_year(baseline_config, 2026, 'compensation_limit')
        print(f"Resolved IRS Comp Limit for 2026: {comp_limit_2026}")
        catchup_2028 = get_irs_limit_for_year(baseline_config.plan_rules, 2028, 'catchup_limit') # Can pass PlanRules too
        print(f"Resolved IRS Catchup Limit for 2028: {catchup_2028}")
        missing_limit = get_irs_limit_for_year(baseline_config, 2030, 'deferral_limit') # Test fallback
        print(f"Resolved IRS Deferral Limit for 2030 (fallback): {missing_limit}")


    except Exception as e:
        print(f"Error getting baseline config: {e}")

    # --- Example 2: Get Scenario with Overrides ---
    print("\n--- Getting 'new_hire_auto_escalate' Scenario Config ---")
    try:
        nhae_config = get_scenario_config(main_config_obj, 'new_hire_auto_escalate')
        print(f"Scenario Name (from original def): {main_config_obj.scenarios['new_hire_auto_escalate'].name}")
        # Check inherited value
        print(f"Resolved Eligibility Min Age: {nhae_config.plan_rules.eligibility.min_age}")
        # Check overridden value
        print(f"Resolved Auto Increase Enabled: {nhae_config.plan_rules.auto_increase.enabled}")
        print(f"Resolved Auto Increase Rate: {nhae_config.plan_rules.auto_increase.increase_rate}")
        print(f"Resolved Auto Increase New Hires Only: {nhae_config.plan_rules.auto_increase.apply_to_new_hires_only}")
        # Check value that wasn't overridden in this scenario but exists globally
        print(f"Resolved NEC Rate: {nhae_config.plan_rules.employer_nec.rate}")

    except Exception as e:
        print(f"Error getting nhae_config config: {e}")

    # --- Example 3: Get TinyDev Config ---
    print("\n--- Getting TinyDev Scenario Config (from dev_tiny.yaml) ---")
    try:
        config_path_tiny = Path(__file__).parent.parent.parent / 'configs' / 'dev_tiny.yaml'
        print(f"Loading tiny config from: {config_path_tiny}")
        raw_config_tiny = load_yaml_config(config_path_tiny)
        if not raw_config_tiny:
             print("Failed to load raw tiny config data.")
             sys.exit(1)
        main_config_tiny = MainConfig(**raw_config_tiny)
        print("Tiny config loaded and validated.")

        tiny_config = get_scenario_config(main_config_tiny, 'baseline') # 'baseline' scenario in tiny file is named 'TinyDev'
        print(f"Scenario Name (from original def): {main_config_tiny.scenarios['baseline'].name}")
        print(f"Resolved Maintain Headcount: {tiny_config.maintain_headcount}")
        print(f"Resolved Termination Rate: {tiny_config.annual_termination_rate}")
        print(f"Resolved Auto Enroll Enabled: {tiny_config.plan_rules.auto_enrollment.enabled}")
        print(f"Resolved NEC Rate: {tiny_config.plan_rules.employer_nec.rate}")

    except Exception as e:
        print(f"Error getting TinyDev config: {e}")
