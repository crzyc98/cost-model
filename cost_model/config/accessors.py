# cost_model/config/accessors.py
"""
Helper functions to access and process configuration data, primarily handling
the merging of global parameters and scenario-specific overrides.
"""

import logging
from copy import deepcopy
from typing import Dict, Any, Optional, Union  # Added Union

# Import the Pydantic models
try:
    # Use relative import if accessors.py is inside config package
    from .models import (
        MainConfig,
        GlobalParameters,
        PlanRules,
    )
except ImportError:
    # Fallback if running standalone or structure differs
    try:
        from cost_model.config.models import (
            MainConfig,
            GlobalParameters,
            PlanRules,
        )
    except ImportError:
        print(
            "Warning (accessors.py): Could not import config models. Using Any type hint."
        )
        from typing import Any as MainConfig  # type: ignore
        from typing import Any as GlobalParameters  # type: ignore
        from typing import Any as PlanRules  # type: ignore


logger = logging.getLogger(__name__)


# --- Helper for Deep Merging Dictionaries ---
def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merges 'override' dict into 'base' dict.
    Creates new dictionaries for nested structures to avoid modifying originals.
    Simple values in override replace values in base.
    Handles nested PlanRules objects by converting them to dicts first if needed.
    """
    merged = deepcopy(base)  # Start with a copy of the base

    for key, override_value in override.items():
        base_value = merged.get(key)

        # Special handling for PlanRules merging if needed
        # This assumes override_value is already a dict if it's overriding plan_rules
        if (
            key == "plan_rules"
            and isinstance(base_value, dict)
            and isinstance(override_value, dict)
        ):
            # Make sure base_value is a dict (it should be from global_params.dict())
            merged[key] = _deep_merge_dicts(base_value, override_value)
        elif isinstance(base_value, dict) and isinstance(override_value, dict):
            # If both values are other dicts, recurse
            merged[key] = _deep_merge_dicts(base_value, override_value)
        elif override_value is not None:
            # Otherwise, override value takes precedence (if not None)
            merged[key] = deepcopy(override_value)
        # If override_value is None, the base_value (if exists) is kept due to deepcopy

    return merged


# --- Main Accessor Function ---


def get_scenario_config(
    main_config: MainConfig, scenario_name: str
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
        TypeError: If main_config is not a valid MainConfig object.
        ValueError: If the merged configuration fails validation.
    """
    logger.info(f"Resolving configuration for scenario: '{scenario_name}'")

    # 1. Get the specific scenario definition
    scenario_def = main_config.scenarios.get(scenario_name)
    if scenario_def is None:
        logger.error(f"Scenario '{scenario_name}' not found in the configuration.")
        raise KeyError(f"Scenario '{scenario_name}' not found.")

    # 2. Start with a deep copy of the global parameters dictionary
    # Use .dict() to work with dictionaries for merging, exclude defaults that weren't set
    global_dict = main_config.global_parameters.dict(
        exclude_unset=False
    )  # Keep defaults

    # 3. Get the scenario overrides dictionary, excluding None values and 'name'
    scenario_overrides = scenario_def.dict(exclude={"name"}, exclude_none=True)

    # 4. Perform the deep merge
    merged_config_dict = _deep_merge_dicts(global_dict, scenario_overrides)

    # 5. Re-validate the merged dictionary back into a GlobalParameters object
    try:
        # Pass the merged dictionary to the GlobalParameters model constructor
        resolved_config = GlobalParameters(**merged_config_dict)
        logger.info(
            f"Successfully resolved configuration for scenario: '{scenario_name}'"
        )
        return resolved_config
    except Exception as e:  # Catch Pydantic validation errors or others
        logger.exception(
            f"Error validating merged configuration for scenario '{scenario_name}': {e}\nMerged Dict: {merged_config_dict}"
        )
        raise ValueError(
            f"Merged configuration for scenario '{scenario_name}' failed validation."
        ) from e


def get_irs_limit_for_year(
    config: Union[GlobalParameters, PlanRules], year: int, limit_type: str
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
    # Determine where the plan_rules are located
    if isinstance(config, PlanRules):
        plan_rules = config
    elif hasattr(config, "plan_rules") and isinstance(config.plan_rules, PlanRules):
        plan_rules = config.plan_rules
    else:
        logger.error("Could not find PlanRules within the provided config object.")
        return None

    irs_limits_dict = plan_rules.irs_limits

    if not irs_limits_dict:
        logger.warning("IRS limits dictionary is empty in the configuration.")
        return None

    # Get limits for the specific year
    year_limits_model = irs_limits_dict.get(year)

    if year_limits_model is None:
        # Fallback to the latest available year if the specific year is missing
        try:
            latest_year = max(k for k in irs_limits_dict.keys() if isinstance(k, int))
            logger.warning(
                f"IRS limits for year {year} not found. Using limits from latest available year: {latest_year}"
            )
            year_limits_model = irs_limits_dict.get(latest_year)
        except ValueError:  # Handle case where dict is empty or has non-int keys
            logger.error("Could not find any valid year keys in IRS limits dictionary.")
            return None

        if (
            year_limits_model is None
        ):  # Should not happen if dict not empty, but check anyway
            logger.error(
                f"Could not find any IRS limits, even for fallback year {latest_year}."
            )
            return None

    # Access the specific limit using getattr from the Pydantic model instance
    limit_value = getattr(year_limits_model, limit_type, None)

    if limit_value is None:
        logger.warning(
            f"Limit type '{limit_type}' not found for year {year} (or fallback year)."
        )

    return limit_value
