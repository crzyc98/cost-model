# engine/config_loader.py

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)


def _normalize_config(conf: Dict[str, Any]) -> Dict[str, Any]:
    """Purely normalize config: rename legacy keys, extract plan_rules, and return new dict."""
    # work on a copy to avoid mutating input
    new_conf = conf.copy()
    # Legacy top-level key renames
    for old, new in [
        ('annual_compensation_increase_rate','comp_increase_rate'),
        ('annual_termination_rate','termination_rate'),
        ('annual_growth_rate','hire_rate'),
    ]:
        if old in new_conf:
            new_conf[new] = new_conf.pop(old)
            logger.debug("Renamed config key %s -> %s", old, new)

    # Pull structured plan rule blocks into a nested dict
    plan_keys = [
        'eligibility',
        'auto_enrollment',
        'auto_increase',
        # structured plan rule blocks
        'employer_match',
        'employer_nec',
        'min_hours_worked',
        'last_day_work_rule',
        'match_change_response',
        'irs_limits',
        'behavioral_params',
    ]
    plan_rules: Dict[str, Any] = {}
    for key in plan_keys:
        if key in new_conf:
            plan_rules[key] = new_conf.pop(key)
    # always include plan_rules key
    new_conf['plan_rules'] = plan_rules
    logger.debug("Extracted plan_rules keys: %s", list(plan_rules.keys()))

    # support additional plan_rules transformations under plan_rules
    pr_cfg = new_conf.get('plan_rules', {})
    # Extract nested rate keys from plan_rules
    for old, new in [
        ('annual_compensation_increase_rate','comp_increase_rate'),
        ('annual_termination_rate','termination_rate'),
        ('annual_growth_rate','hire_rate'),
    ]:
        if old in pr_cfg:
            new_conf[new] = pr_cfg.pop(old)
            logger.debug("Renamed plan_rules key %s -> %s", old, new)
    if 'maintain_headcount' in pr_cfg:
        new_conf['maintain_headcount'] = pr_cfg.pop('maintain_headcount')
        logger.debug("Extracted maintain_headcount: %s", new_conf['maintain_headcount'])

    # Ensure plan_rules always present
    if 'plan_rules' not in new_conf:
        new_conf['plan_rules'] = {}
    return new_conf


def load_scenarios(config_path: str) -> List[Dict[str, Any]]:
    """
    Load one or more scenarios.

    - Reads engine defaults from `defaults/defaults.yaml`.
    - Reads user scenarios from the single file at `config_path`.
      * If that file has a top-level `scenarios:` mapping, each key becomes a scenario.
      * Otherwise the whole file is treated as one scenario.

    Returns:
        A list of scenario dicts, each with a `scenario_name` key.
    """
    # 1) load user-provided config
    user_cfg: Dict[str, Any] = yaml.safe_load(Path(config_path).read_text()) or {}

    # 2) load engine defaults
    defaults_file = Path(__file__).parent.parent / 'defaults' / 'defaults.yaml'
    defaults_cfg: Dict[str, Any] = yaml.safe_load(defaults_file.read_text()) or {}
    global_defaults = defaults_cfg.get('global_defaults', {})

    # 3) merge any user-global overrides without mutating defaults
    base_defaults = deepcopy(global_defaults)
    user_globals = user_cfg.get('global_parameters', {}) or {}
    base_defaults.update(user_globals)

    # 4) assemble scenarios
    scenarios: List[Dict[str, Any]] = []
    scenarios_cfg = user_cfg.get('scenarios', {}) or {}

    if scenarios_cfg:
        # multiple named scenarios
        for name, sc in scenarios_cfg.items():
            merged = deepcopy(base_defaults)
            # Deep-merge nested plan_rules: preserve defaults and override with scenario-specific
            scenario_pr = sc.get('plan_rules', None)
            # Merge top-level except plan_rules
            top_level = {k: v for k, v in sc.items() if k != 'plan_rules'}
            merged.update(top_level or {})
            # Handle plan_rules deep merge
            default_pr = deepcopy(base_defaults.get('plan_rules', {}))
            if scenario_pr is not None:
                default_pr.update(scenario_pr or {})
            merged['plan_rules'] = default_pr
            merged = _normalize_config(merged)
            merged['scenario_name'] = name
            scenarios.append(merged)
    else:
        # single scenario file
        sc = deepcopy(user_cfg)
        sc.pop('global_parameters', None)
        sc.pop('scenarios', None)

        name = sc.pop('scenario_name', None) \
               or sc.pop('name', None) \
               or Path(config_path).stem

        sc = _normalize_config(sc)
        sc['scenario_name'] = name
        scenarios.append(sc)

    # log load summary
    logger.info("Loaded %d scenarios from %s", len(scenarios), config_path)
    return scenarios