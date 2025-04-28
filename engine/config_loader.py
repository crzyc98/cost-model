# engine/config_loader.py

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)


def _normalize_config(conf: Dict[str, Any]) -> Dict[str, Any]:
    """Map legacy top‐level keys into the engine’s unified config schema."""
    # Legacy renames for top-level growth/termination keys
    if 'annual_compensation_increase_rate' in conf:
        conf['comp_increase_rate'] = conf.pop('annual_compensation_increase_rate')
    if 'annual_termination_rate' in conf:
        conf['termination_rate'] = conf.pop('annual_termination_rate')
    if 'annual_growth_rate' in conf:
        conf['hire_rate'] = conf.pop('annual_growth_rate')

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
        if key in conf:
            plan_rules[key] = conf.pop(key)
    if plan_rules:
        conf['plan_rules'] = plan_rules

    # support additional plan_rules transformations under plan_rules
    pr_cfg = conf.get('plan_rules', {})
    # extract annual rates from plan_rules into top-level keys
    if 'annual_compensation_increase_rate' in pr_cfg:
        conf['comp_increase_rate'] = pr_cfg.pop('annual_compensation_increase_rate')
    if 'annual_termination_rate' in pr_cfg:
        conf['termination_rate'] = pr_cfg.pop('annual_termination_rate')
    if 'annual_growth_rate' in pr_cfg:
        conf['hire_rate'] = pr_cfg.pop('annual_growth_rate')
    # moved comp_increase_rate and hire_rate already handled above
    if 'maintain_headcount' in pr_cfg:
        conf['maintain_headcount'] = pr_cfg.pop('maintain_headcount')

    return conf


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
            _normalize_config(merged)
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

        _normalize_config(sc)
        sc['scenario_name'] = name
        scenarios.append(sc)

    # log load summary
    logger.info("Loaded %d scenarios from %s", len(scenarios), config_path)
    return scenarios