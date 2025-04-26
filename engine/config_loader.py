import yaml
from copy import deepcopy
from pathlib import Path

def _normalize_config(conf: dict):
    # Map YAML keys to engine config keys
    if 'annual_compensation_increase_rate' in conf:
        conf['comp_increase_rate'] = conf.pop('annual_compensation_increase_rate')
    if 'annual_termination_rate' in conf:
        conf['termination_rate'] = conf.pop('annual_termination_rate')
    if 'annual_growth_rate' in conf:
        conf['hire_rate'] = conf.pop('annual_growth_rate')
    # Extract plan rules defined at root
    plan_keys = [
        'eligibility', 'auto_enrollment', 'auto_increase',
        'employer_match_formula', 'employer_nec_formula',
        'min_hours_worked', 'last_day_work_rule', 'match_change_response'
    ]
    plan_rules = {}
    for key in plan_keys:
        if key in conf:
            plan_rules[key] = conf.pop(key)
    if plan_rules:
        conf['plan_rules'] = plan_rules
    return conf

def load_scenarios(config_path: str):
    """
    Load scenarios from a YAML at config_path.
    Expects optional top-level 'global_parameters' and 'scenarios' mapping.
    If 'scenarios' is absent, treat the entire file as one scenario.
    Returns a list of scenario dicts with 'scenario_name' keys.
    """
    cfg = yaml.safe_load(Path(config_path).read_text()) or {}
    global_params = cfg.get('global_parameters', {}) or {}
    scenarios_cfg = cfg.get('scenarios', {}) or {}
    scenarios = []
    if scenarios_cfg:
        # Multiple scenarios defined
        for name, sc in scenarios_cfg.items():
            merged = deepcopy(global_params)
            merged.update(sc or {})
            _normalize_config(merged)
            merged['scenario_name'] = name
            scenarios.append(merged)
    else:
        # Single scenario file
        sc = deepcopy(cfg)
        # remove global_parameters and scenarios keys
        sc.pop('global_parameters', None)
        sc.pop('scenarios', None)
        # determine scenario name
        name = sc.pop('scenario_name', None) or sc.pop('name', None) or Path(config_path).stem
        _normalize_config(sc)
        sc['scenario_name'] = name
        scenarios.append(sc)
    return scenarios
