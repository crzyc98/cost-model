# tests/test_plan_rules_smoke.py
import os
import yaml
import pytest
import copy
from utils.rules.validators import PlanRules, ValidationError

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def validate_rules(raw: dict, context: str):
    try:
        # this will raise ValidationError if any field is missing or wrong
        PlanRules(**raw)
    except ValidationError as e:
        pytest.fail(f"{context} plan_rules validation failed:\n{e}")

def deep_merge(a: dict, b: dict) -> dict:
    """Recursively merge b into a without modifying inputs."""
    result = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result

def test_global_plan_rules():
    cfg = load_config()
    global_pr = cfg.get('global_parameters', {}).get('plan_rules', {})
    validate_rules(global_pr, "Global")

def test_each_scenario_plan_rules():
    cfg = load_config()
    global_pr = cfg.get('global_parameters', {}).get('plan_rules', {})
    scenarios = cfg.get('scenarios', {})
    for name, sc in scenarios.items():
        merged = deep_merge(global_pr, sc.get('plan_rules', {}))
        validate_rules(merged, f"Scenario '{name}'")