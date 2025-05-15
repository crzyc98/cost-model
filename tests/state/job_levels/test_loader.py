import os
import tempfile
import yaml
import pytest
from cost_model.state.job_levels.loader import load_job_levels_from_config, load_from_yaml, ConfigError

VALID_YAML = """
job_levels:
  - level_id: 0
    name: Test
    description: Desc
    min_compensation: 100
    max_compensation: 200
    comp_base_salary: 150
    comp_age_factor: 0.01
    comp_stochastic_std_dev: 0.2
    avg_annual_merit_increase: 0.03
    promotion_probability: 0.1
    target_bonus_percent: 0.05
    job_families: [X]
"""

INVALID_YAML = "not: valid: ["

def test_load_config_strict_valid():
    cfg = yaml.safe_load(VALID_YAML)
    levels = load_job_levels_from_config(cfg, strict_validation=True)
    assert 0 in levels
    lvl = levels[0]
    assert lvl.name == "Test"
    assert lvl.min_compensation == 100

def test_load_config_strict_missing_id():
    cfg = {"job_levels": [{}]}
    with pytest.raises(ConfigError):
        load_job_levels_from_config(cfg, strict_validation=True)

def test_load_from_yaml_file(tmp_path):
    p = tmp_path / "levels.yaml"
    p.write_text(VALID_YAML)
    levels = load_from_yaml(str(p), strict_validation=True)
    assert 0 in levels

def test_load_from_yaml_malformed(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(INVALID_YAML)
    with pytest.raises(ConfigError):
        load_from_yaml(str(p), strict_validation=True)