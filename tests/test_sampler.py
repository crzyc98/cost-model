"""
Tests for ParameterSampler (distribution logic, reproducibility, error handling).
"""
import pytest
import json
# Update this import if ParameterSampler has moved:
from cost_model.sampler import ParameterSampler


def write_json(tmp_path, data):
    f = tmp_path / "mc.json"
    f.write_text(json.dumps(data))
    return str(f)


def test_constant_and_edge_distributions(tmp_path):
    data = {
        "pkg": {
            "a": {"dist": "constant", "value": 10},
            "b": {"dist": "uniform", "low": 5, "high": 5},
            "c": {"dist": "normal", "mean": 2, "sd": 0}
        }
    }
    path = write_json(tmp_path, data)
    sampler = ParameterSampler(path, seed=42)
    sample = sampler.sample("pkg")
    assert sample["a"] == 10
    assert sample["b"] == 5
    assert sample["c"] == pytest.approx(2)


def test_reproducible_with_same_seed(tmp_path):
    data = {"pkg": {"x": {"dist": "uniform", "low": 0, "high": 1},
                     "y": {"dist": "normal", "mean": 0, "sd": 1}}}
    path = write_json(tmp_path, data)
    s1 = ParameterSampler(path, seed=123).sample("pkg")
    s2 = ParameterSampler(path, seed=123).sample("pkg")
    assert s1 == s2


def test_missing_package_raises_key_error(tmp_path):
    data = {"pkg": {"a": {"dist": "constant", "value": 1}}}
    path = write_json(tmp_path, data)
    sampler = ParameterSampler(path, seed=1)
    with pytest.raises(KeyError):
        sampler.sample("none")


def test_unknown_distribution_raises_value_error(tmp_path):
    data = {"pkg": {"a": {"dist": "unknown", "value": 1}}}
    path = write_json(tmp_path, data)
    sampler = ParameterSampler(path, seed=1)
    with pytest.raises(ValueError):
        sampler.sample("pkg")
