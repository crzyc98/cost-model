import os
import pytest
from tempfile import NamedTemporaryFile
from cost_model.state.job_levels.init import init_job_levels, refresh_job_levels
from cost_model.state.job_levels.loader import load_from_yaml
from cost_model.state.job_levels.models import ConfigError

def test_init_with_defaults(monkeypatch):
    # No config path or env var
    monkeypatch.delenv("COST_MODEL_JOB_LEVELS_CONFIG", raising=False)
    success = init_job_levels(config_path=None)
    assert success is True

def test_init_with_nonexistent_path(monkeypatch):
    monkeypatch.setenv("COST_MODEL_JOB_LEVELS_CONFIG", "/does/not/exist.yaml")
    # Should fall back to defaults, not crash
    success = init_job_levels(config_path=None, strict_validation=False)
    assert success is True

def test_refresh_job_levels(monkeypatch):
    monkeypatch.setenv("COST_MODEL_JOB_LEVELS_CONFIG", "/does/not/exist.yaml")
    init_job_levels()
    # Now refresh should also return True and reset warnings
    assert refresh_job_levels() is True