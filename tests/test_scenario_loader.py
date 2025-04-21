import pytest
import yaml
from cost_model.scenario_loader import load


def test_simple_load_file(tmp_path):
    # simple config, no extends
    cfg = {"a": 1, "b": {"c": 2}}
    f = tmp_path / "simple.yaml"
    f.write_text(yaml.safe_dump(cfg))
    assert load(str(f)) == cfg


def test_load_directory(tmp_path):
    # two scenario files in dir
    cfg1 = {"x": "X"}
    f1 = tmp_path / "one.yaml"
    f1.write_text(yaml.safe_dump(cfg1))
    cfg2 = {"y": "Y"}
    f2 = tmp_path / "two.yml"
    f2.write_text(yaml.safe_dump(cfg2))
    result = load(str(tmp_path))
    assert result == {"one": cfg1, "two": cfg2}


def test_extends_deep_merge(tmp_path):
    # grandparent -> parent -> child
    gp = tmp_path / "gp.yaml"
    gp_cfg = {"a": 1, "nested": {"k": 10}}
    gp.write_text(yaml.safe_dump(gp_cfg))
    p = tmp_path / "parent.yaml"
    p_cfg = {"extends": "gp.yaml", "nested": {"k": 20, "new": 30}, "b": 2}
    p.write_text(yaml.safe_dump(p_cfg))
    c = tmp_path / "child.yaml"
    c_cfg = {"extends": "parent.yaml", "nested": {"deep": 40}, "c": 3}
    c.write_text(yaml.safe_dump(c_cfg))
    merged = load(str(c))
    expected = {"a": 1, "nested": {"k": 20, "new": 30, "deep": 40}, "b": 2, "c": 3}
    assert merged == expected


def test_missing_parent_error(tmp_path):
    f = tmp_path / "child.yaml"
    # extends a non-existent file
    f.write_text(yaml.safe_dump({"extends": "nope.yaml", "foo": "bar"}))
    with pytest.raises(FileNotFoundError):
        load(str(f))
