import os
import yaml

def load(path):
    """
    Load one or many scenario YAML(s), resolve `extends` chains, and return merged configs.

    If `path` is a dir, returns {scenario_name: config_dict, ...}.
    If `path` is a file, returns config_dict.
    """
    def _deep_merge(a: dict, b: dict) -> dict:
        """Recursively merge b over a and return new dict."""
        out = dict(a)
        for k, v in b.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = _deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    def _load_file(fp: str, seen=None) -> dict:
        if seen is None:
            seen = set()
        real = os.path.realpath(fp)
        if real in seen:
            raise RuntimeError(f"Circular extends detected: {fp}")
        seen.add(real)

        with open(fp) as f:
            data = yaml.safe_load(f) or {}

        parent_cfg = {}
        if "extends" in data:
            parent = data.pop("extends")
            # allow parent to be a relative path or name.yaml
            parent_fp = os.path.join(os.path.dirname(fp), parent)
            if not os.path.exists(parent_fp):
                parent_fp_yaml = parent_fp + (".yaml" if not parent.endswith((".yml", ".yaml")) else "")
                if os.path.exists(parent_fp_yaml):
                    parent_fp = parent_fp_yaml
            if not os.path.exists(parent_fp):
                raise FileNotFoundError(f"Cannot find parent '{parent}' for '{fp}'")
            parent_cfg = _load_file(parent_fp, seen)

        return _deep_merge(parent_cfg, data)

    if os.path.isdir(path):
        scenarios = {}
        for fn in os.listdir(path):
            if fn.lower().endswith((".yaml", ".yml")):
                name = os.path.splitext(fn)[0]
                scenarios[name] = _load_file(os.path.join(path, fn))
        return scenarios

    # single file
    return _load_file(path)
