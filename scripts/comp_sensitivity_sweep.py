"""
Compensation Sensitivity Sweep Script

This script performs a parameter sweep over compensation settings (new_hire and role-based)
and collects year-by-year metrics stratified by employment_status. It is fully extensible:
- Specify which parameters to sweep and their ranges below.
- Handles arbitrary nested config parameters.
- Aggregates and analyzes results for business target filtering.

Usage:
    python scripts/comp_sensitivity_sweep.py
"""
import itertools
import yaml
import numpy as np
import pandas as pd
from types import SimpleNamespace
from pathlib import Path
import copy
from ruamel.yaml import YAML

from cost_model.config.loaders import load_config_to_namespace
from cost_model.engines.run_one_year import run_one_year
from cost_model.state.event_log import EVENT_COLS
from cost_model.projections.hazard import build_hazard_table
from cost_model.utils.columns import EMP_ROLE, EMP_TENURE_BAND  

# Add skopt for Bayesian optimization
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# === Requirements ===
# pip install scikit-optimize

# ---- USER CONFIGURABLE SECTION ----
# Specify the config file and data paths
CONFIG_PATH = Path("config/dev_tiny.yaml")
STARTING_SNAPSHOT_PATH = Path("data/census_preprocessed.parquet")

# Specify the compensation parameters to optimize and their bounds
# Key format: nested path as tuple, e.g., ("compensation", "new_hire", "comp_base_salary")
PARAM_BOUNDS = [
    (("compensation", "new_hire", "comp_base_salary"),   (90000, 130000)),
    (("compensation", "roles", "Staff", "comp_age_factor"),     (0.003, 0.012)),
    (("compensation", "roles", "Manager", "comp_age_factor"),   (0.003, 0.012)),
    (("compensation", "roles", "Executive", "comp_age_factor"), (0.003, 0.012)),
]

# Build skopt space and name mapping
SKOPT_SPACE = [
    Real(bounds[0], bounds[1], name=f"param_{i}")
    for i, (_, bounds) in enumerate(PARAM_BOUNDS)
]

# Map skopt param indices to config paths
SKOPT_PARAM_PATHS = [path for path, _ in PARAM_BOUNDS]


# ---- END USER CONFIGURABLE SECTION ----

def dict_to_namespace(d):
    """Recursively convert a dict to a SimpleNamespace, handling lists and non-string keys."""
    if isinstance(d, dict):
        # Only use string keys
        return SimpleNamespace(**{str(k): dict_to_namespace(v) for k, v in d.items() if isinstance(k, str)})
    elif isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    return d

def set_nested_attr(obj, attr_path, value):
    """Set a nested attribute (or dict key) given a tuple path."""
    for key in attr_path[:-1]:
        if hasattr(obj, key):
            obj = getattr(obj, key)
        elif isinstance(obj, dict):
            obj = obj[key]
        else:
            raise KeyError(f"Cannot traverse key '{key}' in {obj}")
    last = attr_path[-1]
    if hasattr(obj, last):
        setattr(obj, last, value)
    elif isinstance(obj, dict):
        obj[last] = value
    else:
        raise KeyError(f"Cannot set key '{last}' in {obj}")

def run_simulation_and_penalty(param_values, base_cfg, pid=0):
    # Map param_values to config paths
    param_dict = {path: val for path, val in zip(SKOPT_PARAM_PATHS, param_values)}
    ns = load_config_to_namespace(CONFIG_PATH)
    for path, val in param_dict.items():
        set_nested_attr(ns.global_parameters, path, val)
    event_log = pd.DataFrame(columns=EVENT_COLS)
    snapshot = pd.read_parquet(STARTING_SNAPSHOT_PATH)
    if EMP_TENURE_BAND not in snapshot.columns:
        snapshot[EMP_TENURE_BAND] = 'all'
    if 'employment_status' not in snapshot.columns:
        snapshot['employment_status'] = 'Active'
    plan_rules_ns = dict_to_namespace(base_cfg["plan_rules"])
    years = list(range(
        ns.global_parameters.start_year,
        ns.global_parameters.start_year + ns.global_parameters.projection_years
    ))
    hazard_table = build_hazard_table(
        years,
        snapshot,
        ns.global_parameters,
        plan_rules_ns
    )
    hazard_table["cfg"] = [plan_rules_ns] * len(hazard_table)
    records = []
    for year in range(ns.global_parameters.start_year,
                      ns.global_parameters.start_year + ns.global_parameters.projection_years):
        event_log, snapshot = run_one_year(
            event_log,
            snapshot,
            year,
            SimpleNamespace(**{
                **ns.global_parameters.__dict__,
                'plan_rules': plan_rules_ns
            }),
            None,
            hazard_table,
            np.random.default_rng(1000 + pid),
            ns.global_parameters.census_template_path
        )
        for (is_active, emp_role), grp in snapshot.groupby(["active", EMP_ROLE]):
            status = 'Active' if is_active else 'Terminated'
            avg_c = grp["employee_gross_compensation"].mean()
            tot_c = grp["employee_gross_compensation"].sum()
            records.append({
                "year": year,
                "status": status,
                "role": emp_role,
                **{'.'.join(path): val for path, val in param_dict.items()},
                "avg_comp": avg_c,
                "total_comp": tot_c,
                "headcount": len(grp)
            })
    df = pd.DataFrame(records)
    df = df.sort_values(["status", "year"]).reset_index(drop=True)
    df["avg_growth"] = (
        df.groupby(["status"])["avg_comp"]
        .pct_change()
        .fillna(0)
    )
    tg = base_cfg['global_parameters'].get('target_growth', None)
    if tg is None:
        raise RuntimeError("Please define `target_growth` under global_parameters in your YAML.")
    df['penalty'] = (df['avg_growth'] - tg) ** 2
    total_penalty = df['penalty'].sum()
    return total_penalty, param_dict, df

# Decorator for skopt
@use_named_args(SKOPT_SPACE)
def objective(*args, **kwargs):
    # skopt passes named args, but we use positional order
    param_values = [kwargs[f"param_{i}"] for i in range(len(SKOPT_PARAM_PATHS))]
    # Load base config once per call
    yaml = YAML()
    with open(CONFIG_PATH, "r") as f:
        base_cfg = yaml.load(f)
    penalty, _, _ = run_simulation_and_penalty(param_values, base_cfg)
    print(f"Params: {param_values} → penalty: {penalty:.5f}")
    return penalty

def main():
    yaml = YAML()
    with open(CONFIG_PATH, "r") as f:
        base_cfg = yaml.load(f)
    # Run Bayesian optimization
    print("Starting Bayesian optimization (gp_minimize)...")
    res = gp_minimize(
        objective,
        SKOPT_SPACE,
        n_calls=60,  # You can increase this for more thorough search
        random_state=42,
        verbose=True
    )
    print("\nBest parameters:", res.x)
    print("Best penalty:", res.fun)
    # Run one more time to get full results for the best params
    penalty, best_param_dict, df = run_simulation_and_penalty(res.x, base_cfg, pid=0)
    print("\nBest parameter values:")
    for path, val in best_param_dict.items():
        print(f"  {'.'.join(path)} = {val}")
    # Save results
    output_dir = Path("output_dev")
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "comp_bayes_sweep_results.csv", index=False)
    print(f"Results saved to {output_dir / 'comp_bayes_sweep_results.csv'}")
    # Write best config to YAML
    import ruamel.yaml
    yaml = ruamel.yaml.YAML()
    cfg = yaml.load(open(CONFIG_PATH))
    for path, val in best_param_dict.items():
        d = cfg['global_parameters']
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = float(val)
    with open("config/best_compensation.yaml", "w") as f:
        yaml.dump(cfg, f)
    print("Best config written to config/best_compensation.yaml")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compensation Sensitivity Sweep")
    parser.add_argument('--mode', choices=['grid', 'bayes'], default='bayes', help='Sweep mode: grid or bayes (default: bayes)')
    parser.add_argument('--n_calls', type=int, default=60, help='Number of calls for Bayesian optimization')
    args = parser.parse_args()
    def run_grid():
        print("Running grid search...")
        sweep_keys, sweep_values = zip(*PARAM_BOUNDS)
        grid = [dict(zip(SKOPT_PARAM_PATHS, vals)) for vals in itertools.product(*[np.linspace(*b, 10) for _, b in PARAM_BOUNDS])]
        records = []
        yaml_loader = YAML()
        with open(CONFIG_PATH, "r") as f:
            base_cfg = yaml_loader.load(f)
        # Optionally: Use concurrent.futures.ProcessPoolExecutor for parallel grid runs
        for pid, param_dict in enumerate(grid):
            param_values = [param_dict[path] for path in SKOPT_PARAM_PATHS]
            penalty, best_param_dict, df = run_simulation_and_penalty(param_values, base_cfg, pid=pid)
            print(f"Grid {pid}: {param_values} → penalty: {penalty:.5f}")
            # Save/aggregate records as needed (see bayes branch for output logic)
        print("Grid search complete.")
    def run_bayes():
        print("Running Bayesian optimization...")
        yaml_loader = YAML()
        with open(CONFIG_PATH, "r") as f:
            base_cfg = yaml_loader.load(f)
        res = gp_minimize(
            objective,
            SKOPT_SPACE,
            n_calls=args.n_calls,
            random_state=42,
            verbose=True
        )
        print("\nBest parameters:", res.x)
        print("Best penalty:", res.fun)
        penalty, best_param_dict, df = run_simulation_and_penalty(res.x, base_cfg, pid=0)
        print("\nBest parameter values:")
        for path, val in best_param_dict.items():
            print(f"  {'.'.join(path)} = {val}")
        output_dir = Path("output_dev")
        output_dir.mkdir(exist_ok=True)
        df.to_csv(output_dir / "comp_bayes_sweep_results.csv", index=False)
        print(f"Results saved to {output_dir / 'comp_bayes_sweep_results.csv'}")
        import ruamel.yaml
        yaml = ruamel.yaml.YAML()
        cfg = yaml.load(open(CONFIG_PATH))
        for path, val in best_param_dict.items():
            d = cfg['global_parameters']
            for key in path[:-1]:
                d = d.setdefault(key, {})
            d[path[-1]] = float(val)
        with open("config/best_compensation.yaml", "w") as f:
            yaml.dump(cfg, f)
        print("Best config written to config/best_compensation.yaml")
    # --- Main execution ---
    if args.mode == 'grid':
        run_grid()
    else:
        run_bayes()

# Notes for future speedup:
# - You can cache the hazard table if it does not depend on sweep parameters.
# - For grid mode, use concurrent.futures.ProcessPoolExecutor for parallel runs.
# - For bayes mode, consider using skopt's n_jobs for parallel evaluations (if your objective is picklable).
