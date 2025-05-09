#!/usr/bin/env python3
"""
Phase III: Monte Carlo wrapper.
Produces:
  - mc_raw.parquet    (every row = one employee, tagged with rep/scenario/year)
  - mc_metrics.csv    (one row per rep/scenario/year with your 15 summary stats)
"""
import argparse
import os
import sys
import yaml
import pandas as pd
from cost_model.utils.projection_utils import project_hr, apply_plan_rules
from cost_model.rules.validators import PlanRules, ValidationError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_single(config, census_df, seed):
    hr = project_hr(census_df, config["baseline_cfg"], random_seed=seed)
    raw_rows, sum_rows = [], []

    for scen_key, scen in config["scenarios"].items():
        # build scen_cfg (same as run_plan_rules.py)
        scen_cfg = dict(config["global_params"])
        for k, v in scen.items():
            if k != "plan_rules":
                scen_cfg[k] = v
        pr = dict(config["global_params"]["plan_rules"])
        pr.update(scen.get("plan_rules", {}))
        scen_cfg["plan_rules"] = pr

        for year, df in hr.items():
            df2 = apply_plan_rules(df, scen_cfg, year)

            # --- raw, employee-level ---
            df2 = df2.assign(rep=seed, scenario=scen_key, year=year)
            raw_rows.append(df2)

            # --- summary metrics ---
            hc = len(df2)
            elig = int(df2["is_eligible"].sum())
            part = int(df2["is_participating"].sum())
            # avoid division by zero
            p_rate_elig = part / elig if elig > 0 else 0.0
            p_rate_tot = part / hc if hc > 0 else 0.0

            # deferral rates
            avg_def_part = df2.loc[df2["is_participating"], "deferral_rate"].mean()
            avg_def_tot = df2["deferral_rate"].mean()

            # contributions — adjust column names to match what apply_contributions creates!
            emp_pre_tax = df2["pre_tax_contributions"].sum()
            emp_match = df2["employer_match_contribution"].sum()
            emp_nec = df2["employer_non_elective_contribution"].sum()
            emp_cost = emp_match + emp_nec

            total_contrib = emp_pre_tax + emp_match + emp_nec
            plan_comp = df2["plan_year_compensation"].sum()
            cap_comp = df2["capped_compensation"].sum()

            sum_rows.append(
                {
                    "rep": seed,
                    "scenario": scen_key,
                    "year": year,
                    "headcount": hc,
                    "eligible": elig,
                    "participating": part,
                    "participation_rate_eligible": p_rate_elig,
                    "participation_rate_total": p_rate_tot,
                    "avg_deferral_rate_participants": avg_def_part,
                    "avg_deferral_rate_total": avg_def_tot,
                    "total_employee_pre_tax": emp_pre_tax,
                    "total_employer_match": emp_match,
                    "total_employer_nec": emp_nec,
                    "total_employer_cost": emp_cost,
                    "total_contributions": total_contrib,
                    "total_plan_year_compensation": plan_comp,
                    "total_capped_compensation": cap_comp,
                    "employer_cost_pct_plan_comp": (
                        emp_cost / plan_comp if plan_comp > 0 else 0
                    ),
                    "employer_cost_pct_capped_comp": (
                        emp_cost / cap_comp if cap_comp > 0 else 0
                    ),
                }
            )

    return pd.concat(raw_rows, ignore_index=True), pd.DataFrame(sum_rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("-d", "--census", required=True)
    p.add_argument("-n", "--nreps", type=int, default=100)
    p.add_argument("-o", "--outfile", default="results/mc_metrics.csv")
    args = p.parse_args()

    # load
    cfg_full = yaml.safe_load(open(args.config))
    census_df = pd.read_csv(
        args.census, parse_dates=["hire_date", "termination_date", "birth_date"]
    )
    global_params = cfg_full["global_parameters"]
    # validate global plan_rules schema
    gp_pr = global_params.get("plan_rules", {})
    try:
        valid_pr = PlanRules(**gp_pr)
    except ValidationError as e:
        print("Invalid global plan_rules configuration (monte_carlo):\n", e)
        sys.exit(1)
    global_params["plan_rules"] = valid_pr.dict()
    scenarios = cfg_full["scenarios"].copy()

    # build baseline_cfg
    baseline = scenarios.pop("baseline")
    baseline_cfg = dict(global_params)
    for k, v in baseline.items():
        if k != "plan_rules":
            baseline_cfg[k] = v

    cfg = {
        "baseline_cfg": baseline_cfg,
        "scenarios": scenarios,
        "global_params": global_params,
    }

    all_raw, all_sum = [], []
    base_seed = global_params.get("random_seed", 0)

    for rep in range(args.nreps):
        seed = base_seed + rep
        raw_df, sum_df = run_single(cfg, census_df, seed)
        all_raw.append(raw_df)
        all_sum.append(sum_df)

    full_raw = pd.concat(all_raw, ignore_index=True)
    full_sum = pd.concat(all_sum, ignore_index=True)

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    raw_out = args.outfile.replace(".csv", "_raw.parquet")

    full_raw.to_parquet(raw_out)
    full_sum.to_csv(args.outfile, index=False)

    print(f"✔ Raw employee-level saved to {raw_out}")
    print(f"✔ Summary metrics saved to    {args.outfile}")


if __name__ == "__main__":
    main()
