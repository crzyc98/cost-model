# scripts/run_plan_rules.py

"""
Script to apply plan rules for each scenario and year.
"""

#!/usr/bin/env python3
import os
import sys
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import pandas as pd
import yaml
from utils.projection_utils import apply_plan_rules
from utils.rules.validators import PlanRules, ValidationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def load_and_validate_plan_rules(raw: dict) -> dict:
    try:
        return PlanRules(**raw).dict()
    except ValidationError as e:
        logger.error("Invalid plan_rules configuration", exc_info=e)
        sys.exit(1)

def main(config_path: str, snapshots_dir: str, output_dir: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # extract global parameters and plan_rules
    global_params = cfg.get("global_parameters", {})
    # validate global plan_rules
    global_pr = load_and_validate_plan_rules(global_params.get('plan_rules', {}))
    # ensure output dir
    os.makedirs(output_dir, exist_ok=True)
    for key, sc in cfg.get('scenarios', {}).items():
        # 1) Start from global defaults (including full plan_rules)
        scen_cfg = dict(global_params)

        # 2) Override top-level fields (excluding plan_rules)
        for k, v in sc.items():
            if k != 'plan_rules':
                scen_cfg[k] = v

        # 3) Merge scenario plan_rules atop global ones
        merged = dict(global_pr)
        merged.update(sc.get('plan_rules', {}))
        scen_cfg['plan_rules'] = load_and_validate_plan_rules(merged)
        name = sc.get('scenario_name') or sc.get('name') or key
        logger.info("Applying plan rules for scenario: %s", name)
        metrics = []
        
        # --- Participation persistence logic ---
        prev_participation = None  # dict: employee_ssn -> is_participating
        prev_eligibility = None    # dict: employee_ssn -> is_eligible
        prev_ids = set()
        
        # validate snapshots directory
        if not os.path.isdir(snapshots_dir):
            logger.error("Snapshots dir not found: %s", snapshots_dir)
            sys.exit(1)
        files = [f for f in os.listdir(snapshots_dir) if f.startswith('base_run_year') and f.endswith('.parquet')]
        if not files:
            logger.error("No snapshot files in %s", snapshots_dir)
            sys.exit(1)
        for fname in sorted(files):
            if not fname.startswith("base_run_year") or not fname.endswith(".parquet"):
                continue
            year_num = int(fname[len("base_run_year"): -len(".parquet")])
            df = pd.read_parquet(os.path.join(snapshots_dir, fname))

            # If not first year, merge in prior participation and eligibility
            if prev_participation is not None and 'employee_ssn' in df:
                df = df.copy()
                df['was_participating'] = df['employee_ssn'].map(prev_participation).astype('boolean').fillna(False)
                df['was_eligible'] = df['employee_ssn'].map(prev_eligibility).astype('boolean').fillna(False)
            else:
                df['was_participating'] = df['is_participating'] if 'is_participating' in df else False
                df['was_eligible'] = df['is_eligible'] if 'is_eligible' in df else False

            out_df = apply_plan_rules(df, scen_cfg, year_num)

            # --- Participation persistence enforcement ---
            if prev_participation is not None and 'employee_ssn' in out_df:
                # Carry forward participation for continuing, eligible employees
                mask_carry = (
                    out_df['employee_ssn'].isin(prev_ids)
                    & out_df['was_participating']
                    & out_df['is_eligible']
                )
                out_df.loc[mask_carry, 'is_participating'] = True
                # Only apply auto-enrollment to those NOT previously participating
                # (If your auto-enrollment logic is in apply_plan_rules, make sure it checks was_participating)

            # Save prior participation/eligibility for next year
            if 'employee_ssn' in out_df:
                prev_participation = dict(zip(out_df['employee_ssn'], out_df['is_participating']))
                prev_eligibility = dict(zip(out_df['employee_ssn'], out_df['is_eligible']))
                prev_ids = set(out_df['employee_ssn'])

            # save detailed output per year
            out_path = os.path.join(output_dir, f"{name}_year{year_num}.parquet")
            out_df.to_parquet(out_path)
            logger.info("Wrote %s", out_path)
            # compute summary metrics
            # use original snapshot metrics to avoid FP drift
            # Use out_df (after plan rules) for metrics, matching run_monte_carlo.py
            hc = len(out_df)
            elig = int(out_df['is_eligible'].sum()) if 'is_eligible' in out_df else 0
            part = int(out_df['is_participating'].sum()) if 'is_participating' in out_df else 0
            p_rate_elig = part / elig if elig > 0 else 0.0
            p_rate_tot = part / hc if hc > 0 else 0.0

            avg_def_part = out_df.loc[out_df['is_participating'], 'deferral_rate'].mean() if 'is_participating' in out_df and 'deferral_rate' in out_df else 0.0
            avg_def_tot = out_df['deferral_rate'].mean() if 'deferral_rate' in out_df else 0.0

            emp_pre_tax = out_df['pre_tax_contributions'].sum() if 'pre_tax_contributions' in out_df else 0.0
            emp_match = out_df['employer_match_contribution'].sum() if 'employer_match_contribution' in out_df else 0.0
            emp_nec = out_df['employer_non_elective_contribution'].sum() if 'employer_non_elective_contribution' in out_df else 0.0
            emp_cost = emp_match + emp_nec

            total_contrib = emp_pre_tax + emp_match + emp_nec
            plan_comp = out_df['plan_year_compensation'].sum() if 'plan_year_compensation' in out_df else 0.0
            cap_comp = out_df['capped_compensation'].sum() if 'capped_compensation' in out_df else 0.0

            summary = {
                'scenario': name,
                'year': year_num,
                'headcount': hc,
                'eligible': elig,
                'participating': part,
                'participation_rate_eligible': p_rate_elig,
                'participation_rate_total': p_rate_tot,
                'avg_deferral_rate_participants': avg_def_part,
                'avg_deferral_rate_total': avg_def_tot,
                'total_employee_pre_tax': emp_pre_tax,
                'total_employer_match': emp_match,
                'total_employer_nec': emp_nec,
                'total_employer_cost': emp_cost,
                'total_contributions': total_contrib,
                'total_plan_year_compensation': plan_comp,
                'total_capped_compensation': cap_comp,
                'employer_cost_pct_plan_comp': emp_cost / plan_comp if plan_comp > 0 else 0.0,
                'employer_cost_pct_capped_comp': emp_cost / cap_comp if cap_comp > 0 else 0.0,
            }
            metrics.append(summary)
        # write metrics table
        metrics_df = pd.DataFrame(metrics)
        metrics_path = os.path.join(output_dir, f"{name}_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        logger.info("Wrote metrics %s", metrics_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase II: apply plan rules')
    parser.add_argument('--config', required=True, dest='config_path', help='YAML config file path')
    parser.add_argument('--snapshots-dir', '--snapshots', required=True, dest='snapshots_dir', help='Directory of HR snapshots')
    parser.add_argument('--output-dir', '--outdir', required=True, dest='output_dir', help='Output directory for plan outputs')
    args = parser.parse_args()
    main(args.config_path, args.snapshots_dir, args.output_dir)
