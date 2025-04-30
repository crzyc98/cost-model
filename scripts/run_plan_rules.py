#!/usr/bin/env python3

"""
Script to apply plan rules for each scenario and year.
"""

import sys
from pathlib import Path
import logging
import argparse
import pandas as pd
import yaml
from utils.columns import (
    EMP_SSN, IS_PARTICIPATING, IS_ELIGIBLE, EMP_CONTR,
    RAW_TO_STD_COLS, EMP_DEFERRAL_RATE, EMPLOYER_MATCH,
    EMPLOYER_CORE, EMP_PLAN_YEAR_COMP, EMP_CAPPED_COMP
)
from utils.projection_utils import apply_plan_rules
from utils.rules.validators import PlanRules, ValidationError

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def load_and_validate_plan_rules(raw: dict) -> PlanRules:
    try:
        return PlanRules(**raw)
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
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for key, sc in cfg.get('scenarios', {}).items():
        # 1) Start from global defaults (including full plan_rules)
        scen_cfg = dict(global_params)

        # 2) Override top-level fields (excluding plan_rules)
        for k, v in sc.items():
            if k != 'plan_rules':
                scen_cfg[k] = v

        # merge the raw plan_rules dicts and keep them as dict for apply_plan_rules
        raw_global_rules   = global_params.get('plan_rules', {}) or {}
        raw_scenario_rules = sc.get('plan_rules', {})     or {}
        merged_raw_rules   = { **raw_global_rules, **raw_scenario_rules }
        # validate schema, but store the dict
        _ = load_and_validate_plan_rules(merged_raw_rules)
        scen_cfg['plan_rules'] = merged_raw_rules
        name = sc.get('scenario_name') or sc.get('name') or key
        logger.info("Applying plan rules for scenario: %s", name)
        metrics = []
        
        # --- Participation persistence logic ---
        prev_participation = None  # dict: employee_ssn -> is_participating
        prev_eligibility = None    # dict: employee_ssn -> is_eligible
        prev_ids = set()
        
        # validate snapshots directory
        if not Path(snapshots_dir).is_dir():
            logger.error("Snapshots dir not found: %s", snapshots_dir)
            sys.exit(1)
        snapshots = sorted(Path(snapshots_dir).glob('base_run_year*.parquet'))
        if not snapshots:
            logger.error("No snapshot files in %s", snapshots_dir)
            sys.exit(1)
        for snap in snapshots:
            year_num = int(snap.stem.replace("base_run_year", ""))
            df = pd.read_parquet(snap)
            logger.debug("Raw columns before rename: %r", df.columns.tolist())
            # normalize raw column names to standardized columns
            df.rename(columns=RAW_TO_STD_COLS, inplace=True)
            logger.debug("Columns after rename:    %r", df.columns.tolist())

            # If not first year, merge in prior participation and eligibility
            if prev_participation is not None and EMP_SSN in df:
                df = df.copy()
                df['was_participating'] = df[EMP_SSN].map(prev_participation).astype('boolean').fillna(False)
                df['was_eligible']      = df[EMP_SSN].map(prev_eligibility).astype('boolean').fillna(False)
            else:
                df['was_participating'] = df[IS_PARTICIPATING] if IS_PARTICIPATING in df else False
                df['was_eligible']      = df[IS_ELIGIBLE] if IS_ELIGIBLE in df else False

            logger.info("Scenario‐level eligibility config: %r",
                        scen_cfg['plan_rules']['eligibility'])

            out_df = apply_plan_rules(df, scen_cfg, year_num)

            # --- Participation persistence enforcement ---
            if prev_participation is not None and EMP_SSN in out_df:
                # Carry forward participation for continuing, eligible employees
                mask_carry = (
                    out_df[EMP_SSN].isin(prev_ids)
                    & out_df['was_participating']
                    & out_df[IS_ELIGIBLE]
                )
                out_df.loc[mask_carry, 'is_participating'] = True
                # Only apply auto-enrollment to those NOT previously participating
                # (If your auto-enrollment logic is in apply_plan_rules, make sure it checks was_participating)

            # Save prior participation/eligibility for next year
            if EMP_SSN in out_df:
                prev_participation = dict(zip(out_df[EMP_SSN], out_df[IS_PARTICIPATING]))
                prev_eligibility = dict(zip(out_df[EMP_SSN], out_df[IS_ELIGIBLE]))
                prev_ids = set(out_df[EMP_SSN])

            # save detailed output per year
            out_path = Path(output_dir) / f"{name}_year{year_num}.parquet"
            out_df.to_parquet(out_path)
            logger.info("Wrote %s", out_path)
            # compute summary metrics
            # use original snapshot metrics to avoid FP drift
            # Use out_df (after plan rules) for metrics, matching run_monte_carlo.py
            hc = len(out_df)
            elig = int(out_df[IS_ELIGIBLE].sum()) if IS_ELIGIBLE in out_df else 0
            part = int(out_df[IS_PARTICIPATING].sum()) if IS_PARTICIPATING in out_df else 0
            p_rate_elig = part / elig if elig > 0 else 0.0
            p_rate_tot = part / hc if hc > 0 else 0.0

            avg_def_part = out_df.loc[out_df[IS_PARTICIPATING], EMP_DEFERRAL_RATE].mean() if IS_PARTICIPATING in out_df and EMP_DEFERRAL_RATE in out_df else 0.0
            avg_def_tot = out_df[EMP_DEFERRAL_RATE].mean() if EMP_DEFERRAL_RATE in out_df else 0.0

            emp_pre_tax = out_df[EMP_CONTR].sum() if EMP_CONTR in out_df else 0.0
            emp_match = out_df[EMPLOYER_MATCH].sum() if EMPLOYER_MATCH in out_df else 0.0
            emp_nec = out_df[EMPLOYER_CORE].sum() if EMPLOYER_CORE in out_df else 0.0
            emp_cost = emp_match + emp_nec

            total_contrib = emp_pre_tax + emp_match + emp_nec
            plan_comp = out_df[EMP_PLAN_YEAR_COMP].sum() if EMP_PLAN_YEAR_COMP in out_df else 0.0
            cap_comp = out_df[EMP_CAPPED_COMP].sum() if EMP_CAPPED_COMP in out_df else 0.0

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
        metrics_path = Path(output_dir) / f"{name}_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info("Wrote metrics %s", metrics_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase II: apply plan rules')
    parser.add_argument('--config', required=True, dest='config_path', help='YAML config file path')
    parser.add_argument('--snapshots-dir', '--snapshots', required=True, dest='snapshots_dir', help='Directory of HR snapshots')
    parser.add_argument('--output-dir', '--outdir', required=True, dest='output_dir', help='Output directory for plan outputs')
    args = parser.parse_args()
    main(args.config_path, args.snapshots_dir, args.output_dir)
