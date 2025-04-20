#!/usr/bin/env python3
"""
Monte Carlo simulation experiments for the retirement plan ABM.

Usage:
  python3 scripts/monte_carlo_experiments.py \
    --config data/config.yaml \
    --census data/census_data.csv \
    --runs 100 \
    --output_dir output/monte_carlo

This script will:
  1. Randomly jitter key parameters (h_ex, h_nh, g) around their config values.
  2. Run the ABM for each scenario, saving per-run outputs.
  3. Aggregate final-year metrics into a summary CSV.
"""
import os
import argparse
import yaml
import random
import subprocess
import pandas as pd

def sample_config(base_cfg):
    cfg = base_cfg.copy()
    # Sample experienced attrition ±10%
    ex_mean = cfg.get('annual_termination_rate', 0.13)
    sampled_ex = max(0, random.uniform(ex_mean * 0.9, ex_mean * 1.1))
    cfg['annual_termination_rate'] = sampled_ex
    # Tie new-hire attrition to experienced attrition
    cfg['new_hire_termination_rate'] = sampled_ex * 1.6
    # Sample net growth ±10%
    g_mean = cfg.get('annual_growth_rate', 0.02)
    sampled_g = max(0, random.uniform(g_mean * 0.9, g_mean * 1.1))
    cfg['annual_growth_rate'] = sampled_g
    # Sample salary growth via normal draw and truncate
    sal_mean = cfg.get('annual_compensation_increase_rate', 0.03)
    sal_std = cfg.get('salary_growth_std', sal_mean * 0.02)
    raw_sal = random.gauss(sal_mean, sal_std)
    lower = sal_mean - 2 * sal_std
    upper = sal_mean + 2 * sal_std
    sal = max(min(raw_sal, upper), lower)
    cfg['annual_compensation_increase_rate'] = max(0, sal)
    return cfg

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--census', required=True)
    p.add_argument('--runs', type=int, default=100)
    p.add_argument('--output_dir', default='output/monte_carlo')
    args = p.parse_args()

    base_cfg = yaml.safe_load(open(args.config))
    os.makedirs(args.output_dir, exist_ok=True)
    summary = []

    for i in range(1, args.runs + 1):
        cfg_i = sample_config(base_cfg)
        cfg_path = os.path.join(args.output_dir, f'config_run_{i}.yaml')
        with open(cfg_path, 'w') as f:
            yaml.safe_dump(cfg_i, f)
        prefix = os.path.join(args.output_dir, f'run_{i}')
        # run simulation
        subprocess.run([
            'python3', 'scripts/run_retirement_plan_abm.py',
            '--config', cfg_path,
            '--census', args.census,
            '--output', prefix
        ], check=True)
        # read model output from ABM's output directory
        base_name = os.path.basename(prefix)
        model_file = os.path.join('output', f'{base_name}_model_results.csv')
        df = pd.read_csv(model_file)
        # compute total active and growth manually
        totals = df['Continuous Active'] + df['New Hire Active']
        final_total = int(totals.iloc[-1])
        if len(totals) > 1:
            growth_value = int(totals.iloc[-1] - totals.iloc[-2])
        else:
            growth_value = 0
        plan_cost = df['PlanCost'].iloc[-1] if 'PlanCost' in df.columns else None
        avg_deferral_pct = df['AvgDeferralPct'].iloc[-1] if 'AvgDeferralPct' in df.columns else None
        summary.append({
            'run': i,
            'annual_termination_rate': cfg_i['annual_termination_rate'],
            'new_hire_termination_rate': cfg_i['new_hire_termination_rate'],
            'annual_growth_rate': cfg_i['annual_growth_rate'],
            'salary_growth': cfg_i.get('annual_compensation_increase_rate'),
            'final_total_active': final_total,
            'growth': growth_value,
            'plan_cost': plan_cost,
            'avg_deferral_pct': avg_deferral_pct
        })

    summary_df = pd.DataFrame(summary)
    out_summary = os.path.join(args.output_dir, 'monte_carlo_summary.csv')
    summary_df.to_csv(out_summary, index=False)
    print(f'Saved Monte Carlo summary: {out_summary}')
