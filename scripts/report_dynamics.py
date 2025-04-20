#!/usr/bin/env python3
"""
Script to visualize ABM workforce dynamics:
  1. Net growth decomposition (Δ, T_ex, T_nh, H_t)
  2. Cohort counts by tenure group over time
  3. Hire vs. termination dynamics
Usage:
  python3 scripts/report_dynamics.py \
    --model_csv output/abm_onboarding_test_model_results.csv \
    --agent_csv output/abm_onboarding_test_agent_results.csv
"""
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_net_growth(df, outdir):
    df = df.copy()
    df['Hires'] = df['New Hire Active'] + df['New Hire Terminated']
    df['TotalActive'] = df['Continuous Active'] + df['New Hire Active']
    df['Delta'] = df['TotalActive'].diff().fillna(0).astype(int)
    decomp = df[['Year','Delta','Experienced Terminated','New Hire Terminated','Hires']].set_index('Year')
    decomp.plot(kind='bar', stacked=True, figsize=(8,6))
    plt.title('Net Growth Decomposition')
    plt.ylabel('Count')
    plt.tight_layout()
    out = os.path.join(outdir, 'net_growth_decomposition.png')
    plt.savefig(out)
    print(f"Saved: {out}")
    plt.close()


def plot_cohort_counts(agent_df, outdir):
    df = agent_df.copy()
    # Count agents by Year and Cohort
    cohort_counts = df.groupby(['Year','Cohort']).size().unstack(fill_value=0)
    cohort_counts.plot(figsize=(8,6))
    plt.title('Cohort Counts by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Agents')
    plt.legend(title='Cohort')
    plt.tight_layout()
    out = os.path.join(outdir, 'cohort_counts.png')
    plt.savefig(out)
    print(f"Saved: {out}")
    plt.close()


def plot_hire_termination(df, outdir):
    hires = df['New Hire Active'] + df['New Hire Terminated']
    exp_term = df['Experienced Terminated']
    new_term = df['New Hire Terminated']
    dynamics = pd.DataFrame({
        'Hires': hires.values,
        'Experienced Terminations': exp_term.values,
        'New Hire Terminations': new_term.values
    }, index=df['Year'])
    dynamics.plot(marker='o', figsize=(8,6))
    plt.title('Hire vs Termination Dynamics')
    plt.ylabel('Count')
    plt.tight_layout()
    out = os.path.join(outdir, 'hire_termination_dynamics.png')
    plt.savefig(out)
    print(f"Saved: {out}")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_csv', type=str, required=True)
    p.add_argument('--agent_csv', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='output')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_df = pd.read_csv(args.model_csv)
    agent_df = pd.read_csv(args.agent_csv)

    plot_net_growth(model_df, args.output_dir)
    plot_cohort_counts(agent_df, args.output_dir)
    plot_hire_termination(model_df, args.output_dir)

if __name__ == '__main__':
    main()
